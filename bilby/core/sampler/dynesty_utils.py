import csv
import os
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
from dynesty.internal_samplers import InternalSampler, SamplerReturn
from dynesty.utils import SamplerHistoryItem, apply_reflect, get_random_generator
from scipy.stats import binom

from ...bilby_mcmc.chain import calculate_tau
from ..utils.log import logger

EnsembleSamplerArgument = namedtuple(
    "EnsembleSamplerArgument",
    [
        "u",
        "loglstar",
        "live_points",
        "prior_transform",
        "loglikelihood",
        "rseed",
        "kwargs",
    ],
)
EnsembleAxisSamplerArgument = namedtuple(
    "EnsembleAxisSamplerArgument",
    [
        "u",
        "loglstar",
        "axes",
        "live_points",
        "prior_transform",
        "loglikelihood",
        "rseed",
        "kwargs",
    ],
)


# ---------------------------------------------------------------------------
# Autocorrelation-time diagnostics
#
# When the environment variable BILBY_ACT_STATS_FILE is set, the
# acceptance-walk and act-walk samplers append one CSV row per MCMC chain
# (acceptance-walk) or per cache build (act-walk).  For example
#
#     BILBY_ACT_STATS_FILE=/path/to/event00_act_stats.csv
#
# writes to /path/to/event00_act_stats_<sample>_pid<PID>.csv.  Writing one
# file per process avoids races between multiprocessing workers; concatenate
# the files after the run.  Set BILBY_ACT_STATS_SESSION (e.g. to a job id) to
# add a _run<SESSION> suffix so that restarts do not append to old files.
#
# The diagnostics use no random numbers and no likelihood calls, so the
# sampling is identical with and without them.  Failures while recording are
# logged and never interrupt the sampling.
# ---------------------------------------------------------------------------

_ACT_STATS_FILE_ENV = "BILBY_ACT_STATS_FILE"
_ACT_STATS_SESSION_ENV = "BILBY_ACT_STATS_SESSION"

# Module-level per-process state.  Sampler instances are pickled for every
# pool.map call, so instance attributes would be reset in the workers.
_act_stats_counters = dict()
_act_stats_failed = set()


def _act_stats_enabled():
    return bool(os.environ.get(_ACT_STATS_FILE_ENV))


def _act_components(accept, iteration, samples, most_failures):
    """
    Return the conservative ACT used by act-walk together with its components.

    Returns
    -------
    total, full, naive, maxfail
        :code:`total = max(full, naive, maxfail)`, with the same value (and
        evaluation order) as the original :code:`_calculate_act`.
    """
    if accept > 0:
        full_act = calculate_tau(samples)
        naive_act = 2 / accept * iteration - 1
        total = max(full_act, naive_act, most_failures)
        return total, float(full_act), float(naive_act), float(most_failures)
    return np.inf, np.nan, np.inf, float(most_failures)


def _per_dimension_act(samples, autocorr_c=5):
    """
    Integrated autocorrelation time of each unit-cube coordinate.

    :code:`samples` has shape (n_step, n_dim) and is treated as a single chain
    per coordinate.  This differs from :code:`calculate_tau`, which treats the
    coordinates as walkers and averages their autocorrelation functions, so
    that the full ACT can be shorter than the slowest coordinate.

    Constant or non-finite coordinates and failed estimates are returned as
    inf.  :code:`tol=1` only requires the chain to be longer than tau, so
    estimates from short chains are biased low.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2:
        return np.array([])
    n_step, n_dim = samples.shape
    result = np.full(n_dim, np.inf)
    if n_step < 2:
        return result

    valid = np.all(np.isfinite(samples), axis=0)
    valid &= np.any(samples != samples[0], axis=0)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return result

    from emcee.autocorr import AutocorrError, integrated_time

    errors = (AutocorrError, ValueError, FloatingPointError)

    def estimate(x):
        # shape (n_step, one walker, n_param) works for every emcee 3 release
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return np.asarray(
                integrated_time(x[:, None, :], c=autocorr_c, tol=1), dtype=float
            ).reshape(-1)

    try:
        taus = dict(zip(indices, estimate(samples[:, indices])))
    except errors:
        # One badly measured coordinate makes the vector call fail, retry
        # coordinate by coordinate so the others are still recorded.
        taus = dict()
        for index in indices:
            try:
                taus[index] = estimate(samples[:, index : index + 1])[0]
            except errors:
                pass
    for index, tau in taus.items():
        if np.isfinite(tau) and tau > 0:
            result[index] = tau
    return result


def _act_stats_filename(sample, schema_index=1):
    base = Path(os.environ[_ACT_STATS_FILE_ENV])
    stem = base.stem if base.suffix.lower() == ".csv" else base.name
    name = f"{stem}_{sample}_pid{os.getpid()}"
    session = os.environ.get(_ACT_STATS_SESSION_ENV, "")
    if session:
        name += f"_run{session}"
    if schema_index > 1:
        name += f"_schema{schema_index}"
    return base.with_name(f"{name}.csv")


def _write_act_stats(sample, record):
    """
    Append a row, never below a different header: if an existing file (e.g.
    from a reused PID) has other columns, use a _schema<N> suffix instead.
    """
    fieldnames = list(record)
    schema_index = 1
    while True:
        filename = _act_stats_filename(sample, schema_index)
        if not filename.exists() or filename.stat().st_size == 0:
            write_header = True
            break
        with filename.open(newline="", encoding="utf-8") as file:
            if next(csv.reader(file), []) == fieldnames:
                write_header = False
                break
        schema_index += 1
    filename.parent.mkdir(parents=True, exist_ok=True)
    with filename.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def _record_act_stats(sample, samples, loglstar, act_components=None, **fields):
    """
    Record ACT diagnostics of one MCMC chain.  Never raises.

    Parameters
    ----------
    sample: str
        Name of the sampling method, used in the file name.
    samples: array_like, shape (n_step, n_dim)
        Unit-cube state after every proposal, including repeated states after
        rejected or out-of-bounds proposals.
    loglstar: float
        The likelihood constraint.
    act_components: tuple, optional
        :code:`(total, full, naive, maxfail)` if already computed, otherwise
        they are computed from :code:`fields["accept"]`,
        :code:`fields["iteration"]` and :code:`fields["max_consecutive_fail"]`.
    fields:
        Further columns, written in the given order.
    """
    if not _act_stats_enabled():
        return
    key = (os.getpid(), sample)
    try:
        record_id = _act_stats_counters.get(key, 0) + 1
        _act_stats_counters[key] = record_id
        samples = np.asarray(samples, dtype=float)
        iteration = fields["iteration"]
        if act_components is None:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                act_components = _act_components(
                    accept=fields["accept"],
                    iteration=iteration,
                    samples=samples,
                    most_failures=fields["max_consecutive_fail"],
                )
        total, full, naive, maxfail = (float(value) for value in act_components)
        per_dim = _per_dimension_act(samples)
        finite = per_dim[np.isfinite(per_dim)]
        dim_max = float(finite.max()) if finite.size else np.nan

        record = dict(
            pid=key[0],
            run_id=os.environ.get(_ACT_STATS_SESSION_ENV, ""),
            record_id=record_id,
            sample=sample,
            loglstar=float(loglstar),
            act_total=total,
            act_full=full,
            act_naive=naive,
            act_maxfail=maxfail,
            achieved_nact=(
                iteration / total if np.isfinite(total) and total > 0 else np.nan
            ),
        )
        record.update(fields)
        record.update(
            act_dim_max=dim_max,
            act_dim_nfail=int(per_dim.size - finite.size),
            chain_over_act_dim_max=iteration / dim_max if finite.size else np.nan,
        )
        record.update(
            {f"act_dim_{index:02d}": float(tau) for index, tau in enumerate(per_dim)}
        )
        _write_act_stats(sample, record)
    except Exception:
        if key in _act_stats_failed:
            logger.debug("Could not record ACT diagnostics", exc_info=True)
        else:
            _act_stats_failed.add(key)
            logger.exception(
                "Could not record ACT diagnostics; further failures in this "
                "process are logged at DEBUG level"
            )


class BaseEnsembleSampler(InternalSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ncdim = kwargs.get("ncdim")
        self.sampler_kwargs["ncdim"] = self.ncdim
        self.sampler_kwargs["proposals"] = kwargs.get("proposals", ["diff"])

    def prepare_sampler(
        self,
        loglstar=None,
        points=None,
        axes=None,
        seeds=None,
        prior_transform=None,
        loglikelihood=None,
        nested_sampler=None,
    ):
        """
        Prepare the list of arguments for sampling.

        Parameters
        ----------
        loglstar : float
            Ln(likelihood) bound.
        points : `~numpy.ndarray` with shape (n, ndim)
            Initial sample points.
        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points.
        seeds : `~numpy.ndarray` with shape (n,)
            Random number generator seeds.
        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.
        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.
        nested_sampler : `~dynesty.samplers.Sampler`
            The nested sampler object used to sample.

        Returns
        -------
        arglist:
            List of `SamplerArgument` objects containing the parameters
            needed for sampling.
        """
        arg_list = []
        kwargs = self.sampler_kwargs.copy()
        self.nlive = nested_sampler.nlive
        for curp, curaxes, curseed in zip(points, axes, seeds):
            vals = dict(
                u=curp,
                loglstar=loglstar,
                live_points=nested_sampler.live_u,
                prior_transform=prior_transform,
                loglikelihood=loglikelihood,
                rseed=curseed,
                kwargs=kwargs,
            )
            if "volumetric" in kwargs["proposals"]:
                vals["axes"] = curaxes
                curarg = EnsembleAxisSamplerArgument(**vals)
            else:
                curarg = EnsembleSamplerArgument(**vals)
            arg_list.append(curarg)
        return arg_list


class EnsembleWalkSampler(BaseEnsembleSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.walks = max(2, kwargs.get("walks", 25))
        self.sampler_kwargs["walks"] = self.walks
        self.naccept = kwargs.get("naccept", 10)
        self.maxmcmc = kwargs.get("maxmcmc", 5000)
        self.sampler_kwargs["target"] = self.naccept
        self.sampler_kwargs["maxmcmc"] = self.maxmcmc

    def tune(self, tuning_info, update=True):
        """
        Update the proposal parameters based on the number of accepted steps
        and MCMC chain length.

        The :code:`walks` parameter to asymptotically approaches the desired number
        of accepted steps. Update :code:`self.scale` for inclusion in the state plot.
        """
        # update walks to match target naccept
        nsteps = tuning_info["accept"] + tuning_info["reject"]
        accept_prob = max(0.5, tuning_info["accept"]) / nsteps
        delay = max(self.nlive // 10 - 1, 0)
        self.scale = tuning_info["accept"]
        self.walks = (self.walks * delay + self.naccept / accept_prob) / (delay + 1)
        self.sampler_kwargs["walks"] = min(int(np.ceil(self.walks)), self.maxmcmc)

    @staticmethod
    def sample(args):
        """
        Return a new live point proposed by random walking away from an
        existing live point.

        Parameters
        ----------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the initial sample. **This is a copy of an existing
            live point.**

        loglstar : float
            Ln(likelihood) bound.

        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points. For random walks new positions are
            proposed using the :class:`~dynesty.bounding.Ellipsoid` whose
            shape is defined by axes.

        scale : float
            Value used to scale the provided axes.

        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.

        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.

        kwargs : dict
            A dictionary of additional method-specific parameters.

        Returns
        -------
        u : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point within the unit cube.

        v : `~numpy.ndarray` with shape (ndim,)
            Position of the final proposed point in the target parameter space.

        logl : float
            Ln(likelihood) of the final proposed point.

        nc : int
            Number of function calls used to generate the sample.

        sampling_info : dict
            Collection of ancillary quantities used to tune :data:`scale`.

        """
        current_u = args.u
        naccept = 0
        ncall = 0

        periodic = args.kwargs["periodic"]
        reflective = args.kwargs["reflective"]
        boundary_kwargs = dict(periodic=periodic, reflective=reflective)

        proposals, common_kwargs, proposal_kwargs = _get_proposal_kwargs(args)
        walks = len(proposals)
        evaluation_history = list()

        # ACT diagnostics: chain state after every proposal (see
        # _record_act_stats).  The fallback random point is not part of it.
        record_act = _act_stats_enabled()
        chain = np.empty((walks, len(current_u))) if record_act else None
        boundary_fail = 0
        likelihood_reject = 0
        current_failures = 0
        most_failures = 0

        for index, prop in enumerate(proposals):
            u_prop = proposal_funcs[prop](
                u=current_u, **common_kwargs, **proposal_kwargs[prop]
            )
            u_prop = apply_boundaries_(u_prop=u_prop, **boundary_kwargs)
            if u_prop is None:
                boundary_fail += 1
                current_failures += 1
                most_failures = max(most_failures, current_failures)
                if record_act:
                    chain[index] = current_u
                continue

            v_prop = args.prior_transform(u_prop)
            logl_prop = args.loglikelihood(v_prop)
            evaluation_history.append(
                SamplerHistoryItem(u=v_prop, v=u_prop, logl=logl_prop)
            )
            ncall += 1

            if logl_prop > args.loglstar:
                current_u = u_prop
                current_v = v_prop
                logl = logl_prop
                naccept += 1
                current_failures = 0
            else:
                likelihood_reject += 1
                current_failures += 1
                most_failures = max(most_failures, current_failures)
            if record_act:
                chain[index] = current_u

        fallback = naccept == 0
        if naccept == 0:
            logger.debug(
                "Unable to find a new point using walk: returning a random point. "
                "If this warning occurs often, increase naccept."
            )
            # Technically we can find out the likelihood value
            # stored somewhere
            # But I'm currently recomputing it
            current_u = common_kwargs["rstate"].uniform(0, 1, len(current_u))
            current_v = args.prior_transform(current_u)
            logl = args.loglikelihood(current_v)

        sampling_info = {
            "accept": naccept,
            "reject": walks - naccept,
        }

        # check that the number of accepted steps is consistent with the target
        # number of accepted steps, if it is much lower than the target, our
        # mcmc chain is too short and we may be open to bias. If we're hitting
        # maxmcmc, this is expected, but it also happens at the beginning of the
        # run or around phase transitions, so we discard this point and so an
        # equivalent longer mcmc will happen at the next iteration
        dist = binom(n=walks, p=args.kwargs["target"] / walks)
        minimum = dist.ppf(1e-4)
        undersampled = naccept < minimum
        if undersampled and walks >= args.kwargs["maxmcmc"]:
            logger.debug(
                "Requested %s accepted steps per MCMC chain, "
                "but only %s were accepted out of the maximum allowed number "
                "of steps (%s). Increase 'maxmcmc' to improve convergence.",
                args.kwargs["target"],
                naccept,
                walks,
            )
        elif undersampled:
            logger.info(
                "Only %s accepted steps out of target %s. "
                "Discarding this point. "
                "This is expected at the beginning of the run, increase 'walks'. "
                "This can also happen when the geometry of the likelihood surface "
                "changes drastically.",
                naccept,
                args.kwargs["target"],
            )
            logl = args.loglstar - 1

        if record_act:
            # .get: the diagnostics must not fail where the sampling itself does not
            # (the sampling reads maxmcmc only for undersampled chains).
            maxmcmc = args.kwargs.get("maxmcmc", np.inf)
            _record_act_stats(
                "acceptance-walk",
                samples=chain,
                loglstar=args.loglstar,
                iteration=walks,
                accept=naccept,
                target_accept=args.kwargs["target"],
                ncall=ncall,
                likelihood_evaluations=ncall + int(fallback),
                boundary_fail=boundary_fail,
                likelihood_reject=likelihood_reject,
                max_consecutive_fail=most_failures,
                accept_fraction=naccept / walks,
                evaluated_fraction=ncall / walks,
                maxmcmc=maxmcmc,
                hit_maxmcmc=walks >= maxmcmc,
                fallback_random_point=fallback,
                discarded_undersampled=bool(undersampled and walks < maxmcmc),
            )

        return SamplerReturn(
            u=current_u,
            v=current_v,
            logl=logl,
            tuning_info=sampling_info,
            ncalls=ncall,
            proposal_stats=sampling_info,
            evaluation_history=evaluation_history,
        )

        # return current_u, current_v, logl, ncall, sampling_info


class ACTTrackingEnsembleWalk(BaseEnsembleSampler):
    """
    Run the MCMC sampler for many iterations in order to reliably estimate
    the autocorrelation time.

    This builds a cache of :code:`nact / thin` points consistent with the
    likelihood constraint.
    While this approach is the most robust, it is not well optimized for
    parallelised sampling as the length of the MCMC will be different for each
    parallel process.

    When the environment variable :code:`BILBY_ACT_STATS_FILE` is set, one CSV
    row is written per cache build with the three ACT components, the
    per-coordinate ACTs of the unthinned chain (:code:`act_dim_XX`), the chain
    length, likelihood-call counts and the thinning. These diagnostics never
    change the ACT used for thinning.
    """

    # the _cache is a class level variable to avoid being forgotten at every
    # iteration when using multiprocessing
    _cache = list()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.act = 1
        self.thin = kwargs.get("nact", 2)
        self.maxmcmc = kwargs.get("maxmcmc", 5000) * 50
        self.sampler_kwargs["rebuild"] = True
        self.sampler_kwargs["thin"] = self.thin
        self.sampler_kwargs["act"] = self.act
        self.sampler_kwargs["maxmcmc"] = self.maxmcmc
        # reset the cache at instantiation to avoid contamination from
        # previous analyses
        self.__class__._cache = list()

    def prepare_sampler(
        self,
        loglstar=None,
        points=None,
        axes=None,
        seeds=None,
        prior_transform=None,
        loglikelihood=None,
        nested_sampler=None,
    ):
        """
        Prepare the list of arguments for sampling.

        Parameters
        ----------
        loglstar : float
            Ln(likelihood) bound.
        points : `~numpy.ndarray` with shape (n, ndim)
            Initial sample points.
        axes : `~numpy.ndarray` with shape (ndim, ndim)
            Axes used to propose new points.
        seeds : `~numpy.ndarray` with shape (n,)
            Random number generator seeds.
        prior_transform : function
            Function transforming a sample from the a unit cube to the
            parameter space of interest according to the prior.
        loglikelihood : function
            Function returning ln(likelihood) given parameters as a 1-d
            `~numpy` array of length `ndim`.
        nested_sampler : `~dynesty.samplers.Sampler`
            The nested sampler object used to sample.

        Returns
        -------
        arglist:
            List of `SamplerArgument` objects containing the parameters
            needed for sampling.
        """
        arg_list = super().prepare_sampler(
            loglstar=loglstar,
            points=points,
            axes=axes,
            seeds=seeds,
            prior_transform=prior_transform,
            loglikelihood=loglikelihood,
            nested_sampler=nested_sampler,
        )
        self.sampler_kwargs["rebuild"] = False
        return arg_list

    def tune(self, tuning_info, update=True):
        """
        Update the proposal parameters based on the number of accepted steps
        and MCMC chain length.

        The :code:`walks` parameter to asymptotically approach the
        desired number of accepted steps.
        """
        if tuning_info.get("remaining", 0) == 0:
            self.sampler_kwargs["rebuild"] = True
        self.scale = tuning_info["accept"]
        self.sampler_kwargs["act"] = tuning_info["act"]

    @staticmethod
    def sample(args):
        cache = ACTTrackingEnsembleWalk._cache
        if args.kwargs.get("rebuild", False):
            logger.debug(f"Force rebuilding cache with {len(cache)}.")
            cache.clear()
        if len(cache) == 0:
            ACTTrackingEnsembleWalk.build_cache(args)

        while len(cache) > 0 and cache[0][2] < args.loglstar:
            state = cache.pop(0)

        if len(cache) == 0:
            current_u, current_v, logl, ncall, blob, evaluation_history = state
        else:
            current_u, current_v, logl, ncall, blob, evaluation_history = cache.pop(0)

        blob["remaining"] = len(cache)
        logger.debug(f"Returning point from cache with {len(cache)} remaining")
        return SamplerReturn(
            u=current_u,
            v=current_v,
            logl=logl,
            tuning_info=blob,
            ncalls=ncall,
            proposal_stats=blob,
            evaluation_history=evaluation_history,
        )

    @staticmethod
    def build_cache(args):
        # Bounds
        periodic = args.kwargs.get("periodic", None)
        reflective = args.kwargs.get("reflective", None)
        boundary_kwargs = dict(periodic=periodic, reflective=reflective)

        proposals, common_kwargs, proposal_kwargs = _get_proposal_kwargs(args)

        # Setup
        current_u = args.u
        old_act = args.kwargs.get("act", 100)
        check_interval = ACTTrackingEnsembleWalk.integer_act(old_act)
        target_nact = 50
        next_check = check_interval
        n_checks = 0
        maxmcmc = args.kwargs.get("maxmcmc", 5000)
        evaluation_history = list()

        # Initialize internal variables
        current_v = args.prior_transform(np.array(current_u))
        logl = args.loglikelihood(np.array(current_v))
        evaluation_history.append(
            SamplerHistoryItem(u=current_u, v=current_v, logl=logl)
        )
        accept = 0
        reject = 0
        nfail = 0
        ncall = 0
        boundary_fail = 0
        likelihood_reject = 0
        act = np.inf
        u_list = list()
        v_list = list()
        logl_list = list()
        most_failures = 0
        current_failures = 0

        iteration = 0
        while iteration < min(target_nact * act, maxmcmc):
            iteration += 1

            prop = proposals[iteration % len(proposals)]
            u_prop = proposal_funcs[prop](
                u=current_u, **common_kwargs, **proposal_kwargs[prop]
            )
            u_prop = apply_boundaries_(u_prop=u_prop, **boundary_kwargs)
            success = False
            if u_prop is None:
                boundary_fail += 1
            else:
                v_prop = args.prior_transform(np.array(u_prop))
                logl_prop = args.loglikelihood(np.array(v_prop))
                evaluation_history.append(
                    SamplerHistoryItem(u=v_prop, v=u_prop, logl=logl_prop)
                )
                ncall += 1
                if logl_prop > args.loglstar:
                    success = True
                    current_u = u_prop
                    current_v = v_prop
                    logl = logl_prop
                else:
                    likelihood_reject += 1

            u_list.append(current_u)
            v_list.append(current_v)
            logl_list.append(logl)

            if success:
                accept += 1
                most_failures = max(most_failures, current_failures)
                current_failures = 0
            else:
                nfail += 1
                current_failures += 1

            # If we've taken the minimum number of steps, calculate the ACT
            if iteration > next_check and accept > target_nact:
                n_checks += 1
                most_failures = max(most_failures, current_failures)
                act = ACTTrackingEnsembleWalk._calculate_act(
                    accept=accept,
                    iteration=iteration,
                    samples=np.array(u_list),
                    most_failures=most_failures,
                )
                to_next_update = (act * target_nact - iteration) // 2
                to_next_update = max(to_next_update, iteration // 100)
                next_check += to_next_update
                logger.debug(
                    f"it: {iteration}, accept: {accept}, reject: {reject}, "
                    f"fail: {nfail}, act: {act:.2f}, nact: {iteration / act:.2f} "
                )
            elif iteration > next_check:
                next_check += check_interval

        most_failures = max(most_failures, current_failures)
        samples = np.array(u_list)
        act_components = _act_components(
            accept=accept,
            iteration=iteration,
            samples=samples,
            most_failures=most_failures,
        )
        act = act_components[0]
        # build-level counters for the diagnostics, before division by n_found
        build_accept = accept
        build_ncall = ncall
        reject += nfail
        blob = {"accept": accept, "reject": reject, "act": act}
        iact = ACTTrackingEnsembleWalk.integer_act(act)
        thin = args.kwargs.get("thin", 2) * iact

        cache = ACTTrackingEnsembleWalk._cache
        n_found = 1

        if accept == 0:
            outcome = "random_point"
            logger.warning(
                "Unable to find a new point using walk: returning a random point"
            )
            u = common_kwargs["rstate"].uniform(size=len(current_u))
            v = args.prior_transform(u)
            logl = args.loglikelihood(v)
            evaluation_history = [
                SamplerHistoryItem(u=current_u, v=current_v, logl=logl)
            ]
            cache.append((u, v, logl, ncall, blob, evaluation_history))
        elif not np.isfinite(act):
            outcome = "act_not_finite"
            logger.warning(
                "Unable to find a new point using walk: try increasing maxmcmc"
            )
            cache.append((current_u, current_v, logl, ncall, blob, evaluation_history))
        elif (thin == -1) or (len(u_list) <= thin):
            outcome = "not_thinned"
            cache.append((current_u, current_v, logl, ncall, blob, evaluation_history))
        else:
            outcome = "thinned"
            u_list = u_list[thin::thin]
            v_list = v_list[thin::thin]
            logl_list = logl_list[thin::thin]
            evaluation_history_list = (
                evaluation_history[thin * ii : thin * (ii + 1)]
                for ii in range(len(u_list))
            )
            n_found = len(u_list)
            accept = max(accept // n_found, 1)
            reject //= n_found
            nfail //= n_found
            ncall_list = [ncall // n_found] * n_found
            blob_list = [
                dict(accept=accept, reject=reject, fail=nfail, act=act)
            ] * n_found
            cache.extend(
                zip(
                    u_list,
                    v_list,
                    logl_list,
                    ncall_list,
                    blob_list,
                    evaluation_history_list,
                )
            )
            logger.debug(
                f"act: {act:.2f}, max failures: {most_failures}, thin: {thin}, "
                f"iteration: {iteration}, n_found: {n_found}"
            )
        logger.debug(
            f"Finished building cache with length {len(cache)} after "
            f"{iteration} iterations with {ncall} likelihood calls and ACT={act:.2f}"
        )

        if _act_stats_enabled():
            _record_act_stats(
                "act-walk",
                samples=samples,
                loglstar=args.loglstar,
                act_components=act_components,
                iteration=iteration,
                accept=build_accept,
                target_nact=target_nact,
                previous_act=old_act,
                n_checks=n_checks,
                ncall=build_ncall,
                boundary_fail=boundary_fail,
                likelihood_reject=likelihood_reject,
                max_consecutive_fail=most_failures,
                accept_fraction=build_accept / iteration if iteration else np.nan,
                evaluated_fraction=build_ncall / iteration if iteration else np.nan,
                thin_by_nact=args.kwargs.get("thin", 2),
                thin=thin,
                n_found=n_found,
                outcome=outcome,
                cache_len_after=len(cache),
                maxmcmc=maxmcmc,
                hit_maxmcmc=iteration >= maxmcmc,
            )

    @staticmethod
    def _calculate_act(accept, iteration, samples, most_failures):
        """
        Take the maximum of three ACT estimates, leading to a conservative estimate:

        - a full ACT estimate as done in :code:`bilby_mcmc`. This is almost always the
          longest estimator, and is the most computationally expensive. The other
          methods mostly catch cases where the estimated ACT is very small.
        - the naive ACT used for the acceptance tracking walk.
        - the most failed proposals between any pair of accepted steps. This is a strong
          lower bound, because we know that if we thin by less than this, there will be
          duplicate points.

        See :code:`_act_components` for the individual estimates.
        """
        return _act_components(accept, iteration, samples, most_failures)[0]

    @staticmethod
    def integer_act(act):
        if np.isinf(act):
            return act
        else:
            return int(np.ceil(act))


class AcceptanceTrackingRWalk(EnsembleWalkSampler):
    """
    This is a modified version of dynesty.sampling.sample_rwalk that runs the
    MCMC random walk for a user-specified number of a crude approximation to
    the autocorrelation time.

    This is the proposal method used by default for :code:`Bilby<2` and
    corresponds to specifying :code:`sample="rwalk"`
    """

    # to retain state between calls to pool.Map, this needs to be a class
    # level attribute
    old_act = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nact = kwargs.get("nact", 40)
        self.sampler_kwargs["nact"] = self.nact
        self.sampler_kwargs["maxmcmc"] = self.maxmcmc

    def tune(self, tuning_info, update=True):
        """
        The tuning all happens inside the MCMC for this proposal
        so all we do is extract the number of accepted steps for plotting.
        """
        self.scale = tuning_info["accept"]

    @staticmethod
    def sample(args):
        rstate = get_random_generator(args.rseed)

        periodic = args.kwargs.get("periodic", None)
        reflective = args.kwargs.get("reflective", None)
        boundary_kwargs = dict(periodic=periodic, reflective=reflective)

        current_u = args.u
        nlive = args.kwargs.get("nlive", args.kwargs.get("walks", 100))

        proposals, common_kwargs, proposal_kwargs = _get_proposal_kwargs(args)

        accept = 0
        reject = 0
        nfail = 0
        act = np.inf
        nact = args.kwargs.get("nact", 40)
        maxmcmc = args.kwargs.get("maxmcmc", 5000)
        evaluation_history = list()

        iteration = 0
        while iteration < nact * act:
            iteration += 1

            prop = proposals[iteration % len(proposals)]
            u_prop = proposal_funcs[prop](
                current_u, **common_kwargs, **proposal_kwargs[prop]
            )
            u_prop = apply_boundaries_(u_prop, **boundary_kwargs)

            if u_prop is None:
                nfail += 1
                continue

            # Check proposed point.
            v_prop = args.prior_transform(np.array(u_prop))
            logl_prop = args.loglikelihood(np.array(v_prop))
            evaluation_history.append(
                SamplerHistoryItem(u=v_prop, v=u_prop, logl=logl_prop)
            )
            if logl_prop > args.loglstar:
                current_u = u_prop
                current_v = v_prop
                logl = logl_prop
                accept += 1
            else:
                reject += 1

            # If we've taken the minimum number of steps, calculate the ACT
            if iteration > nact:
                act = AcceptanceTrackingRWalk.estimate_nmcmc(
                    accept_ratio=accept / (accept + reject + nfail),
                    safety=1,
                    tau=nlive,
                    maxmcmc=maxmcmc,
                    old_act=AcceptanceTrackingRWalk.old_act,
                )

            # If we've taken too many likelihood evaluations then break
            if accept + reject > maxmcmc:
                warnings.warn(
                    f"Hit maximum number of walks {maxmcmc} with accept={accept},"
                    f" reject={reject}, and nfail={nfail} try increasing maxmcmc"
                )
                break

        if not (np.isfinite(act) and accept > 0):
            logger.debug(
                "Unable to find a new point using walk: returning a random point"
            )
            current_u = rstate.uniform(size=len(current_u))
            current_v = args.prior_transform(current_u)
            logl = args.loglikelihood(current_v)

        sampling_info = {"accept": accept, "reject": reject + nfail}
        AcceptanceTrackingRWalk.old_act = act

        ncall = accept + reject
        return SamplerReturn(
            u=current_u,
            v=current_v,
            logl=logl,
            tuning_info=sampling_info,
            ncalls=ncall,
            proposal_stats=sampling_info,
            evaluation_history=evaluation_history,
        )

    @staticmethod
    def estimate_nmcmc(accept_ratio, safety=5, tau=None, maxmcmc=5000, old_act=None):
        """Estimate autocorrelation length of chain using acceptance fraction

        Using ACL = (2/acc) - 1 multiplied by a safety margin. Code adapted from:

        - https://github.com/farr/Ensemble.jl
        - https://github.com/johnveitch/cpnest/blob/master/cpnest/sampler.py

        Parameters
        ==========
        accept_ratio: float [0, 1]
            Ratio of the number of accepted points to the total number of points
        old_act: int
            The ACT of the last iteration
        maxmcmc: int
            The maximum length of the MCMC chain to use
        safety: int
            A safety factor applied in the calculation
        tau: int (optional)
            The ACT, if given, otherwise estimated.

        Notes
        =====
        This method does not compute a reliable estimate of the autocorrelation
        length for our proposal distributions.
        """
        if tau is None:
            tau = maxmcmc / safety

        if accept_ratio == 0.0:
            if old_act is None:
                Nmcmc_exact = np.inf
            else:
                Nmcmc_exact = (1 + 1 / tau) * old_act
        else:
            estimated_act = 2 / accept_ratio - 1
            Nmcmc_exact = safety * estimated_act
            if old_act is not None:
                Nmcmc_exact = (1 - 1 / tau) * old_act + Nmcmc_exact / tau
        Nmcmc_exact = float(min(Nmcmc_exact, maxmcmc))
        return max(safety, Nmcmc_exact)


def _get_proposal_kwargs(args):
    """
    Resolve the proposal cycle from the provided keyword arguments.

    The steps involved are:

    - extract the requested proposal types from the kwargs passed through from dynesty.
      If none are specified, only differential evolution will be used.
    - differential evolution requires the live points to be passed. If they are
      not present, raise an error.
    - if a dictionary, e.g., :code:`dict(diff=5, volumetric=1)` is specified,
      the keys will be used weighted by the values, e.g., 5:1 differential
      evolution to volumetric.
    - each proposal needs different keyword arguments, see the specific functions
      for what requires which.


    Parameters
    ==========
    args: dynesty.sampler.SamplerArgument
        Object that carries around various pieces of information about the
        analysis.
    """
    rstate = get_random_generator(args.rseed)
    walks = args.kwargs.get("walks", 100)
    current_u = args.u

    proposals = args.kwargs.get("proposals", None)
    if proposals is None:
        proposals = ["diff"]

    n_cluster = args.kwargs.get("ncdim", None)
    if n_cluster is None:
        if hasattr(args, "live_points"):
            n_cluster = args.live_points.shape[1]
        elif hasattr(args, "axes"):
            n_cluster = args.axes.shape[0]

    if "diff" in proposals:
        live = args.live_points
        if live is None:
            raise ValueError(
                "Live points not passed for differential evolution, specify "
                "bound='live' to use differential evolution proposals."
            )
        live = np.unique(live, axis=0)
        matches = np.where(np.equal(current_u, live).all(axis=1))[0]
        np.delete(live, matches, 0)

    if isinstance(proposals, (list, set, tuple)):
        proposals = rstate.choice(proposals, int(walks))
    elif isinstance(proposals, dict):
        props, weights = zip(*proposals.items())
        weights = np.array(weights) / sum(weights)
        proposals = rstate.choice(list(props), int(walks), p=weights)

    common_kwargs = dict(
        n=len(current_u),
        n_cluster=n_cluster,
        rstate=rstate,
    )
    proposal_kwargs = dict()
    if "diff" in proposals:
        proposal_kwargs["diff"] = dict(
            live=live[:, :n_cluster],
            mix=0.5,
            scale=2.38 / (2 * n_cluster) ** 0.5,
        )
    if "volumetric" in proposals:
        proposal_kwargs["volumetric"] = dict(
            axes=args.axes,
            scale=args.scale,
        )
    return proposals, common_kwargs, proposal_kwargs


def propose_differential_evolution(
    u,
    live,
    n,
    n_cluster,
    rstate,
    mix=0.5,
    scale=1,
):
    r"""
    Propose a new point using ensemble differential evolution
    (`ter Braak + (2006) <https://doi.org/10.1007/s11222-006-8769-1>`_).

    .. math::

        u_{\rm prop} = u + \gamma (v_{a} - v_{b})

    We consider two choices for :math:`\gamma`: weighted by :code:`mix`.

    - :math:`\gamma = 1`: this is a mode-hopping mode for efficiently
      exploring multi-modal spaces
    - :math:`\gamma \sim \Gamma\left(\gamma; k=4, \theta=\frac{\kappa}{4}\right)`

    Here :math:`\kappa = 2.38 / \sqrt{2 n}` unless specified by the user and
    we scale by a random draw from a Gamma distribution. The specific
    distribution was chosen somewhat arbitrarily to have mean and mode close to
    :math:`\kappa` and give good acceptance and autocorrelation times on a subset
    of problems.

    Parameters
    ----------
    u: np.ndarray
        The current point.
    live: np.ndarray
        The ensemble of live points to select :math:`v` from.
    n: int
        The number of dimensions being explored
    n_cluster: int
        The number of dimensions to run the differential evolution over, the
        first :code:`n_cluster` dimensions are used. The rest are randomly
        sampled from the prior.
    rstate: numpy.random.Generator
        The numpy generator instance used for random number generation.
        Consider using built in `bilby.core.utils.random.rng`.
    mix: float
        The fraction of proposed points that should follow the specified scale
        rather than mode hopping. :code:`default=0.5`
    scale: float
        The amount to scale the difference vector by.
        :code:`default = 2.38 / (2 * n_cluster)**0.5)`

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    """
    delta = np.diff(rstate.choice(live, 2, replace=False), axis=0)[0]
    if rstate.uniform(0, 1) < mix:
        if scale is None:
            scale = 2.38 / (2 * n_cluster) ** 0.5
        scale *= rstate.gamma(4, 0.25)
    else:
        scale = 1
    u_prop = u.copy()
    u_prop[:n_cluster] += delta * scale
    u_prop[n_cluster:] = rstate.uniform(0, 1, n - n_cluster)
    return u_prop


def propose_volumetric(
    u,
    axes,
    scale,
    n,
    n_cluster,
    rstate,
):
    """
    Propose a new point using the default :code:`dynesty` proposal.

    The new difference vector is scaled by a vector isotropically drawn
    from an ellipsoid centered on zero with covariance given by the
    provided axis. Note that the magnitude of this proposal is heavily
    skewed to the size of the ellipsoid.

    Parameters
    ----------
    u: np.ndarray
        The current point.
    n: int
        The number of dimensions being explored.
    scale: float
        The amount to scale the proposed point by.
    n_cluster: int
        The number of dimensions to run the differential evolution over, the
        first :code:`n_cluster` dimensions are used. The rest are randomly
        sampled from the prior.
    rstate: numpy.random.Generator
        The numpy generator instance used for random number generation.
        Consider using built in `bilby.core.utils.random.rng`.

    Returns
    -------
    u_prop: np.ndarray
        The proposed point.
    """
    # Propose a direction on the unit n-sphere.
    drhat = rstate.normal(0, 1, n_cluster)
    drhat /= np.linalg.norm(drhat)

    # Scale based on dimensionality.
    dr = drhat * rstate.uniform(0, 1) ** (1.0 / n_cluster)

    # Transform to proposal distribution.
    delta = scale * np.dot(axes, dr)
    u_prop = u.copy()
    u_prop[:n_cluster] += delta
    u_prop[n_cluster:] = rstate.uniform(0, 1, n - n_cluster)
    return u_prop


def apply_boundaries_(u_prop, periodic, reflective):
    """
    Apply the periodic and reflective boundaries and test if we are inside the
    unit cube.

    Parameters
    ----------
    u_prop: np.ndarray
        The proposed point in the unit hypercube space.
    periodic: np.ndarray
        Indices of the parameters with periodic boundaries.
    reflective: np.ndarray
        Indices of the parameters with reflective boundaries.

    Returns
    =======
    [np.ndarray, None]:
        Either the remapped proposed point, or None if the proposed point
        lies outside the unit cube.
    """
    # Wrap periodic parameters
    if periodic is not None:
        u_prop[periodic] = np.mod(u_prop[periodic], 1)

    # Reflect
    if reflective is not None:
        u_prop[reflective] = apply_reflect(u_prop[reflective])

    if u_prop.min() < 0 or u_prop.max() > 1:
        return None
    else:
        return u_prop


proposal_funcs = dict(
    diff=propose_differential_evolution, volumetric=propose_volumetric
)
