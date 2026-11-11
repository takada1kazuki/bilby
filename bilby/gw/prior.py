import os
import copy
import zipfile

import array_api_extra as xpx
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid, quad
from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import beta as beta_function
from scipy.special import betainc, hyp2f1, spence
from scipy.stats import norm

from ..compat.utils import xp_wrap
from ..core.prior import (
    PriorDict,
    Uniform,
    Prior,
    DeltaFunction,
    Gaussian,
    Interped,
    Constraint,
    conditional_prior_factory,
    PowerLaw,
    ConditionalLogUniform,
    ConditionalPriorDict,
    ConditionalBasePrior,
    BaseJointPriorDist,
    JointPrior,
    JointPriorDistError,
)
from ..core.utils import infer_args_from_method, logger, random, interp1d
from .conversion import (
    convert_to_lal_binary_black_hole_parameters,
    convert_to_lal_binary_neutron_star_parameters,
    generate_mass_parameters,
    generate_tidal_parameters,
    fill_from_fixed_priors,
    generate_all_bbh_parameters,
    chirp_mass_and_mass_ratio_to_total_mass,
    total_mass_and_mass_ratio_to_component_masses,
)
from .cosmology import get_cosmology, z_at_value
from .source import PARAMETER_SETS
from .utils import calculate_time_to_merger


DEFAULT_PRIOR_DIR = os.path.join(os.path.dirname(__file__), "prior_files")


class BilbyPriorConversionError(Exception):
    pass


def convert_to_flat_in_component_mass_prior(result, fraction=0.25):
    """Converts samples with a defined prior in chirp-mass and mass-ratio to flat in component mass by resampling with
    the posterior with weights defined as ratio in new:old prior values times the jacobian which for
    F(mc, q) -> G(m1, m2) is defined as J := m1^2 / mc

    Parameters
    ==========
    result: bilby.core.result.Result
        The output result complete with priors and posteriors
    fraction: float [0, 1]
        The fraction of samples to draw (default=0.25). Note, if too high a
        fraction of samples are draw, the reweighting will not be applied in
        effect.

    """
    if getattr(result, "priors") is not None:
        for key in ["chirp_mass", "mass_ratio"]:
            if key not in result.priors.keys():
                BilbyPriorConversionError(
                    "{} Prior not found in result object".format(key)
                )
            if isinstance(result.priors[key], Constraint):
                BilbyPriorConversionError(
                    "{} Prior should not be a Constraint".format(key)
                )
        for key in ["mass_1", "mass_2"]:
            if not isinstance(result.priors[key], Constraint):
                BilbyPriorConversionError(
                    "{} Prior should be a Constraint Prior".format(key)
                )
    else:
        BilbyPriorConversionError("No prior in the result: unable to convert")

    result = copy.copy(result)
    priors = result.priors
    old_priors = copy.copy(result.priors)
    posterior = result.posterior

    for key in ["chirp_mass", "mass_ratio"]:
        priors[key] = Constraint(
            priors[key].minimum,
            priors[key].maximum,
            key,
            latex_label=priors[key].latex_label,
        )
    for key in ["mass_1", "mass_2"]:
        priors[key] = Uniform(
            priors[key].minimum,
            priors[key].maximum,
            key,
            latex_label=priors[key].latex_label,
            unit=r"$M_{\odot}$",
        )

    weights = np.array(
        result.get_weights_by_new_prior(
            old_priors,
            priors,
            prior_names=["chirp_mass", "mass_ratio", "mass_1", "mass_2"],
        )
    )
    jacobian = posterior["mass_1"] ** 2 / posterior["chirp_mass"]
    weights = jacobian * weights
    result.posterior = posterior.sample(frac=fraction, weights=weights)

    logger.info("Resampling posterior to flat-in-component mass")
    effective_sample_size = sum(weights) ** 2 / sum(weights**2)
    n_posterior = len(posterior)
    if fraction > effective_sample_size / n_posterior:
        logger.warning(
            "Sampling posterior of length {} with fraction {}, but "
            "effective_sample_size / len(posterior) = {}. This may produce "
            "biased results".format(
                n_posterior, fraction, effective_sample_size / n_posterior
            )
        )
    result.posterior = posterior.sample(frac=fraction, weights=weights, replace=True)
    result.meta_data["reweighted_to_flat_in_component_mass"] = True
    return result


class Cosmological(Interped):
    """
    Base prior class for cosmologically aware distance parameters.
    """

    @property
    def _default_args_dict(self):
        from astropy import units

        return dict(
            redshift=dict(name="redshift", latex_label="$z$", unit=None),
            luminosity_distance=dict(
                name="luminosity_distance", latex_label="$d_L$", unit=units.Mpc
            ),
            comoving_distance=dict(
                name="comoving_distance", latex_label="$d_C$", unit=units.Mpc
            ),
        )

    def __init__(
        self,
        minimum,
        maximum,
        cosmology=None,
        name=None,
        latex_label=None,
        unit=None,
        boundary=None,
    ):
        """

        Parameters
        ----------
        minimum: float
            The minimum value for the distribution.
        maximum: float
            The maximum value for the distribution.
        cosmology: Union[None, str, dict, astropy.cosmology.FLRW]
            The cosmology to use, see `bilby.gw.cosmology.set_cosmology`
            for more details.
            If :code:`None`, the default cosmology will be used.
        name: str
            The name of the parameter, should be one of
            :code:`[luminosity_distance, comoving_distance, redshift]`.
        latex_label: str
            Latex string to use in plotting, if :code:`None`, this will
            be set automatically using the name.
        unit: Union[str, astropy.units.Unit]
            The unit of the maximum and minimum values, :code:`default=Mpc`.
        boundary: str
            The boundary condition to apply to the prior when sampling.
        """
        from astropy import units

        self.cosmology = get_cosmology(cosmology)
        if name not in self._default_args_dict:
            raise ValueError(
                "Name {} not recognised. Must be one of luminosity_distance, "
                "comoving_distance, redshift".format(name)
            )
        self.name = name
        label_args = self._default_args_dict[self.name]
        if latex_label is not None:
            label_args["latex_label"] = latex_label
        if unit is not None:
            if not isinstance(unit, units.Unit):
                unit = units.Unit(unit)
            label_args["unit"] = unit
        self.unit = label_args["unit"]
        self._minimum = dict()
        self._maximum = dict()
        self.minimum = minimum
        self.maximum = maximum
        if name == "redshift":
            xx, yy = self._get_redshift_arrays()
        elif name == "comoving_distance":
            xx, yy = self._get_comoving_distance_arrays()
        elif name == "luminosity_distance":
            xx, yy = self._get_luminosity_distance_arrays()
        else:
            raise ValueError("Name {} not recognized.".format(name))
        super(Cosmological, self).__init__(
            xx=xx,
            yy=yy,
            minimum=minimum,
            maximum=maximum,
            boundary=boundary,
            **label_args,
        )

    @property
    def minimum(self):
        return self._minimum[self.name]

    @minimum.setter
    def minimum(self, minimum):
        if (self.name in self._minimum) and (minimum < self.minimum):
            self._set_limit(
                value=minimum, limit_dict=self._minimum, recalculate_array=True
            )
        else:
            self._set_limit(value=minimum, limit_dict=self._minimum)

    @property
    def maximum(self):
        return self._maximum[self.name]

    @maximum.setter
    def maximum(self, maximum):
        if (self.name in self._maximum) and (maximum > self.maximum):
            self._set_limit(
                value=maximum, limit_dict=self._maximum, recalculate_array=True
            )
        else:
            self._set_limit(value=maximum, limit_dict=self._maximum)

    def _set_limit(self, value, limit_dict, recalculate_array=False):
        """
        Set either of the limits for redshift, luminosity, and comoving distances

        Parameters
        ==========
        value: float
            Limit value in current class' parameter
        limit_dict: dict
            The limit dictionary to modify in place
        recalculate_array: boolean
            Determines if the distance arrays are recalculated
        """
        cosmology = get_cosmology(self.cosmology)
        limit_dict[self.name] = value
        if self.name == "redshift":
            limit_dict["luminosity_distance"] = cosmology.luminosity_distance(
                value
            ).value
            limit_dict["comoving_distance"] = cosmology.comoving_distance(value).value
        elif self.name == "luminosity_distance":
            if value == 0:
                limit_dict["redshift"] = 0
            else:
                limit_dict["redshift"] = z_at_value(
                    cosmology.luminosity_distance, value * self.unit
                )
            limit_dict["comoving_distance"] = cosmology.comoving_distance(
                limit_dict["redshift"]
            ).value
        elif self.name == "comoving_distance":
            if value == 0:
                limit_dict["redshift"] = 0
            else:
                limit_dict["redshift"] = z_at_value(
                    cosmology.comoving_distance, value * self.unit
                )
            limit_dict["luminosity_distance"] = cosmology.luminosity_distance(
                limit_dict["redshift"]
            ).value
        if recalculate_array:
            if self.name == "redshift":
                self.xx, self.yy = self._get_redshift_arrays()
            elif self.name == "comoving_distance":
                self.xx, self.yy = self._get_comoving_distance_arrays()
            elif self.name == "luminosity_distance":
                self.xx, self.yy = self._get_luminosity_distance_arrays()
        try:
            self._update_instance()
        except (AttributeError, KeyError):
            pass

    def get_corresponding_prior(self, name=None, unit=None):
        """
        Utility method to convert between equivalent redshift, comoving
        distance, and luminosity distance priors.

        Parameters
        ----------
        name: str
            The name of the new prior, this should be one of
            :code:`[luminosity_distance, comoving_distance, redshift]`.
        unit: Union[str, astropy.units.Unit]
            The unit of the output prior.

        Returns
        -------
        The corresponding prior.
        """
        subclass_args = infer_args_from_method(self.__init__)
        args_dict = {key: getattr(self, key) for key in subclass_args}
        self._convert_to(new=name, args_dict=args_dict)
        if unit is not None:
            args_dict["unit"] = unit
        return self.__class__(**args_dict)

    def _convert_to(self, new, args_dict):
        args_dict.update(self._default_args_dict[new])
        args_dict["minimum"] = self._minimum[args_dict["name"]]
        args_dict["maximum"] = self._maximum[args_dict["name"]]

    def _get_comoving_distance_arrays(self):
        zs, p_dz = self._get_redshift_arrays()
        dc_of_z = self.cosmology.comoving_distance(zs).value
        ddc_dz = np.gradient(dc_of_z, zs)
        p_dc = p_dz / ddc_dz
        return dc_of_z, p_dc

    def _get_luminosity_distance_arrays(self):
        zs, p_dz = self._get_redshift_arrays()
        dl_of_z = self.cosmology.luminosity_distance(zs).value
        ddl_dz = np.gradient(dl_of_z, zs)
        p_dl = p_dz / ddl_dz
        return dl_of_z, p_dl

    def _get_redshift_arrays(self):
        raise NotImplementedError

    @classmethod
    def from_repr(cls, string):
        if "FlatLambdaCDM" in string:
            logger.warning(
                "Cosmological priors cannot be loaded from a string. "
                "If the prior has a name, use that instead."
            )
            return string
        else:
            return cls._from_repr(string)

    def get_instantiation_dict(self):
        from astropy import units
        from .cosmology import get_available_cosmologies

        available = get_available_cosmologies()
        instantiation_dict = super().get_instantiation_dict()
        if self.cosmology.name in available:
            instantiation_dict["cosmology"] = self.cosmology.name
        if isinstance(self.unit, units.Unit):
            instantiation_dict["unit"] = self.unit.to_string()
        return instantiation_dict


class UniformComovingVolume(Cosmological):
    r"""
    Prior that is uniform in the comoving volume.

    Note that this does not account for time dilation in the source frame
    use `bilby.gw.prior.UniformSourceFrame` to include that effect.

    .. math::

        p(z) \propto \frac{d V_{c}}{dz}
    """

    def _get_redshift_arrays(self):
        zs = np.linspace(
            self._minimum["redshift"] * 0.99, self._maximum["redshift"] * 1.01, 1000
        )
        p_dz = self.cosmology.differential_comoving_volume(zs).value
        return zs, p_dz


class UniformSourceFrame(Cosmological):
    r"""
    Prior for redshift which is uniform in comoving volume and source frame
    time.

    .. math::

        p(z) \propto \frac{1}{1 + z}\frac{dV_{c}}{dz}

    The extra :math:`1+z` is due to doppler shifting of the source frame time.
    """

    def _get_redshift_arrays(self):
        zs = np.linspace(
            self._minimum["redshift"] * 0.99, self._maximum["redshift"] * 1.01, 1000
        )
        p_dz = self.cosmology.differential_comoving_volume(zs).value / (1 + zs)
        return zs, p_dz


class UniformInComponentsChirpMass(PowerLaw):
    r"""
    Prior distribution for chirp mass which is uniform in component masses.

    This is useful when chirp mass and mass ratio are sampled while the
    prior is uniform in component masses.

    .. math::

        p({\cal M}) \propto {\cal M}

    Notes
    -----
    This prior is intended to be used in conjunction with the corresponding
    :code:`bilby.gw.prior.UniformInComponentsMassRatio`.
    """

    _leaves = ["_minimum", "_maximum", "alpha"]

    def __init__(
        self,
        minimum,
        maximum,
        name="chirp_mass",
        latex_label=r"$\mathcal{M}$",
        unit=None,
        boundary=None,
    ):
        """
        Parameters
        ==========
        minimum : float
            The minimum of chirp mass
        maximum : float
            The maximum of chirp mass
        name: see superclass
        latex_label: see superclass
        unit: see superclass
        boundary: see superclass
        """
        super(UniformInComponentsChirpMass, self).__init__(
            alpha=1.0,
            minimum=minimum,
            maximum=maximum,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
        )


class UniformInComponentsMassRatio(Prior):
    r"""
    Prior distribution for chirp mass which is uniform in component masses.

    This is useful when chirp mass and mass ratio are sampled while the
    prior is uniform in component masses.

    .. math::

        p(q) \propto \frac{(1 + q)^{2/5}}{q^{6/5}}

    Notes
    -----
    This prior is intended to be used in conjunction with the corresponding
    :code:`bilby.gw.prior.UniformInComponentsChirpMass`.
    """

    _leaves = ["norm", "icdf"]

    def __init__(
        self,
        minimum,
        maximum,
        name="mass_ratio",
        latex_label="$q$",
        unit=None,
        boundary=None,
        equal_mass=False,
    ):
        """
        Parameters
        ==========
        minimum : float
            The minimum of mass ratio
        maximum : float
            The maximum of mass ratio
        name: see superclass
        latex_label: see superclass
        unit: see superclass
        boundary: see superclass
        equal_mass: bool
            Whether the likelihood being considered is expected to peak at
            equal masses. If True, the mapping described in Appendix A of
            arXiv:2111.13619 is used in the :code:`rescale` method.
            default=False

        """
        self.equal_mass = equal_mass
        super(UniformInComponentsMassRatio, self).__init__(
            minimum=minimum,
            maximum=maximum,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
        )
        self.norm = self._integral(maximum) - self._integral(minimum)
        qs = np.linspace(minimum, maximum, 1000)
        self.icdf = interp1d(
            self.cdf(qs),
            qs,
            kind="cubic",
            bounds_error=False,
            fill_value=(minimum, maximum),
        )

    @staticmethod
    def _integral(q):
        return -5.0 * q ** (-1.0 / 5.0) * hyp2f1(-2.0 / 5.0, -1.0 / 5.0, 4.0 / 5.0, -q)

    def cdf(self, val, *, xp=np):
        return (self._integral(val) - self._integral(self.minimum)) / self.norm

    @xp_wrap
    def rescale(self, val, *, xp=None):
        if self.equal_mass:
            val = 2 * xp.minimum(val, 1 - val)
        return self.icdf(val)

    def prob(self, val, *, xp=None):
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        with np.errstate(invalid="ignore"):
            prob = (
                (1.0 + val) ** (2.0 / 5.0) / (val ** (6.0 / 5.0)) / self.norm * in_prior
            )
        return prob

    @xp_wrap
    def ln_prob(self, val, *, xp=None):
        with np.errstate(divide="ignore"):
            return xp.log(self.prob(val, xp=xp))


class AlignedSpin(Interped):
    r"""
    Prior distribution for the aligned (z) component of the spin.

    This takes prior distributions for the magnitude and cosine of the tilt
    and forms a compound prior using a numerical convolution integral.

    .. math::

        \pi(\chi) = \int_{0}^{1} da \int_{-1}^{1} d\cos\theta
        \pi(a) \pi(\cos\theta) \delta(\chi - a \cos\theta)

    This is useful when using aligned-spin only waveform approximants.

    This is an extension of e.g., (A7) of https://arxiv.org/abs/1805.10457.
    For the simple case of uniform in magnitude and orientation priors, we use an
    analytic form of the PDF.
    """

    _leaves = Interped._leaves + ["a_prior", "z_prior"]

    def __init__(
        self,
        a_prior=Uniform(0, 1),
        z_prior=Uniform(-1, 1),
        name=None,
        latex_label=None,
        unit=None,
        boundary=None,
        minimum=np.nan,
        maximum=np.nan,
        num_interp=None,
    ):
        """
        Parameters
        ==========
        a_prior: Prior
            Prior distribution for spin magnitude
        z_prior: Prior
            Prior distribution for cosine spin tilt
        name: see superclass
        latex_label: see superclass
        unit: see superclass
        num_interp: int
            The number of points to use in the interpolation. Defaults to 100,000
            for simple aligned priors and 10,000 for general priors.
        """
        self.a_prior = a_prior
        self.z_prior = z_prior
        chi_min = min(
            a_prior.maximum * z_prior.minimum, a_prior.minimum * z_prior.maximum
        )
        chi_max = a_prior.maximum * z_prior.maximum
        if self._is_simple_aligned_prior:
            self.num_interp = 100_000 if num_interp is None else num_interp
            xx = np.linspace(chi_min, chi_max, self.num_interp)
            yy = -np.log(np.abs(xx) / a_prior.maximum) / (2 * a_prior.maximum)
        else:

            def integrand(aa, chi):
                """
                The integrand for the aligned spin (chi) probability density
                after performing the integral over spin orientation using a
                delta function identity.
                """
                return a_prior.prob(aa, xp=None) * z_prior.prob(chi / aa, xp=None) / aa

            self.num_interp = 10_000 if num_interp is None else num_interp
            xx = np.linspace(chi_min, chi_max, self.num_interp)
            yy = [
                quad(integrand, a_prior.minimum, a_prior.maximum, chi)[0] for chi in xx
            ]
        super().__init__(
            xx=xx,
            yy=yy,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
            minimum=minimum,
            maximum=maximum,
        )

    @property
    def _is_simple_aligned_prior(self):
        return (
            isinstance(self.a_prior, Uniform)
            and isinstance(self.z_prior, Uniform)
            and self.z_prior.minimum == -1
            and self.z_prior.maximum == 1
        )


class ConditionalChiUniformSpinMagnitude(ConditionalLogUniform):
    r"""
    This prior characterizes the conditional prior on the spin magnitude given
    the aligned component of the spin  such that the marginal prior is uniform
    if the distribution of spin orientations is isotropic.

    .. math::

        p(a) &= \frac{1}{a_{\max}} \\
        p(\chi) &= - \frac{\ln(|\chi|)}{2 a_{\max}}  \\
        p(a | \chi) &\propto \frac{1}{a}
    """

    def __init__(
        self, minimum, maximum, name, latex_label=None, unit=None, boundary=None
    ):
        super(ConditionalChiUniformSpinMagnitude, self).__init__(
            minimum=minimum,
            maximum=maximum,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
            condition_func=self._condition_function,
        )
        self._required_variables = [name.replace("a", "chi")]
        self.__class__.__name__ = "ConditionalChiUniformSpinMagnitude"
        self.__class__.__qualname__ = "ConditionalChiUniformSpinMagnitude"

    def _condition_function(self, reference_params, **kwargs):
        return dict(
            minimum=np.abs(kwargs[self._required_variables[0]]),
            maximum=reference_params["maximum"],
        )

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict


class ConditionalChiInPlane(ConditionalBasePrior):
    r"""
    This prior characterizes the conditional prior on the in-plane spin magnitude
    given the aligned component of the spin  such that the marginal prior is uniform
    if the distribution of spin orientations is isotropic.

    .. math::

        p(a) &= \frac{1}{a_{\max}} \\
        p(\chi_\perp) &= \frac{2 N \chi_\perp}{\chi^2 + \chi_\perp^2} \\
        N^{-1} &= 2 \ln\left(\frac{a_\max}{|\chi|}\right)
    """

    def __init__(
        self, minimum, maximum, name, latex_label=None, unit=None, boundary=None
    ):
        super(ConditionalChiInPlane, self).__init__(
            minimum=minimum,
            maximum=maximum,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
            condition_func=self._condition_function,
        )
        self._required_variables = [name[:5]]
        self._reference_maximum = maximum
        self.__class__.__name__ = "ConditionalChiInPlane"
        self.__class__.__qualname__ = "ConditionalChiInPlane"

    @xp_wrap
    def prob(self, val, *, xp=np, **required_variables):
        parameters = self.condition_func(
            self.reference_params.copy(), **required_variables
        )
        chi_aligned = abs(required_variables[self._required_variables[0]])
        minimum = parameters.get("minimum", self.minimum)
        maximum = parameters.get("maximum", self.maximum)
        return (
            (val >= minimum)
            * (val <= maximum)
            * val
            / (chi_aligned**2 + val**2)
            / xp.log(self._reference_maximum / chi_aligned)
        )

    @xp_wrap
    def ln_prob(self, val, *, xp=np, **required_variables):
        with np.errstate(divide="ignore"):
            return xp.log(self.prob(val, **required_variables))

    @xp_wrap
    def cdf(self, val, *, xp=np, **required_variables):
        r"""
        .. math::
            \text{CDF}(\chi_\per) = N ln(1 + (\chi_\perp / \chi) ** 2)

        Parameters
        ----------
        val: (float, array-like)
            The value at which to evaluate the CDF
        required_variables: dict
            A dictionary containing the aligned component of the spin

        Returns
        -------
        (float, array-like)
            The value of the CDF

        """
        self.update_conditions(**required_variables)
        chi_aligned = abs(required_variables[self._required_variables[0]])
        return xp.clip(
            (val >= self.minimum)
            * (val <= self.maximum)
            * xp.log(1 + (val / chi_aligned) ** 2)
            / 2
            / xp.log(self._reference_maximum / chi_aligned),
            0,
            1,
        )

    def rescale(self, val, *, xp=np, **required_variables):
        r"""
        .. math::
            \text{PPF}(\chi_\perp) = ((a_\max / \chi) ** (2x) - 1) ** 0.5 * \chi

        Parameters
        ----------
        val: (float, array-like)
            The value to rescale
        required_variables: dict
            Dictionary containing the aligned spin component

        Returns
        -------
        (float, array-like)
            The in-plane component of the spin
        """
        self.update_conditions(**required_variables)
        chi_aligned = abs(required_variables[self._required_variables[0]])
        return (
            chi_aligned
            * ((self._reference_maximum / chi_aligned) ** (2 * val) - 1) ** 0.5
        )

    def _condition_function(self, reference_params, **kwargs):
        with np.errstate(invalid="ignore"):
            maximum = (
                self._reference_maximum**2 - kwargs[self._required_variables[0]] ** 2
            ) ** 0.5
        return dict(minimum=0, maximum=maximum)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict


DEFAULT_CHI_EFF_TABLE_GRID = dict(n_q=200, n_u=1025)
DEFAULT_CHI2_TABLE_GRID = dict(n_q=120, n_chieff=301, n_u=801)

# ============================================================
# Loading / building the chi_eff and chi_2 tables
# ============================================================


def _check_table(table, q_min, q_max, a_max, grid):
    """Raise ValueError if a loaded table does not match the requested settings
    or has non-finite entries (table.filename is the loaded file)."""
    problems = []
    if not np.isclose(table.a_max, a_max):
        problems.append(f"a_max={table.a_max} (expected {a_max})")
    if not np.isclose(table.q_grid[0], q_min) or not np.isclose(
        table.q_grid[-1], q_max
    ):
        problems.append(
            f"q range=({table.q_grid[0]}, {table.q_grid[-1]}) "
            f"(expected ({q_min}, {q_max}))"
        )

    sizes = dict(n_q=len(table.q_grid), n_u=len(table.u_grid))
    if hasattr(table, "chieff_grid"):
        sizes["n_chieff"] = len(table.chieff_grid)
    for key, expected in grid.items():
        if sizes.get(key) != int(expected):
            problems.append(f"{key}={sizes.get(key)} (expected {expected})")

    n_bad = int(np.sum(~np.isfinite(table.inv_table)))
    if n_bad:
        problems.append(f"{n_bad} non-finite (NaN or inf) table entries")

    if problems:
        raise ValueError(
            f"The existing table {table.filename} does not match the requested settings: "
            + "; ".join(problems)
            + ". Delete the file to rebuild it, or pass the matching settings."
        )


def load_or_build_tables(
    q_min,
    q_max,
    a_max,
    chi_eff_file,
    chi2_file,
    chi_eff_grid=None,
    chi2_grid=None,
    chi_eff_dtype=np.float64,
    chi2_dtype=np.float32,
):
    """Load or build the chi_eff / chi_2 inverse-CDF tables.

    If a file already exists, the table is loaded and settings will be checked (_check_table);
    otherwise it is newly built, checked for non-finite entries and saved.
    """
    chi_eff_grid = dict(
        DEFAULT_CHI_EFF_TABLE_GRID if chi_eff_grid is None else chi_eff_grid
    )
    chi2_grid = dict(DEFAULT_CHI2_TABLE_GRID if chi2_grid is None else chi2_grid)

    if os.path.exists(chi_eff_file):
        logger.info(f"Loading chi_eff table {chi_eff_file}")
        chi_eff_table = ChieffTable.load(chi_eff_file)
        _check_table(chi_eff_table, q_min, q_max, a_max, chi_eff_grid)
    else:
        logger.info(f"Building chi_eff table {chi_eff_file}")
        chi_eff_table = ChieffTable(
            q_min=q_min,
            q_max=q_max,
            a_max=a_max,
            dtype=chi_eff_dtype,
            **chi_eff_grid,
        )
        if not np.all(np.isfinite(chi_eff_table.inv_table)):
            raise ValueError("The newly built chi_eff table has non-finite entries.")
        chi_eff_table.save(chi_eff_file)

    if os.path.exists(chi2_file):
        logger.info(f"Loading chi_2 table {chi2_file}")
        chi2_table = Chi2zTable.load(chi2_file)
        _check_table(chi2_table, q_min, q_max, a_max, chi2_grid)
    else:
        logger.info(f"Building chi_2 table {chi2_file}")
        chi2_table = Chi2zTable(
            q_min=q_min,
            q_max=q_max,
            a_max=a_max,
            dtype=chi2_dtype,
            **chi2_grid,
        )
        if not np.all(np.isfinite(chi2_table.inv_table)):
            raise ValueError("The newly built chi_2 table has non-finite entries.")
        chi2_table.save(chi2_file)

    logger.info(
        f"Inverse-CDF tables ready: chi_eff {chi_eff_table.inv_table.shape}, "
        f"chi_2 {chi2_table.inv_table.shape}"
    )
    return chi_eff_table, chi2_table


def _newton_update(
    x, u, lower, upper, cdf_pdf_func, n_newton, pdf_floor=1e-300, atol=1e-11
):
    """
    Solve F(x) = u on (lower, upper) with Newton's method.

    Takes up to ``n_newton`` steps and stops as soon as |F(x) - u| <= tolerance,
    where tolerance = min(atol, max(5e-16, 1e-7 min(u, 1 - u))) is tightened
    near u = 0 and 1 to keep a relative accuracy in the tails.
    It stops without converging if the pdf is <= ``pdf_floor`` or NaN, a step
    leaves (lower, upper), or a step does not change x.

    Parameters
    ==========
    x: float
        Initial guess (from the inverse-CDF table).
    u: float
        Target CDF value.
    lower, upper: float
        Interval in which the root is searched.
    cdf_pdf_func: callable
        cdf_pdf_func(x) returns (F(x), pdf(x)).
    n_newton: int
        Maximum number of Newton steps.

    Returns
    =======
    (x, F, converged): (float, float, bool)
        current value, its CDF value, and judgement of convergence.
        If not converged they are passed on to _brent_update.
    """
    tolerance = min(atol, max(5e-16, 1e-7 * min(u, 1.0 - u)))

    F, pdf = cdf_pdf_func(x)
    for step in range(n_newton + 1):
        # If the CDF value F is close enough to the given u, quit _newton_update.
        if abs(F - u) <= tolerance:
            return float(x), F, True
        # If the step count reaches n_newton, or the pdf is <= pdf_floor or NaN,
        # stop the Newton steps and go to _brent_update.
        if step == n_newton or not pdf > pdf_floor:
            break
        candidate = x - (F - u) / pdf
        # If the candidate is out of (lower, upper) or does not change x,
        # stop the Newton steps and go to _brent_update.
        if not lower < candidate < upper or candidate == x:
            break
        x = float(candidate)
        F, pdf = cdf_pdf_func(x)
    return float(x), F, False


def _brent_update(x, F, u, lower, upper, cdf_func):
    """
    Solve F(x) = u on [lower, upper] with Brent's method.

    Parameters
    ==========
    x, F: float
        The last point of _newton_update and its CDF value.
    u: float
        Target CDF value.
    lower, upper: float
        Interval in which the root is searched.
    cdf_func: callable
        cdf_func(x) returns F(x).

    Returns
    =======
    float
        The root, i.e. the parameter value itself (chi_eff or chi_2).

    Raises
    ======
    FloatingPointError
        If [lower, upper] is not a finite, non-empty interval.
    """
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise FloatingPointError("Invalid support in inverse CDF correction")
    if np.isfinite(F) and lower < x < upper:
        if F < u:
            lower = x
        else:
            upper = x
    lower, upper = float(lower), float(upper)
    try:
        root = brentq(
            lambda y: float(cdf_func(y)) - u,
            lower,
            upper,
            xtol=max(1e-12 * (upper - lower), 1e-16),
            maxiter=120,
        )
    except (ValueError, RuntimeError) as exc:
        # ValueError: no sign change on [lower, upper], or the CDF returned NaN.
        # RuntimeError: no convergence within maxiter iterations.
        logger.warning(
            f"Inverse CDF: Brent failed for u={u} on [{lower}, {upper}] ({exc!r}); "
            "using the last Newton value instead."
        )
        best = float(x) if np.isfinite(x) else 0.5 * (lower + upper)
        return min(max(best, lower), upper)
    return float(root)


def _table_for_prior(table, table_file, table_class, table_settings=None):
    """
    Return the inverse-CDF table of a prior.

    ``table`` is returned if given.  Otherwise (e.g. for a prior restored from
    JSON) the table is loaded from ``table_file``.  If the file is missing,
    broken or does not match ``table_settings``, the table is rebuilt from
    ``table_settings`` with a warning; the rebuilt table is the same as the
    original one, since the table is fully determined by its settings.

    Parameters
    ==========
    table: ChieffTable or Chi2zTable, or None
        Table passed directly to the prior.
    table_file: str or None
        Path of the saved table (stored in the prior JSON).
    table_class: type
        ChieffTable or Chi2zTable, used to load or rebuild the table.
    table_settings: dict, optional
        table.settings() of the original table (stored in the prior JSON).

    Returns
    =======
    ChieffTable or Chi2zTable
        The given, loaded or rebuilt table.

    Raises
    ======
    ValueError
        If the table can be neither given nor loaded, and no table_settings
        are given to rebuild it.
    """
    if table is not None:
        return table
    problem = "is not given"
    if table_file is not None:
        try:
            loaded = table_class.load(table_file)
        except (OSError, EOFError, KeyError, ValueError, zipfile.BadZipFile) as exc:
            problem = f"cannot be loaded ({exc!r})"
        else:
            if table_settings is None or loaded.settings() == table_settings:
                return loaded
            problem = "does not match table_settings"
    if table_settings is None:
        raise ValueError(
            f"The inverse-CDF table {table_file!r} {problem}, "
            "and no table_settings are given to rebuild it."
        )
    logger.warning(
        f"The inverse-CDF table {table_file!r} {problem}; rebuilding it from table_settings."
    )
    return table_class(**table_settings)


# ============================================================
# p(chi_eff | q)
#
# In terms of u = (1 + q) |chi_eff| / a_max the density and the CDF are
# piecewise closed-form on the regions
#     q < 1/2:  A (0, q),      B (q, 1 - q),  D (1 - q, 1),  E (1, 1 + q)
#     q > 1/2:  A (0, 1 - q),  C (1 - q, q),  D (q, 1),      E (1, 1 + q)


# _FP_X returns (F, pdf) on region X.
# F is a primitive (antiderivative) of 4q * pdf in chi_eff.  On A and B it is the
# one vanishing at chi_eff = 0, G(u) = int_0^u H(u') du' with
# pdf = (1 + q) H(u) / (4 q a_max); on C, D and E it is G + delta with
# delta = _delta_at_1_q(q).  The CDF is 1/2 + sign(chi_eff) G / (4 q).
# _FP_AB_at_q, _FP_CD_at_q and _FP_DE_at_1 return the same pair at the
# region boundaries u = q and u = 1 (where the
# region formulas contain 0 * log 0).  At u = 1 - q the formulas are regular,
# so the region on its left (B for q < 1/2, A for q > 1/2) is used there.
# ============================================================
def _reli2_scalar(x):
    """
    Real part of the dilogarithm Li2(x) for a real scalar x.

    Li2(x) = spence(1 - x) for x <= 1.
    For x > 1 Li2 is complex and its real part follows from the inversion formula,
        Re Li2(x) = pi^2 / 3 - log(x)^2 / 2 - Li2(1 / x).
    Every dilogarithm in this module is real-part, so this single function is used everywhere.
    """
    if x <= 1.0:
        return float(spence(1.0 - x))
    return np.pi**2 / 3.0 - 0.5 * np.log(x) ** 2 - float(spence(1.0 - 1.0 / x))


def _delta_at_1_q(q):
    """Constant delta(q) by which the primitive F on C, D and E exceeds the one
    vanishing at chi_eff = 0 (subtracted again in _cdf_pdf_chi_eff_scalar)."""
    return 0.5 * (1.0 - q) ** 2 + 0.25 * np.log(q)


def _FP_AB_at_q(q, a_max):
    """(F, pdf) at u = q for q <= 1/2 (boundary of A and B), with F a primitive of 4 q pdf."""
    log2q = np.log(2.0 * q)
    F = q**2 * (4.5 - 2.0 * log2q - np.pi**2 / 8.0)
    p = (1.0 + q) / (4.0 * a_max) * (4.0 - 2.0 * log2q - np.pi**2 / 4.0)
    return float(F), float(p)


def _FP_CD_at_q(q, a_max):
    """(F, pdf) at u = q for q > 1/2 (boundary of C and D), with F a primitive of 4 q pdf.
    Region C at u = q, with the 0 log 0 terms dropped and Li2(q / u) = Li2(1) = pi^2 / 6."""
    if q == 1.0:
        return 11.0 / 4.0 - np.pi**2 / 12.0, (1.0 / a_max) * (1.0 - np.pi**2 / 12.0)
    q2, logq, log1mq = q * q, np.log(q), np.log1p(-q)
    dilog = _reli2_scalar(1.0 - 1.0 / q) - np.pi**2 / 6.0
    F = (
        (-0.75 * q2 + q + 0.5 * q2 * logq) * logq
        + (0.75 * q2 - q + 0.25) * log1mq
        - 0.5 * q2 * logq * log1mq
        + 0.5 * q2 * dilog
        - 1.75 * q2
        + 2.25 * (1.0 + q) * q
    )
    p = (
        (1.0 + q)
        / (4.0 * q * a_max)
        * (
            2.0
            + (1.0 - q + q * logq) * logq
            + (q - 1.0 - q * logq) * log1mq
            + q * dilog
        )
    )
    return float(F), float(p)


def _FP_DE_at_1(q, a_max):
    """(F, pdf) at u = 1 (boundary of D and E), with F a primitive of 4 q pdf."""
    if q == 1.0:
        return 11.0 / 4.0 - np.pi**2 / 12.0, (1.0 / a_max) * (1.0 - np.pi**2 / 12.0)
    logq, log1mq, li2q = np.log(q), np.log1p(-q), _reli2_scalar(q)
    F = (0.75 - q + 0.25 * q**2) * log1mq + 0.25 * logq - 0.5 * li2q + 0.5 + 2.25 * q
    p = (1.0 + q) / (4.0 * q * a_max) * (2.0 * q + (1.0 - q) * log1mq - li2q)
    return float(F), float(p)


def _FP_A(u, q, a_max):
    """(F, pdf) on region A, with F a primitive of 4 q pdf."""
    s, d = q + u, q - u
    logs, dlogd = np.log(s), d * np.log(d)
    dilog = _reli2_scalar(-q / u) - _reli2_scalar(q / u)
    F = (
        -(0.75 * u * u + q * u + 0.25 * q * q) * logs
        - 0.75 * (u - q / 3.0) * dlogd
        + 0.5 * u * u * dilog
        + 4.5 * q * u
    )
    p = (1.0 + q) / (4.0 * q * a_max) * (4.0 * q - s * logs - dlogd + u * dilog)
    return float(F), float(p)


def _FP_B(u, q, a_max):
    """(F, pdf) on region B, with F a primitive of 4 q pdf."""
    s, d = u + q, u - q
    logs, dlogd = np.log(s), d * np.log(d)
    dilog = _reli2_scalar(-q / u) - _reli2_scalar(q / u)
    F = (
        -(0.75 * u * u + q * u + 0.25 * q * q) * logs
        + 0.75 * (u - q / 3.0) * dlogd
        + 0.5 * u * u * dilog
        + 4.5 * q * u
    )
    p = (1.0 + q) / (4.0 * q * a_max) * (4.0 * q - s * logs + dlogd + u * dilog)
    return float(F), float(p)


def _FP_C(u, q, a_max):
    """(F, pdf) on region C, with F a primitive of 4 q pdf."""
    u2, logu, logq = u * u, np.log(u), np.log(q)
    log1mu, logqmu = np.log1p(-u), np.log(q - u)
    dilog = _reli2_scalar(1.0 - 1.0 / u) - _reli2_scalar(q / u)
    F = (
        (-0.75 * u2 + u + 0.5 * u2 * logu) * logq
        + (0.75 * u2 - u + 0.25) * log1mu
        + (0.75 * u2 - q * u + 0.25 * q * q) * logqmu
        - 0.5 * u2 * logu * log1mu
        + 0.5 * u2 * dilog
        - 1.75 * u2
        + 2.25 * (1.0 + q) * u
    )
    p = (
        (1.0 + q)
        / (4.0 * q * a_max)
        * (
            2.0 * (1.0 + q - u)
            + (1.0 - u + u * logu) * logq
            + (u - 1.0 - u * logu) * log1mu
            + (u - q) * logqmu
            + u * dilog
        )
    )
    return float(F), float(p)


def _FP_D(u, q, a_max):
    """(F, pdf) on region D, with F a primitive of 4 q pdf."""
    u2, logu, logq = u * u, np.log(u), np.log(q)
    log1mu, logumq = np.log1p(-u), np.log(u - q)
    dilog = _reli2_scalar(1.0 - 1.0 / u) - _reli2_scalar(q / u)
    F = (
        (0.75 * u2 - q * u + 0.25 * q * q) * logumq
        + (0.75 * u2 - u + 0.25) * log1mu
        + (-0.75 * u2 + 0.5 * u2 * logu) * logq
        - 0.5 * u2 * logu * log1mu
        + 0.5 * u2 * dilog
        + (2.25 * (1.0 + q) + logq) * u
        - 1.75 * u2
    )
    p = (
        (1.0 + q)
        / (4.0 * q * a_max)
        * (
            2.0 * (1.0 + q - u)
            + (1.0 - u + u * logu) * logq
            + (u - 1.0 - u * logu) * log1mu
            + (u - q) * logumq
            + u * dilog
        )
    )
    return float(F), float(p)


def _FP_E(u, q, a_max):
    """(F, pdf) on region E, with F a primitive of 4 q pdf."""
    u2, logu, logq = u * u, np.log(u), np.log(q)
    logum1, logumq = np.log(u - 1.0), np.log(u - q)
    dilog = _reli2_scalar(1.0 - 1.0 / u) - _reli2_scalar(q / u)
    F = (
        (0.75 * u2 - u + 0.25) * logum1
        + (-0.75 * u2 + 0.5 * u2 * logu + u) * logq
        + (0.75 * u2 - q * u + 0.25 * q * q) * logumq
        - 0.5 * u2 * logu * logum1
        + 0.5 * u2 * dilog
        + 2.25 * (1.0 + q) * u
        - 1.75 * u2
    )
    p = (
        (1.0 + q)
        / (4.0 * q * a_max)
        * (
            2.0 * (1.0 + q - u)
            + (1.0 - u + u * logu) * logq
            + (u - 1.0 - u * logu) * logum1
            + (u - q) * logumq
            + u * dilog
        )
    )
    return float(F), float(p)


def _cdf_pdf_chi_eff_scalar(chi_eff, q, a_max):
    """Return the scalar (CDF, PDF) of p(chi_eff | q), sharing the special functions."""
    if chi_eff <= -a_max:
        return 0.0, 0.0
    if chi_eff >= a_max:
        return 1.0, 0.0
    if chi_eff == 0.0:
        # cdf = 1/2 and pdf = p(0) on region A
        pdf = (1.0 + q) / (2.0 * a_max) * (2.0 - np.log(q))
        return 0.5, float(pdf)

    # Regions in increasing u.  At the boundaries u = q and u = 1 the region
    # formulas contain 0 log 0, so the boundary values are used there.
    u = (1.0 + q) * abs(chi_eff) / a_max
    if q <= 0.5:  # A (0, q), B (q, 1 - q), D (1 - q, 1), E (1, 1 + q)
        if u < q:
            cumulative, pdf = _FP_A(u, q, a_max)
        elif u == q:
            cumulative, pdf = _FP_AB_at_q(q, a_max)
        elif u <= 1.0 - q:
            cumulative, pdf = _FP_B(u, q, a_max)
        elif u < 1.0:
            cumulative, pdf = _FP_D(u, q, a_max)
        elif u == 1.0:
            cumulative, pdf = _FP_DE_at_1(q, a_max)
        else:
            cumulative, pdf = _FP_E(u, q, a_max)
    else:  # A (0, 1 - q), C (1 - q, q), D (q, 1), E (1, 1 + q)
        if u <= 1.0 - q:
            cumulative, pdf = _FP_A(u, q, a_max)
        elif u < q:
            cumulative, pdf = _FP_C(u, q, a_max)
        elif u == q:
            cumulative, pdf = _FP_CD_at_q(q, a_max)
        elif u < 1.0:
            cumulative, pdf = _FP_D(u, q, a_max)
        elif u == 1.0:
            cumulative, pdf = _FP_DE_at_1(q, a_max)
        else:
            cumulative, pdf = _FP_E(u, q, a_max)
    # On C, D and E (u > 1 - q) the primitive is shifted by delta(q)
    if u > 1.0 - q:
        cumulative -= _delta_at_1_q(q)

    cdf = 0.5 + np.sign(chi_eff) * cumulative / (4.0 * q)
    return float(min(max(cdf, 0.0), 1.0)), float(max(pdf, 0.0))


# Element-wise versions of the scalar formulas (as bilby's conversion module
# does for scalar-only LAL functions).
_cdf_chi_eff_vectorized = np.vectorize(
    lambda chi_eff, q, a_max: _cdf_pdf_chi_eff_scalar(
        float(chi_eff), float(q), float(a_max)
    )[0],
    otypes=[float],
)
_pdf_chi_eff_vectorized = np.vectorize(
    lambda chi_eff, q, a_max: _cdf_pdf_chi_eff_scalar(
        float(chi_eff), float(q), float(a_max)
    )[1],
    otypes=[float],
)


def _cdf_chi_eff(chi_eff, q, a_max):
    """CDF of chi_eff given the mass ratio (0 < q <= 1); broadcasts over chi_eff and q."""
    return _cdf_chi_eff_vectorized(chi_eff, q, a_max)[()]


def _pdf_chi_eff(chi_eff, q, a_max):
    """Probability density of chi_eff given the mass ratio (0 < q <= 1); broadcasts over chi_eff and q."""
    out = _pdf_chi_eff_vectorized(chi_eff, q, a_max)[()]
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("_pdf_chi_eff produced a non-finite density.")
    return out


class ChieffTable:
    """
    Class for building or loading Inverse-CDF table of p(chi_eff | q).
    """

    def __init__(self, q_min, q_max, a_max, n_q=200, n_u=1025, dtype=np.float64):
        """
        Parameters
        ==========
        q_min, q_max: float
            Range of the mass-ratio grid.
        a_max: float
            Maximum spin magnitude; chi_eff takes values in [-a_max, a_max].
        n_q, n_u: int, optional
            Number of points of the q grid and of the u grid.
        dtype: numpy dtype, optional
            dtype of the stored inverse table.
        """
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.a_max = float(a_max)
        self.n_q = n_q
        self.n_u = n_u
        self.dtype = dtype
        self.q_grid = np.linspace(self.q_min, self.q_max, int(n_q))
        self.u_grid = np.linspace(0.0, 1.0, self.n_u)
        self.inv_table = np.empty((self.n_q, self.n_u), dtype=self.dtype)
        self.filename = None  # set by save / load; stored in the prior JSON
        self.build_table()

    def build_table(self):
        """For each q, tabulate the CDF on a uniform chi_eff grid (monotonic, from 0 to 1)
        and invert it onto the uniform u grid."""
        chi_eff_grid = -self.a_max + 2.0 * self.a_max * self.u_grid
        for i, q in enumerate(self.q_grid):
            F = np.asarray(_cdf_chi_eff(chi_eff_grid, q, self.a_max), dtype=float)
            F[0], F[-1] = 0.0, 1.0
            F = np.maximum.accumulate(np.clip(F, 0.0, 1.0))
            self.inv_table[i] = np.interp(self.u_grid, F, chi_eff_grid)

    def guess(self, u, q):
        """Initial guess of the inverse CDF: bilinear interpolation of the inverse
        table in (q, u) from 4 table values (both grids are uniform)."""
        qg = self.q_grid
        nq = len(qg)
        qc = min(max(q, qg[0]), qg[-1])
        fi = (qc - qg[0]) / (qg[-1] - qg[0]) * (nq - 1)
        i = min(int(fi), nq - 2)
        wq = fi - i
        fk = min(max(u, 0.0), 1.0) * (self.n_u - 1)
        k = min(int(fk), self.n_u - 2)
        wu = fk - k
        r0, r1 = self.inv_table[i], self.inv_table[i + 1]
        return float(
            (1.0 - wq) * ((1.0 - wu) * r0[k] + wu * r0[k + 1])
            + wq * ((1.0 - wu) * r1[k] + wu * r1[k + 1])
        )

    def settings(self):
        """Arguments of __init__ that rebuild this table (stored in the prior JSON)."""
        return dict(
            q_min=float(self.q_min),
            q_max=float(self.q_max),
            a_max=float(self.a_max),
            n_q=len(self.q_grid),
            n_u=len(self.u_grid),
            dtype=np.dtype(self.inv_table.dtype).name,
        )

    def save(self, filename):
        """Save the table to a .npz file."""
        np.savez(
            filename,
            format_version=3,
            q_min=self.q_min,
            q_max=self.q_max,
            a_max=self.a_max,
            q_grid=self.q_grid,
            u_grid=self.u_grid,
            inv_table=self.inv_table,
        )
        self.filename = os.fspath(filename)

    @classmethod
    def load(cls, filename):
        """Load a table written by ``save``."""
        with np.load(filename, allow_pickle=False) as data:
            if "inv_table" not in data.files:
                raise ValueError(
                    f"{filename} is an old incompatible chi_eff table (no inv_table). "
                    "Delete it and rebuild the table."
                )
            obj = cls.__new__(cls)
            obj.q_min = float(data["q_min"])
            obj.q_max = float(data["q_max"])
            obj.a_max = float(data["a_max"])
            obj.q_grid = np.asarray(data["q_grid"], dtype=float).copy()
            obj.u_grid = np.asarray(data["u_grid"], dtype=float).copy()
            obj.inv_table = np.asarray(data["inv_table"]).copy()

        obj.filename = os.fspath(filename)
        obj.dtype = obj.inv_table.dtype
        obj.n_q = len(obj.q_grid)
        obj.n_u = len(obj.u_grid)
        expected_shape = (obj.n_q, obj.n_u)
        if obj.inv_table.shape != expected_shape:
            raise ValueError(
                f"Invalid chi_eff table shape {obj.inv_table.shape}; "
                f"expected {expected_shape}."
            )
        return obj


class ConditionalIsotropicChieffPrior(ConditionalBasePrior):
    r"""
    Conditional prior for the effective spin chi_eff given the mass ratio q,
    for isotropic spins with magnitudes uniform in [0, a_max]:

    .. math::

        \pi(\chi_{\rm eff} | q) \propto \int d\chi_{1z} \int d\chi_{2z}
        \log\frac{a_{\max}}{|\chi_{1z}|} \log\frac{a_{\max}}{|\chi_{2z}|}
        \delta\left(\chi_{\rm eff} - \frac{\chi_{1z} + q \chi_{2z}}{1 + q}\right),

    on |chi_1z|, |chi_2z| <= a_max.
    The PDF and CDF are piecewise analytic."""

    def __init__(
        self,
        table=None,
        name="chi_eff",
        latex_label=r"$\chi_{\rm eff}$",
        unit=None,
        boundary=None,
        n_newton=5,
        table_file=None,
        table_settings=None,
    ):
        """
        Parameters
        ==========
        table: ChieffTable
            Inverse-CDF table for the initial guess. If None, it is loaded from
            ``table_file`` or rebuilt from ``table_settings``.
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        n_newton: int, optional
            Maximum number of Newton steps.
        table_file: str, optional
            Path of the saved table (stored in the prior JSON).
        table_settings: dict, optional
            table.settings(), used to rebuild the table if ``table_file`` is
            missing or broken (stored in the prior JSON).
        """
        table = _table_for_prior(table, table_file, ChieffTable, table_settings)
        self.table = table
        if table_file is None:
            table_file = table.filename
        self.table_file = os.path.abspath(os.fspath(table_file))
        self.table_settings = table.settings()
        self.a_max = table.a_max
        self._current_mass_ratio = None
        self.n_newton = n_newton
        self.scalar_types = (float, int, np.floating, np.integer)
        super().__init__(
            minimum=-self.a_max,
            maximum=self.a_max,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
            condition_func=self._condition_function,
        )

        self.__class__.__name__ = "ConditionalIsotropicChieffPrior"
        self.__class__.__qualname__ = "ConditionalIsotropicChieffPrior"

    def _condition_function(self, reference_params, mass_ratio):
        """Cache the mass ratio"""
        self._current_mass_ratio = np.asarray(mass_ratio, dtype=float)
        return dict(minimum=-self.a_max, maximum=self.a_max)

    def _get_conditions(self, required_variables):
        """Update the conditions with bilby's update_conditions (which also caches
        the mass ratio in _condition_function) and return the mass ratio."""
        self.update_conditions(**required_variables)
        if self._current_mass_ratio is None:
            raise KeyError("No cached mass_ratio found.")
        return self._current_mass_ratio

    def prob(self, val, *, xp=np, **rv):
        """Return the prior probability of val."""
        return _pdf_chi_eff(val, self._get_conditions(rv), self.a_max)

    def ln_prob(self, val, *, xp=np, **rv):
        """Return the log prior probability of val."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(self.prob(val, **rv))

    def cdf(self, val, *, xp=np, **rv):
        """Return the cumulative distribution function at val."""
        return _cdf_chi_eff(val, self._get_conditions(rv), self.a_max)

    def _icdf(self, u, q):
        """Inverse CDF at one point (Python floats u, q): table guess, Newton, and Brent if needed."""
        a = self.a_max
        if u <= 0.0:
            return -a
        if u >= 1.0:
            return a

        def cdf_pdf(x):
            return _cdf_pdf_chi_eff_scalar(x, q, a)

        xk, F, converged = _newton_update(
            self.table.guess(u, q), u, -a, a, cdf_pdf, self.n_newton
        )
        if not converged:
            xk = _brent_update(xk, F, u, -a, a, lambda x: cdf_pdf(x)[0])
        return min(max(xk, -a), a)

    def rescale(self, val, *, xp=np, **rv):
        """Map a sample from the unit interval to the prior (inverse CDF).
        Scalars (as from dynesty) call _icdf directly; arrays go through np.vectorize."""
        if isinstance(val, self.scalar_types) and isinstance(
            rv.get("mass_ratio"), self.scalar_types
        ):
            u, q = float(val), float(rv["mass_ratio"])
            self._current_mass_ratio = q
            return self._icdf(u, q)
        q = self._get_conditions(rv)
        val = np.asarray(val, dtype=float)
        icdf = np.vectorize(
            lambda uu, qq: self._icdf(float(uu), float(qq)), otypes=[float]
        )
        return icdf(val, q)[()]

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        instantiation_dict.pop("table")
        return instantiation_dict

    def __repr__(self):
        return Prior.__repr__(self)


# ============================================================
# p(chi_2 | chi_eff, q)
#
# With chi_1z = (1 + q) chi_eff - q chi_2 the density is proportional to
# log(a_max / |chi_1z|) log(a_max / |chi_2|) on |chi_1z|, |chi_2| <= a_max.
# In t = chi_2 / a_max and alpha = (1 + q) chi_eff / a_max it is
# log|t| log|alpha - q t|, with the antiderivative _func_scalar.
#
# _chi2_precompute collects everything that does not depend on chi_2 (support,
# normalisation, series coefficients), so that the table rows and the
# inverse-CDF iterations compute it once per (chi_eff, q);
# _cdf_pdf_chi2_scalar then returns (CDF, PDF) at one chi_2.
# ============================================================


def _reli2_for_table(x):
    """Array version of _reli2_scalar (real part of Li2), for building the table:
    spence(1 - x) for x <= 1, and the inversion formula for x > 1."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverted = (
            np.pi**2 / 3.0 - 0.5 * np.log(np.abs(x)) ** 2 - spence(1.0 - 1.0 / x)
        )  # Re Li2(x) for x > 1
    return np.where(x <= 1.0, spence(1.0 - x), inverted)


def _func_for_table(t, alpha, q):
    """Array version of _func_scalar (t an array, alpha and q scalars), for building the table."""
    t = np.asarray(t, dtype=float)
    t0 = alpha / q
    with np.errstate(divide="ignore", invalid="ignore"):
        tlogt = np.where(t == 0.0, 0.0, t * np.log(np.abs(t)))  # t log|t|
        if abs(t0) < 1e-300:  # In case of chi_eff = 0
            return (
                (tlogt - t) * np.log(np.maximum(np.abs(q * t), 1e-300))
                - tlogt
                + 2.0 * t
            )
        tau = alpha - q * t
        dh = tlogt - t + t0 - t0 * np.log(abs(t0))  # h(t) - h(t0)
        log_term = np.where(tau == 0.0, 0.0, dh * np.log(np.abs(tau)))
    return log_term - tlogt + 2.0 * t + t0 * _reli2_for_table(1.0 - t / t0)


def _cdf_chi2_for_table(chi2, pre):
    """CDF of chi_2 over an array chi2 for one tuple from _chi2_precompute
    (array version of _cdf_pdf_chi2_scalar(...)[0], for the table)."""
    kind, lower, upper = pre[0], pre[1], pre[2]
    chi2 = np.asarray(chi2, dtype=float)
    if kind == "point":
        return np.where(chi2 >= 0.5 * (lower + upper), 1.0, 0.0)
    inside = (chi2 > lower) & (chi2 < upper)
    F = np.zeros_like(chi2)
    if kind == "narrow":
        _, _, _, width, coef, coef_beta, ps, p1, pdf_scale = pre
        s = (chi2[inside] - lower) / width
        F[inside] = coef_beta @ betainc(
            ps[:, None] + 1.0, p1[:, None] + 1.0, s[None, :]
        )
    else:
        _, _, _, alpha, q, a_max, P0, span = pre
        F[inside] = (_func_for_table(chi2[inside] / a_max, alpha, q) - P0) / span
    F[chi2 >= upper] = 1.0
    return np.clip(F, 0.0, 1.0)


def _func_scalar(t, alpha, q):
    """
    Antiderivative F(t) of log|t| log|alpha - q t| (unnormalised CDF of chi_2):

        F(t) = [h(t) - h(t0)] log|alpha - q t| - t log|t| + 2 t + t0 Re Li2(1 - t / t0),

    where
        t0 = alpha / q,
        h(x) = x log|x| - x.

    In case of t0 = 0 (chi_eff = 0), F(t) uses the limit F(t) = h(t) log|q t| - t log|t| + 2 t,
    and 0 log 0 is taken as 0.
    """
    tlogt = t * np.log(abs(t)) if t != 0.0 else 0.0  # t log|t|
    t0 = alpha / q

    if abs(t0) < 1e-300:  # In case of chi_eff = 0
        return float((tlogt - t) * np.log(max(abs(q * t), 1e-300)) - tlogt + 2.0 * t)

    tau = alpha - q * t
    dh = tlogt - t + t0 - t0 * np.log(abs(t0))  # h(t) - h(t0)
    log_term = dh * np.log(abs(tau)) if tau != 0.0 else 0.0
    return float(log_term - tlogt + 2.0 * t + t0 * _reli2_scalar(1.0 - t / t0))


def _chi2_support_scalar(chi_eff, q, a_max):
    """Support (lower, upper) of chi_2 given chi_eff and q: |chi_2|, |chi_1z| <= a_max."""
    s = (1.0 + q) * chi_eff
    return max(-a_max, (s - a_max) / q), min(a_max, (s + a_max) / q)


_chi2_support_vectorized = np.vectorize(
    lambda chi_eff, q, a_max: _chi2_support_scalar(
        float(chi_eff), float(q), float(a_max)
    ),
    otypes=[float, float],
)


# p(chi_2 | chi_eff, q): with support width w, A = w / a_max and
# B = (1 + q)(1 - |chi_eff| / a_max) (= q A), the CDF/PDF use
#   "point"  if w <= CHI2_DEGENERATE_REL_TOL * a_max  (step CDF, PDF = 0),
#   "narrow" if max(A, B) <= CHI2_EDGE_SERIES_MAX_ARGUMENT  (Beta series),
#   "wide"   otherwise  (analytic antiderivative).
CHI2_DEGENERATE_REL_TOL = 64.0 * np.finfo(float).eps  # chi_eff = +-a_max up to rounding
# The analytic CDF loses accuracy as the support narrows (~3e-12 at A = 0.25);
# the order-16 series has error ~2e-13 (CDF) and ~1e-11 (PDF, relative) there.
# Re-validate both values against mpmath when changing either.
CHI2_EDGE_SERIES_MAX_ARGUMENT = 0.25
CHI2_EDGE_SERIES_ORDER = 16
# Terms of sum_{m,n=1}^{N} A^m B^n / (m n) s^ps (1 - s)^p1 with s = (chi_2 - lower) / w,
# (ps, p1) = (n, m) for chi_eff >= 0 and (m, n) otherwise; Beta(ps + 1, p1 + 1) is symmetric.
_SERIES_M, _SERIES_N = (
    x.ravel().astype(float)
    for x in np.meshgrid(
        np.arange(1, CHI2_EDGE_SERIES_ORDER + 1),
        np.arange(1, CHI2_EDGE_SERIES_ORDER + 1),
        indexing="ij",
    )
)
_SERIES_BETA = beta_function(_SERIES_M + 1.0, _SERIES_N + 1.0)


def _chi2_precompute(chi_eff, q, a_max):
    """Precompute the parts of the chi_2 CDF/PDF that do not depend on chi_2.

    The support [lower, upper] of chi_2 given (chi_eff, q) falls into one of 3 cases below:

    "point":  the support has (numerically) zero width, width <= CHI2_DEGENERATE_REL_TOL * a_max
              (e.g. chi_eff = +-a_max).
    "narrow": the support is narrow (e.g. |chi_eff| close to a_max).
              The CDF/PDF are evaluated with the truncated double Beta series for numerical stability.
    "wide":   otherwise. The CDF is evaluated with the analytic antiderivative (_func_scalar).

    Returns
    -------
    ("point", lower, upper)
    ("narrow", lower, upper, width, coef, coef_beta / normalization, ps, p1, 1 / (width * normalization))
    ("wide", lower, upper, alpha, q, a_max, F0, F_width)
    """
    lower, upper = _chi2_support_scalar(chi_eff, q, a_max)
    width = upper - lower
    if not width > CHI2_DEGENERATE_REL_TOL * a_max:
        return ("point", lower, upper)
    A = width / a_max
    B = (1.0 + q) * max(0.0, 1.0 - abs(chi_eff) / a_max)
    if max(A, B) <= CHI2_EDGE_SERIES_MAX_ARGUMENT:
        coef = A**_SERIES_M * B**_SERIES_N / (_SERIES_M * _SERIES_N)
        coef_beta = coef * _SERIES_BETA
        normalization = float(np.sum(coef_beta))
        if not np.isfinite(normalization):
            raise FloatingPointError(
                "Failed to normalize the narrow-support chi_2 series."
            )
        ps, p1 = (_SERIES_N, _SERIES_M) if chi_eff >= 0.0 else (_SERIES_M, _SERIES_N)
        return (
            "narrow",
            lower,
            upper,
            width,
            coef,
            coef_beta / normalization,
            ps,
            p1,
            1.0 / (width * normalization),
        )
    alpha = (1.0 + q) * chi_eff / a_max
    F0 = _func_scalar(lower / a_max, alpha, q)
    F1 = _func_scalar(upper / a_max, alpha, q)
    F_width = F1 - F0
    if not np.isfinite(F_width):
        raise FloatingPointError(
            f"Failed to compute the normalization of p(chi2 | chi_eff={chi_eff}, q={q})."
        )
    return ("wide", lower, upper, alpha, q, a_max, F0, F_width)


def _cdf_pdf_chi2_scalar(chi2, pre):
    """Return the scalar (CDF, PDF) of chi_2z, where
    pre = _chi2_precompute(chi_eff, q, a_max) holds the parts that depend only
    on (chi_eff, q).  The logarithmic singularities of the density at
    chi_2z = 0 or chi_1z = 0 are represented as zero (a measure-zero convention)."""
    kind, lower, upper = pre[0], pre[1], pre[2]

    if kind == "point":  # In case of degenerate support
        return (1.0 if chi2 >= 0.5 * (lower + upper) else 0.0), 0.0
    if chi2 <= lower:
        return 0.0, 0.0
    if chi2 >= upper:
        return 1.0, 0.0

    if kind == "narrow":  # In case of narrow support
        _, _, _, width, coef, coef_beta, ps, p1, pdf_scale = pre
        s = (chi2 - lower) / width
        F = float(np.dot(coef_beta, betainc(ps + 1.0, p1 + 1.0, s)))
        pdf = float(np.dot(coef, s**ps * (1.0 - s) ** p1)) * pdf_scale

    elif kind == "wide":  # In case of wide support(usual case)
        _, _, _, alpha, q, a_max, F0, F_width = pre
        t = chi2 / a_max
        tau = alpha - q * t
        F = (_func_scalar(t, alpha, q) - F0) / F_width
        pdf = (
            0.0
            if t == 0.0 or tau == 0.0
            else np.log(abs(t)) * np.log(abs(tau)) / (a_max * F_width)
        )

    return float(min(max(F, 0.0), 1.0)), float(max(pdf, 0.0))


# Element-wise versions of the scalar formulas (as for chi_eff above).
_cdf_chi2_vectorized = np.vectorize(
    lambda chi2, chi_eff, q, a_max: _cdf_pdf_chi2_scalar(
        float(chi2), _chi2_precompute(float(chi_eff), float(q), float(a_max))
    )[0],
    otypes=[float],
)
_pdf_chi2_vectorized = np.vectorize(
    lambda chi2, chi_eff, q, a_max: _cdf_pdf_chi2_scalar(
        float(chi2), _chi2_precompute(float(chi_eff), float(q), float(a_max))
    )[1],
    otypes=[float],
)


def _cdf_chi2(chi2, chi_eff, q, a_max):
    """CDF of chi_2 given chi_eff and the mass ratio; broadcasts over chi2, chi_eff and q."""
    return _cdf_chi2_vectorized(chi2, chi_eff, q, a_max)[()]


def _pdf_chi2(chi2, chi_eff, q, a_max):
    """Probability density of chi_2 given chi_eff and the mass ratio; broadcasts over chi2, chi_eff and q."""
    out = _pdf_chi2_vectorized(chi2, chi_eff, q, a_max)[()]
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("_pdf_chi2 produced a non-finite density.")
    return out


class Chi2zTable:
    """
    Class for building or loading Inverse-CDF table of p(chi_2z | chi_eff, q).
    """

    def __init__(
        self, q_min, q_max, a_max, n_q=120, n_chieff=301, n_u=801, dtype=np.float32
    ):
        """
        Parameters
        ==========
        q_min, q_max: float
            Range of the mass-ratio grid.
        a_max: float
            Maximum spin magnitude; the chi_eff grid covers [-a_max, a_max].
        n_q, n_chieff, n_u: int, optional
            Number of points of the q, chi_eff and u grids.
        dtype: numpy dtype, optional
            dtype of the stored inverse table (default: float32).
        """
        self.q_min = float(q_min)
        self.q_max = float(q_max)
        self.a_max = float(a_max)
        self.n_q = n_q
        self.n_chieff = n_chieff
        self.n_u = n_u
        self.dtype = dtype
        self.q_grid = np.linspace(self.q_min, self.q_max, int(n_q))
        self.chieff_grid = np.linspace(-self.a_max, self.a_max, int(n_chieff))
        self.u_grid = np.linspace(0.0, 1.0, self.n_u)
        self.inv_table = np.empty((self.n_q, self.n_chieff, self.n_u), dtype=self.dtype)
        self.filename = None  # set by save / load; stored in the prior JSON
        self.build_table()

    def build_table(self):
        """For each (q, chi_eff), tabulate the CDF on the uniform s grid (monotonic,
        from 0 to 1) and invert it onto the uniform u grid.  At |chi_eff| = a_max
        the support degenerates; its rows use the limiting shape 3 s^2 - 2 s^3 of
        the Beta series.

        On the table rows s is always the u grid, so the regularised incomplete
        Beta functions of the narrow-support series depend only on the sign of
        chi_eff (which fixes (ps, p1)); they are computed once here and each
        narrow row is a matrix-vector product."""
        betainc_s = {
            sign: betainc(ps[:, None] + 1.0, p1[:, None] + 1.0, self.u_grid[None, :])
            for sign, (ps, p1) in (
                (1, (_SERIES_N, _SERIES_M)),  # chi_eff >= 0
                (-1, (_SERIES_M, _SERIES_N)),  # chi_eff < 0
            )
        }
        for i, q in enumerate(self.q_grid):
            if i % max(1, len(self.q_grid) // 10) == 0:
                logger.info(f"Building chi_2 table: q row {i + 1}/{len(self.q_grid)}")
            for j, chi_eff in enumerate(self.chieff_grid):
                pre = _chi2_precompute(chi_eff, q, self.a_max)
                if pre[0] == "point":
                    # Degenerate support (|chi_eff| = a_max): use the limiting shape of the
                    # Beta series in s, 3 s^2 - 2 s^3, so that the table is continuous in chi_eff.
                    F_row = 3.0 * self.u_grid**2 - 2.0 * self.u_grid**3
                elif pre[0] == "narrow":
                    coef_beta = pre[5]
                    F_row = coef_beta @ betainc_s[1 if chi_eff >= 0.0 else -1]
                    F_row = np.maximum.accumulate(np.clip(F_row, 0.0, 1.0))
                else:
                    lower, upper = pre[1], pre[2]
                    chi2_grid = lower + (upper - lower) * self.u_grid
                    F_row = np.maximum.accumulate(
                        np.clip(_cdf_chi2_for_table(chi2_grid, pre), 0.0, 1.0)
                    )
                F_row[0] = 0.0
                F_row[-1] = 1.0
                self.inv_table[i, j, :] = np.interp(self.u_grid, F_row, self.u_grid)

    @staticmethod
    def _frac_index(x, grid):
        """Return (i, w) with x = (1 - w) grid[i] + w grid[i + 1] on a uniform grid, x clipped to the grid."""
        n = len(grid)
        xc = min(max(x, grid[0]), grid[-1])
        f = (xc - grid[0]) / (grid[-1] - grid[0]) * (n - 1)
        i = min(int(f), n - 2)
        return i, f - i

    def guess(self, u, chi_eff, q):
        """Initial guess of the inverse CDF:
        trilinear interpolation of the inverse table in (q, chi_eff, u) from 8 table values."""
        i, wq = self._frac_index(q, self.q_grid)
        j, wc = self._frac_index(chi_eff, self.chieff_grid)
        fk = min(max(u, 0.0), 1.0) * (self.n_u - 1)
        k = min(int(fk), self.n_u - 2)
        wu = fk - k
        T = self.inv_table

        def lin(r):
            return (1.0 - wu) * float(r[k]) + wu * float(r[k + 1])

        return (1.0 - wq) * ((1.0 - wc) * lin(T[i, j]) + wc * lin(T[i, j + 1])) + wq * (
            (1.0 - wc) * lin(T[i + 1, j]) + wc * lin(T[i + 1, j + 1])
        )

    def settings(self):
        """Arguments of __init__ that rebuild this table (stored in the prior JSON)."""
        return dict(
            q_min=float(self.q_min),
            q_max=float(self.q_max),
            a_max=float(self.a_max),
            n_q=len(self.q_grid),
            n_chieff=len(self.chieff_grid),
            n_u=len(self.u_grid),
            dtype=np.dtype(self.inv_table.dtype).name,
        )

    def save(self, filename):
        """Save the table to a .npz file."""
        np.savez(
            filename,
            format_version=3,
            q_min=self.q_min,
            q_max=self.q_max,
            a_max=self.a_max,
            q_grid=self.q_grid,
            chieff_grid=self.chieff_grid,
            u_grid=self.u_grid,
            inv_table=self.inv_table,
        )
        self.filename = os.fspath(filename)

    @classmethod
    def load(cls, filename):
        """Load a table written by ``save``."""
        with np.load(filename, allow_pickle=False) as data:
            if "inv_table" not in data.files:
                raise ValueError(
                    f"{filename} is an old incompatible chi_2 table (no inv_table). "
                    "Delete it and rebuild the table."
                )
            obj = cls.__new__(cls)
            obj.q_min = float(data["q_min"])
            obj.q_max = float(data["q_max"])
            obj.a_max = float(data["a_max"])
            obj.q_grid = np.asarray(data["q_grid"], dtype=float).copy()
            obj.chieff_grid = np.asarray(data["chieff_grid"], dtype=float).copy()
            obj.u_grid = np.asarray(data["u_grid"], dtype=float).copy()
            obj.inv_table = np.asarray(data["inv_table"]).copy()

        obj.filename = os.fspath(filename)
        obj.dtype = obj.inv_table.dtype
        obj.n_q = len(obj.q_grid)
        obj.n_chieff = len(obj.chieff_grid)
        obj.n_u = len(obj.u_grid)
        expected_shape = (obj.n_q, obj.n_chieff, obj.n_u)
        if obj.inv_table.shape != expected_shape:
            raise ValueError(
                f"Invalid chi_2 table shape {obj.inv_table.shape}; "
                f"expected {expected_shape}."
            )
        return obj


class ConditionalIsotropicChi2zPrior(ConditionalBasePrior):
    r"""
    Conditional prior for the aligned spin component chi_2z given chi_eff and
    mass ratio, for isotropic spins with magnitudes uniform in [0, a_max]:

    .. math::

        \pi(\chi_{2z} | \chi_{\rm eff}, q)
        \propto \log\frac{a_{\max}}{|\chi_{1z}|} \log\frac{a_{\max}}{|\chi_{2z}|},
        \qquad \chi_{1z} = (1 + q) \chi_{\rm eff} - q \chi_{2z},

    on |chi_1z|, |chi_2z| <= a_max.
    """

    def __init__(
        self,
        table=None,
        name="chi_2",
        latex_label=r"$\chi_2$",
        unit=None,
        boundary=None,
        n_newton=5,
        table_file=None,
        table_settings=None,
    ):
        """
        Parameters
        ==========
        table: Chi2zTable
            Inverse-CDF table for the initial guess. If None, it is loaded from
            ``table_file`` (or rebuilt from ``table_settings``).
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        n_newton: int, optional
            Maximum number of Newton steps.
        table_file: str, optional
            Path of the saved table (stored in the prior JSON).
        table_settings: dict, optional
            table.settings(), used to rebuild the table if ``table_file`` is
            missing or broken (stored in the prior JSON).
        """
        table = _table_for_prior(table, table_file, Chi2zTable, table_settings)
        self.table = table
        if table_file is None:
            table_file = table.filename
        self.table_file = os.path.abspath(os.fspath(table_file))
        self.table_settings = table.settings()
        self.a_max = table.a_max
        self._current_chi_eff = None
        self._current_mass_ratio = None
        self.n_newton = n_newton
        self.scalar_types = (float, int, np.floating, np.integer)
        super().__init__(
            minimum=-self.a_max,
            maximum=self.a_max,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
            condition_func=self._condition_function,
        )

        self.__class__.__name__ = "ConditionalIsotropicChi2zPrior"
        self.__class__.__qualname__ = "ConditionalIsotropicChi2zPrior"

    def _condition_function(self, reference_params, chi_eff, mass_ratio):
        """Cache (chi_eff, mass_ratio) and return the support of chi_2."""
        chi_eff = np.asarray(chi_eff, dtype=float)
        q = np.asarray(mass_ratio, dtype=float)
        self._current_chi_eff = chi_eff
        self._current_mass_ratio = q
        lower, upper = _chi2_support_vectorized(chi_eff, q, self.a_max)
        return dict(minimum=lower, maximum=upper)

    def _get_conditions(self, required_variables):
        """Update the conditions with bilby's update_conditions (which also caches
        chi_eff and the mass ratio in _condition_function) and return them."""
        self.update_conditions(**required_variables)
        if self._current_chi_eff is None or self._current_mass_ratio is None:
            raise KeyError("No cached chi_eff / mass_ratio found.")
        return self._current_chi_eff, self._current_mass_ratio

    def prob(self, val, *, xp=np, **rv):
        """Return the prior probability of val."""
        chi_eff, q = self._get_conditions(rv)
        return _pdf_chi2(val, chi_eff, q, self.a_max)

    def ln_prob(self, val, *, xp=np, **rv):
        """Return the log prior probability of val."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(self.prob(val, **rv))

    def cdf(self, val, *, xp=np, **rv):
        """Return the cumulative distribution function at val."""
        chi_eff, q = self._get_conditions(rv)
        return _cdf_chi2(val, chi_eff, q, self.a_max)

    def _icdf(self, u, chi_eff, q):
        """Inverse CDF at one point (Python floats u, chi_eff, q): table guess, Newton,
        and Brent if needed."""
        pre = _chi2_precompute(chi_eff, q, self.a_max)
        lower, upper = pre[1], pre[2]
        if pre[0] == "point":
            return 0.5 * (lower + upper)
        if u <= 0.0:
            return lower
        if u >= 1.0:
            return upper

        def cdf_pdf(x):
            return _cdf_pdf_chi2_scalar(x, pre)

        x0 = lower + (upper - lower) * self.table.guess(u, chi_eff, q)
        xk, F, converged = _newton_update(x0, u, lower, upper, cdf_pdf, self.n_newton)
        if not converged:
            xk = _brent_update(xk, F, u, lower, upper, lambda x: cdf_pdf(x)[0])
        return min(max(xk, lower), upper)

    def rescale(self, val, *, xp=np, **rv):
        """Map a sample from the unit interval to the prior (inverse CDF).
        Scalars (as from dynesty) call _icdf directly; arrays go through np.vectorize."""
        if (
            isinstance(val, self.scalar_types)
            and isinstance(rv.get("chi_eff"), self.scalar_types)
            and isinstance(rv.get("mass_ratio"), self.scalar_types)
        ):
            u, ce, q = float(val), float(rv["chi_eff"]), float(rv["mass_ratio"])
            self._current_chi_eff, self._current_mass_ratio = ce, q
            return self._icdf(u, ce, q)
        chi_eff, q = self._get_conditions(rv)
        icdf = np.vectorize(
            lambda uu, cc, qq: self._icdf(float(uu), float(cc), float(qq)),
            otypes=[float],
        )
        return icdf(val, chi_eff, q)[()]

    def chi1_from_chi2(self, chi2, **rv):
        """Return chi_1z = (1 + q) chi_eff - q chi_2 for the given (or cached) conditions."""
        chi_eff, q = self._get_conditions(rv)
        return (1.0 + q) * chi_eff - q * chi2

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        instantiation_dict.pop("table")
        return instantiation_dict

    def __repr__(self):
        return Prior.__repr__(self)


class EOSCheck(Constraint):
    def __init__(self, minimum=-np.inf, maximum=np.inf):
        """
        Constraint used for EoS sampling. Converts the result of various
        checks on the EoS and its parameters into a prior that can reject
        unphysical samples. Necessary for EoS sampling.
        """

        super().__init__(
            minimum=minimum, maximum=maximum, name=None, latex_label=None, unit=None
        )

    def prob(self, val, *, xp=np):
        """
        Returns the result of the equation of state check in the conversion function.
        """
        return val

    def ln_prob(self, val, *, xp=np):
        if val:
            result = 0.0
        elif not val:
            result = -np.inf

        return result


class CBCPriorDict(ConditionalPriorDict):
    @property
    def minimum_chirp_mass(self):
        mass_1 = None
        mass_2 = None
        if "chirp_mass" in self:
            return self["chirp_mass"].minimum
        elif "mass_1" in self:
            mass_1 = self["mass_1"].minimum
            if "mass_2" in self:
                mass_2 = self["mass_2"].minimum
            elif "mass_ratio" in self:
                mass_2 = mass_1 * self["mass_ratio"].minimum
        if mass_1 is not None and mass_2 is not None:
            s = generate_mass_parameters(dict(mass_1=mass_1, mass_2=mass_2))
            return s["chirp_mass"]
        else:
            logger.warning("Unable to determine minimum chirp mass")
            return None

    @property
    def maximum_chirp_mass(self):
        mass_1 = None
        mass_2 = None
        if "chirp_mass" in self:
            return self["chirp_mass"].maximum
        elif "mass_1" in self:
            mass_1 = self["mass_1"].maximum
            if "mass_2" in self:
                mass_2 = self["mass_2"].maximum
            elif "mass_ratio" in self:
                mass_2 = mass_1 * self["mass_ratio"].maximum
        if mass_1 is not None and mass_2 is not None:
            s = generate_mass_parameters(dict(mass_1=mass_1, mass_2=mass_2))
            return s["chirp_mass"]
        else:
            logger.warning("Unable to determine maximum chirp mass")
            return None

    @property
    def minimum_component_mass(self):
        """
        The minimum component mass allowed for the prior dictionary.

        This property requires either:
        * a prior for :code:`mass_2`
        * priors for :code:`chirp_mass` and :code:`mass_ratio`

        Returns
        -------
        mass_2: float
            The minimum allowed component mass.
        """
        if "mass_2" in self:
            return self["mass_2"].minimum
        if "chirp_mass" in self and "mass_ratio" in self:
            total_mass = chirp_mass_and_mass_ratio_to_total_mass(
                self["chirp_mass"].minimum, self["mass_ratio"].minimum
            )
            _, mass_2 = total_mass_and_mass_ratio_to_component_masses(
                self["mass_ratio"].minimum, total_mass
            )
            return mass_2
        else:
            logger.warning("Unable to determine minimum component mass")
            return None

    def is_nonempty_intersection(self, pset):
        """Check if keys in self exist in the parameter set

        Parameters
        ----------
        pset: str, set
            Either a string referencing a parameter set in PARAMETER_SETS or
            a set of keys
        """
        if isinstance(pset, str) and pset in PARAMETER_SETS:
            check_set = PARAMETER_SETS[pset]
        elif isinstance(pset, set):
            check_set = pset
        else:
            raise ValueError(f"pset {pset} not understood")
        if len(check_set.intersection(self.non_fixed_keys)) > 0:
            return True
        else:
            return False

    @property
    def spin(self):
        """Return true if priors include any spin parameters"""
        return self.is_nonempty_intersection("spin")

    @property
    def precession(self):
        """Return true if priors include any precession parameters"""
        return self.is_nonempty_intersection("precession_only")

    @property
    def measured_spin(self):
        """Return true if priors include any measured_spin parameters"""
        return self.is_nonempty_intersection("measured_spin")

    @property
    def intrinsic(self):
        """Return true if priors include any intrinsic parameters"""
        return self.is_nonempty_intersection("intrinsic")

    @property
    def extrinsic(self):
        """Return true if priors include any extrinsic parameters"""
        return self.is_nonempty_intersection("extrinsic")

    @property
    def sky(self):
        """Return true if priors include any extrinsic parameters"""
        return self.is_nonempty_intersection("sky")

    @property
    def distance_inclination(self):
        """Return true if priors include any extrinsic parameters"""
        return self.is_nonempty_intersection("distance_inclination")

    @property
    def mass(self):
        """Return true if priors include any mass parameters"""
        return self.is_nonempty_intersection("mass")

    @property
    def phase(self):
        """Return true if priors include phase parameters"""
        return self.is_nonempty_intersection("phase")

    @property
    def _cosmological_priors(self):
        return [key for key, prior in self.items() if isinstance(prior, Cosmological)]

    @property
    def is_cosmological(self):
        """Return True if any of the priors are cosmological."""
        if self._cosmological_priors:
            return True
        else:
            return False

    @property
    def cosmology(self):
        """The cosmology used in the priors."""
        if self.is_cosmological:
            return self[self._cosmological_priors[0]].cosmology
        else:
            return None

    def check_valid_cosmology(self, error=True, warning=False):
        """Check that all cosmological priors use the same cosmology.

        .. versionadded:: 2.5.0

        Parameters
        ==========
        error: bool
            Whether to raise a ValueError on failure.
        warning: bool
            Whether to log a warning on failure.

        Returns
        =======
        bool: whether the cosmological priors are valid.

        Raises
        ======
        ValueError: if error is True and the cosmological priors are invalid.
        """
        cosmological_priors = self._cosmological_priors
        if not cosmological_priors:
            return True

        from astropy.cosmology import cosmology_equal

        cosmologies = [self[key].cosmology for key in self._cosmological_priors]
        if all(
            cosmology_equal(cosmologies[0], c, allow_equivalent=True)
            for c in cosmologies[1:]
        ):
            return True

        message = (
            f"All cosmological priors must use the same cosmology. Found: {cosmologies}"
        )
        if warning:
            logger.warning(message)
            return False
        if error:
            raise ValueError(message)

    def validate_prior(
        self, duration, minimum_frequency, N=1000, error=True, warning=False
    ):
        """Validate the prior is suitable for use

        Parameters
        ==========
        duration: float
            The data duration in seconds
        minimum_frequency: float
            The minimum frequency in Hz of the analysis
        N: int
            The number of samples to draw when checking
        error: bool
            Whether to raise a ValueError on failure.
        warning: bool
            Whether to log a warning on failure.

        Returns
        =======
        bool: whether the template will fit within the segment duration
        """
        samples = self.sample(N)
        samples = generate_all_bbh_parameters(samples)
        durations = np.array(
            [
                calculate_time_to_merger(
                    frequency=minimum_frequency,
                    mass_1=mass_1,
                    mass_2=mass_2,
                )
                for (mass_1, mass_2) in zip(samples["mass_1"], samples["mass_2"])
            ]
        )
        longest_duration = max(durations)
        if longest_duration < duration:
            return True
        message = (
            "Prior contains regions of parameter space in which the signal"
            f" is longer than the data duration {duration}s."
            f" Maximum estimated duration = {longest_duration:.1f}s."
        )
        if warning:
            logger.warning(message)
            return False
        if error:
            raise ValueError(message)


class BBHPriorDict(CBCPriorDict):
    def __init__(
        self,
        dictionary=None,
        filename=None,
        aligned_spin=False,
        conversion_function=None,
    ):
        """Initialises a Prior set for Binary Black holes

        Parameters
        ==========
        dictionary: dict, optional
            See superclass
        filename: str, optional
            See superclass
        conversion_function: func
            Function to convert between sampled parameters and constraints.
            By default this generates many additional parameters, see
            BBHPriorDict.default_conversion_function
        """
        if dictionary is None and filename is None:
            if aligned_spin:
                fname = "aligned_spins_bbh.prior"
                logger.info("Using aligned spin prior")
            else:
                fname = "precessing_spins_bbh.prior"
            filename = os.path.join(DEFAULT_PRIOR_DIR, fname)
            logger.info(
                "No prior given, using default BBH priors in {}.".format(filename)
            )
        elif filename is not None:
            if not os.path.isfile(filename):
                filename = os.path.join(DEFAULT_PRIOR_DIR, filename)
        super(BBHPriorDict, self).__init__(
            dictionary=dictionary,
            filename=filename,
            conversion_function=conversion_function,
        )

    def default_conversion_function(self, sample):
        """
        Default parameter conversion function for BBH signals.

        This generates:
        - the parameters passed to source.lal_binary_black_hole
        - all mass parameters

        It does not generate:
        - component spins
        - source-frame parameters

        Parameters
        ==========
        sample: dict
            Dictionary to convert

        Returns
        =======
        sample: dict
            Same as input
        """
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_black_hole_parameters(out_sample)
        out_sample = generate_mass_parameters(out_sample)

        return out_sample

    def test_redundancy(self, key, disable_logging=False):
        """
        Test whether adding the key would add be redundant.
        Already existing keys return True.

        Parameters
        ==========
        key: str
            The key to test.
        disable_logging: bool, optional
            Disable logging in this function call. Default is False.

        Returns
        =======
        redundant: bool
            Whether the key is redundant or not
        """
        if key in self:
            logger.debug("{} already in prior".format(key))
            return True

        sampling_parameters = {
            key
            for key in self
            if not isinstance(self[key], (DeltaFunction, Constraint))
        }

        mass_parameters = {
            "mass_1",
            "mass_2",
            "chirp_mass",
            "total_mass",
            "mass_ratio",
            "symmetric_mass_ratio",
        }
        spin_tilt_1_parameters = {"tilt_1", "cos_tilt_1"}
        spin_tilt_2_parameters = {"tilt_2", "cos_tilt_2"}
        spin_azimuth_parameters = {"phi_1", "phi_2", "phi_12", "phi_jl"}
        inclination_parameters = {"theta_jn", "cos_theta_jn"}
        distance_parameters = {"luminosity_distance", "comoving_distance", "redshift"}

        for independent_parameters, parameter_set in zip(
            [2, 2, 1, 1, 1, 1],
            [
                mass_parameters,
                spin_azimuth_parameters,
                spin_tilt_1_parameters,
                spin_tilt_2_parameters,
                inclination_parameters,
                distance_parameters,
            ],
        ):
            if key in parameter_set:
                if (
                    len(parameter_set.intersection(sampling_parameters))
                    >= independent_parameters
                ):
                    logger.disabled = disable_logging
                    logger.warning(
                        "{} already in prior. "
                        "This may lead to unexpected behaviour.".format(
                            parameter_set.intersection(self)
                        )
                    )
                    logger.disabled = False
                    return True
        return False


class BNSPriorDict(CBCPriorDict):
    def __init__(
        self,
        dictionary=None,
        filename=None,
        aligned_spin=True,
        conversion_function=None,
    ):
        """Initialises a Prior set for Binary Neutron Stars

        Parameters
        ==========
        dictionary: dict, optional
            See superclass
        filename: str, optional
            See superclass
        conversion_function: func
            Function to convert between sampled parameters and constraints.
            By default this generates many additional parameters, see
            BNSPriorDict.default_conversion_function
        """
        if aligned_spin:
            default_file = "aligned_spins_bns_tides_on.prior"
        else:
            default_file = "precessing_spins_bns_tides_on.prior"
        if dictionary is None and filename is None:
            filename = os.path.join(DEFAULT_PRIOR_DIR, default_file)
            logger.info(
                "No prior given, using default BNS priors in {}.".format(filename)
            )
        elif filename is not None:
            if not os.path.isfile(filename):
                filename = os.path.join(DEFAULT_PRIOR_DIR, filename)
        super(BNSPriorDict, self).__init__(
            dictionary=dictionary,
            filename=filename,
            conversion_function=conversion_function,
        )

    def default_conversion_function(self, sample):
        """
        Default parameter conversion function for BNS signals.

        This generates:

        * the parameters passed to source.lal_binary_neutron_star
        * all mass parameters
        * all tidal parameters

        It does not generate:

        * component spins
        * source-frame parameters

        Parameters
        ==========
        sample: dict
            Dictionary to convert

        Returns
        =======
        sample: dict
            Same as input
        """
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_neutron_star_parameters(out_sample)
        out_sample = generate_mass_parameters(out_sample)
        out_sample = generate_tidal_parameters(out_sample)
        return out_sample

    def test_redundancy(self, key, disable_logging=False):
        logger.disabled = disable_logging
        logger.info(
            "Performing redundancy check using BBHPriorDict(self).test_redundancy"
        )
        bbh_redundancy = BBHPriorDict(self).test_redundancy(key)
        logger.disabled = False

        if bbh_redundancy:
            return True
        redundant = False

        sampling_parameters = {
            key
            for key in self
            if not isinstance(self[key], (DeltaFunction, Constraint))
        }

        tidal_parameters = {
            "lambda_1",
            "lambda_2",
            "lambda_tilde",
            "delta_lambda_tilde",
        }

        if key in tidal_parameters:
            if len(tidal_parameters.intersection(sampling_parameters)) > 2:
                redundant = True
                logger.disabled = disable_logging
                logger.warning(
                    "{} already in prior. "
                    "This may lead to unexpected behaviour.".format(
                        tidal_parameters.intersection(self)
                    )
                )
                logger.disabled = False
            elif len(tidal_parameters.intersection(sampling_parameters)) == 2:
                redundant = True
        return redundant

    @property
    def tidal(self):
        """Return true if priors include phase parameters"""
        return self.is_nonempty_intersection("tidal")


Prior._default_latex_labels = {
    "mass_1": "$m_1$",
    "mass_2": "$m_2$",
    "total_mass": "$M$",
    "chirp_mass": r"$\mathcal{M}$",
    "mass_ratio": "$q$",
    "symmetric_mass_ratio": r"$\eta$",
    "a_1": "$a_1$",
    "a_2": "$a_2$",
    "tilt_1": r"$\theta_1$",
    "tilt_2": r"$\theta_2$",
    "cos_tilt_1": r"$\cos\theta_1$",
    "cos_tilt_2": r"$\cos\theta_2$",
    "phi_12": r"$\Delta\phi$",
    "phi_jl": r"$\phi_{JL}$",
    "luminosity_distance": "$d_L$",
    "dec": r"$\mathrm{DEC}$",
    "ra": r"$\mathrm{RA}$",
    "iota": r"$\iota$",
    "cos_iota": r"$\cos\iota$",
    "theta_jn": r"$\theta_{JN}$",
    "cos_theta_jn": r"$\cos\theta_{JN}$",
    "psi": r"$\psi$",
    "phase": r"$\phi$",
    "geocent_time": "$t_c$",
    "time_jitter": "$t_j$",
    "lambda_1": r"$\Lambda_1$",
    "lambda_2": r"$\Lambda_2$",
    "lambda_tilde": r"$\tilde{\Lambda}$",
    "delta_lambda_tilde": r"$\delta\tilde{\Lambda}$",
    "chi_1": r"$\chi_1$",
    "chi_2": r"$\chi_2$",
    "chi_1_in_plane": r"$\chi_{1, \perp}$",
    "chi_2_in_plane": r"$\chi_{2, \perp}$",
}


class CalibrationPriorDict(PriorDict):
    """
    Prior dictionary class for calibration parameters.
    This has methods for simplifying the creation of priors for the large
    numbers of parameters used with the spline model.
    """

    def __init__(self, dictionary=None, filename=None):
        """
        Initialises a Prior dictionary for calibration parameters

        Parameters
        ==========
        dictionary: dict, optional
            See superclass
        filename: str, optional
            See superclass
        """
        if dictionary is None and filename is not None:
            filename = os.path.join(DEFAULT_PRIOR_DIR, filename)
        super(CalibrationPriorDict, self).__init__(
            dictionary=dictionary, filename=filename
        )
        self.source = None

    def to_file(self, outdir, label):
        """
        Write the prior to file. This includes information about the source if
        possible.

        Parameters
        ==========
        outdir: str
            Output directory.
        label: str
            Label for prior.
        """
        PriorDict.to_file(self, outdir=outdir, label=label)
        if self.source is not None:
            prior_file = os.path.join(outdir, "{}.prior".format(label))
            with open(prior_file, "a") as outfile:
                outfile.write("# prior source file is {}".format(self.source))

    @staticmethod
    def from_envelope_file(
        envelope_file,
        minimum_frequency,
        maximum_frequency,
        n_nodes,
        label,
        boundary="reflective",
        correction_type=None,
    ):
        """
        Load in the calibration envelope.

        This is a text file with columns

        ::

            frequency median-amplitude median-phase -1-sigma-amplitude
            -1-sigma-phase +1-sigma-amplitude +1-sigma-phase

        There are two definitions of the calibration correction in the
        literature, one defines the correction as mapping calibrated strain
        to theoretical waveform templates (:code:`data`) and the other as
        mapping theoretical waveform templates to calibrated strain
        (:code:`template`). Prior to version 1.4.0, :code:`template` was assumed,
        the default changed to :code:`data` when the :code:`correction` argument
        was added.

        Parameters
        ==========
        envelope_file: str
            Name of file to read in.
        minimum_frequency: float
            Minimum frequency for the spline.
        maximum_frequency: float
            Minimum frequency for the spline.
        n_nodes: int
            Number of nodes for the spline.
        label: str
            Label for the names of the parameters, e.g., `recalib_H1_`
        boundary: None, 'reflective', 'periodic'
            The type of prior boundary to assign
        correction_type: str
            How the correction is defined, either to the :code:`data`
            (default) or the :code:`template`. In general, data products
            produced by the LVK calibration groups assume :code:`data`.
            The default value will be removed in a future release and
            this will need to be explicitly specified.

            .. versionadded:: 1.4.0

        Returns
        =======
        prior: PriorDict
            Priors for the relevant parameters.
            This includes the frequencies of the nodes which are _not_ sampled.
        """
        from .detector.calibration import _check_calibration_correction_type

        correction_type = _check_calibration_correction_type(
            correction_type=correction_type
        )

        calibration_data = np.genfromtxt(envelope_file).T
        log_frequency_array = np.log(calibration_data[0])

        if correction_type.lower() == "data":
            amplitude_median = 1 / calibration_data[1] - 1
            phase_median = -calibration_data[2]
            amplitude_sigma = abs(1 / calibration_data[3] - 1 / calibration_data[5]) / 2
            phase_sigma = abs(calibration_data[6] - calibration_data[4]) / 2
        else:
            amplitude_median = calibration_data[1] - 1
            phase_median = calibration_data[2]
            amplitude_sigma = abs(calibration_data[5] - calibration_data[3]) / 2
            phase_sigma = abs(calibration_data[6] - calibration_data[4]) / 2

        log_nodes = np.linspace(
            np.log(minimum_frequency), np.log(maximum_frequency), n_nodes
        )

        amplitude_mean_nodes = InterpolatedUnivariateSpline(
            log_frequency_array, amplitude_median
        )(log_nodes)
        amplitude_sigma_nodes = InterpolatedUnivariateSpline(
            log_frequency_array, amplitude_sigma
        )(log_nodes)
        phase_mean_nodes = InterpolatedUnivariateSpline(
            log_frequency_array, phase_median
        )(log_nodes)
        phase_sigma_nodes = InterpolatedUnivariateSpline(
            log_frequency_array, phase_sigma
        )(log_nodes)

        prior = CalibrationPriorDict()
        for ii in range(n_nodes):
            name = "recalib_{}_amplitude_{}".format(label, ii)
            latex_label = "$A^{}_{}$".format(label, ii)
            prior[name] = Gaussian(
                mu=amplitude_mean_nodes[ii],
                sigma=amplitude_sigma_nodes[ii],
                name=name,
                latex_label=latex_label,
                boundary=boundary,
            )
        for ii in range(n_nodes):
            name = "recalib_{}_phase_{}".format(label, ii)
            latex_label = r"$\phi^{}_{}$".format(label, ii)
            prior[name] = Gaussian(
                mu=phase_mean_nodes[ii],
                sigma=phase_sigma_nodes[ii],
                name=name,
                latex_label=latex_label,
                boundary=boundary,
            )
        for ii in range(n_nodes):
            name = "recalib_{}_frequency_{}".format(label, ii)
            latex_label = "$f^{}_{}$".format(label, ii)
            prior[name] = DeltaFunction(
                peak=np.exp(log_nodes[ii]), name=name, latex_label=latex_label
            )
        prior.source = os.path.abspath(envelope_file)
        return prior

    @staticmethod
    def constant_uncertainty_spline(
        amplitude_sigma,
        phase_sigma,
        minimum_frequency,
        maximum_frequency,
        n_nodes,
        label,
        boundary="reflective",
    ):
        """
        Make prior assuming constant in frequency calibration uncertainty.

        This assumes Gaussian fluctuations about 0.

        Parameters
        ==========
        amplitude_sigma: float
            Uncertainty in the amplitude.
        phase_sigma: float
            Uncertainty in the phase.
        minimum_frequency: float
            Minimum frequency for the spline.
        maximum_frequency: float
            Minimum frequency for the spline.
        n_nodes: int
            Number of nodes for the spline.
        label: str
            Label for the names of the parameters, e.g., `recalib_H1_`
        boundary: None, 'reflective', 'periodic'
            The type of prior boundary to assign

        Returns
        =======
        prior: PriorDict
            Priors for the relevant parameters.
            This includes the frequencies of the nodes which are _not_ sampled.
        """
        nodes = np.logspace(
            np.log10(minimum_frequency), np.log10(maximum_frequency), n_nodes
        )

        amplitude_mean_nodes = [0] * n_nodes
        amplitude_sigma_nodes = [amplitude_sigma] * n_nodes
        phase_mean_nodes = [0] * n_nodes
        phase_sigma_nodes = [phase_sigma] * n_nodes

        prior = CalibrationPriorDict()
        for ii in range(n_nodes):
            name = "recalib_{}_amplitude_{}".format(label, ii)
            latex_label = "$A^{}_{}$".format(label, ii)
            prior[name] = Gaussian(
                mu=amplitude_mean_nodes[ii],
                sigma=amplitude_sigma_nodes[ii],
                name=name,
                latex_label=latex_label,
                boundary=boundary,
            )
        for ii in range(n_nodes):
            name = "recalib_{}_phase_{}".format(label, ii)
            latex_label = r"$\phi^{}_{}$".format(label, ii)
            prior[name] = Gaussian(
                mu=phase_mean_nodes[ii],
                sigma=phase_sigma_nodes[ii],
                name=name,
                latex_label=latex_label,
                boundary=boundary,
            )
        for ii in range(n_nodes):
            name = "recalib_{}_frequency_{}".format(label, ii)
            latex_label = "$f^{}_{}$".format(label, ii)
            prior[name] = DeltaFunction(
                peak=nodes[ii], name=name, latex_label=latex_label
            )

        return prior


def secondary_mass_condition_function(reference_params, mass_1):
    """
    Condition function to use for a prior that is conditional on the value
    of the primary mass, for example, a prior on the secondary mass that is
    bounded by the primary mass.

    .. code-block:: python

        import bilby
        priors = bilby.gw.prior.CBCPriorDict()
        priors["mass_1"] = bilby.core.prior.Uniform(5, 50)
        priors["mass_2"] = bilby.core.prior.ConditionalUniform(
            condition_func=bilby.gw.prior.secondary_mass_condition_function,
            minimum=5,
            maximum=50,
        )

    Parameters
    ----------
    reference_params: dict
        Reference parameter dictionary, this should contain a "minimum"
        attribute.
    mass_1: float
        The primary mass to use as the upper limit for the prior.

    Returns
    -------
    dict:
        Updated prior limits given the provided primary mass.
    """
    return dict(minimum=reference_params["minimum"], maximum=mass_1)


ConditionalCosmological = conditional_prior_factory(Cosmological)
ConditionalUniformComovingVolume = conditional_prior_factory(UniformComovingVolume)
ConditionalUniformSourceFrame = conditional_prior_factory(UniformSourceFrame)


class HealPixMapPriorDist(BaseJointPriorDist):
    """
    Class defining prior according to given HealPix Map, defaults to 2D in ra and dec but can be set to include
    Distance as well. This only works with skymaps that include the 2D joint probability in ra/dec and that use the
    normal LALInference type skymaps where each pixel has a DISTMU, DISTSIGMA, and DISTNORM defining the conditional
    distance distribution along a given line of sight.

    Parameters
    ==========

    hp_file : file path to .fits file
        .fits file that contains the 2D or 3D Healpix Map
    names : list (optional)
        list of names of parameters included in the JointPriorDist, defaults to ['ra', 'dec']
    bounds : dict or list (optional)
        dictionary or list with given prior bounds. defaults to normal bounds on ra, dev and 0, inf for distance
        if this is for a 3D map

    Returns
    =======

    PriorDist : `bilby.gw.prior.HealPixMapPriorDist`
        A JointPriorDist object to store the joint prior distribution according to passed healpix map
    """

    def __init__(self, hp_file, names=None, bounds=None, distance=False):
        self.hp = self._check_imports()
        self.hp_file = hp_file
        if names is None:
            names = ["ra", "dec"]
        if bounds is None:
            bounds = [[0, 2 * np.pi], [-np.pi / 2.0, np.pi / 2.0]]
        elif isinstance(bounds, dict):
            bs = [[] for _ in bounds.keys()]
            for i, key in enumerate(bounds.keys()):
                bs[i] = (bounds[key][0], bounds[key][1])
            bounds = bs
        if distance:
            if len(names) == 2:
                names.append("distance")
            if len(bounds) == 2:
                bounds.append([0, np.inf])
            self.distance = True
            self.prob, self.distmu, self.distsigma, self.distnorm = self.hp.read_map(
                hp_file, field=range(4)
            )
        else:
            self.distance = False
            self.prob = self.hp.read_map(hp_file)
        self.prob = self._check_norm(self.prob)

        super(HealPixMapPriorDist, self).__init__(names=names, bounds=bounds)
        self.distname = "hpmap"
        self.npix = len(self.prob)
        self.nside = self.hp.npix2nside(self.npix)
        self.pixel_area = self.hp.nside2pixarea(self.nside)
        self.pixel_length = self.pixel_area ** (1 / 2.0)
        self.pix_xx = np.arange(self.npix)
        self._all_interped = interp1d(
            x=self.pix_xx, y=self.prob, bounds_error=False, fill_value=0
        )
        self.inverse_cdf = None
        self.distance_pdf = None
        self.distance_icdf = None
        self._build_attributes()
        name = self.names[-1]
        if self.bounds[name][1] != np.inf and self.bounds[name][0] != -np.inf:
            self.rs = np.linspace(self.bounds[name][0], self.bounds[name][1], 1000)
        else:
            self.rs = np.linspace(0, 5000, 1000)

    def _build_attributes(self):
        """
        Method that builds the inverse cdf of the P(pixel) distribution for rescaling
        """
        yy = self._all_interped(self.pix_xx)
        yy /= trapezoid(yy, self.pix_xx)
        YY = cumulative_trapezoid(yy, self.pix_xx, initial=0)
        YY[-1] = 1
        self.inverse_cdf = interp1d(x=YY, y=self.pix_xx, bounds_error=True)

    @staticmethod
    def _check_imports():
        """
        Static method to check that healpy is installed on the machine running bibly
        """
        try:
            import healpy
        except Exception:
            raise ImportError(
                "Must have healpy installed on this machine to use HealPixMapPrior"
            )
        return healpy

    @xp_wrap
    def _rescale(self, samp, *, xp=None, **kwargs):
        """
        Overwrites the _rescale method of BaseJoint Prior to rescale a single value from the unitcube onto
        two values (ra, dec) or 3 (ra, dec, dist) if distance is included

        Parameters
        ==========
        samp : float, int
            must take in single value for pixel on unitcube to recale onto ra, dec (distance), for the map Prior
        kwargs : dict
            kwargs are all passed to _rescale() method

        Returns
        =======
        rescaled_sample : array_like
            sample to rescale onto the prior
        """
        if self.distance:
            dist_samp = samp[:, -1]
            samp = samp[:, 0]
        else:
            samp = samp[:, 0]
        pix_rescale = self.inverse_cdf(samp)
        sample = xp.empty((len(pix_rescale), 2))
        dist_samples = xp.empty((len(pix_rescale)))
        for i, val in enumerate(pix_rescale):
            theta, ra = self.hp.pix2ang(self.nside, int(round(val)))
            dec = 0.5 * np.pi - theta
            sample = xpx.at(sample, i).set(
                xp.asarray(self.draw_from_pixel(ra, dec, int(round(val))))
            )
            if self.distance:
                self.update_distance(int(round(val)))
                dist_samples = xpx.at(dist_samples, i).set(
                    xp.asarray(self.distance_icdf(dist_samp[i]))
                )
        if self.distance:
            sample = xp.vstack([sample[:, 0], sample[:, 1], dist_samples])
        return sample.reshape((-1, self.num_vars))

    def update_distance(self, pix_idx):
        """
        Method to update the conditional distance distributions at given pixel used for distance handling in the
        JointPrior Parameters. This function updates the current distance pdf, inverse_cdf, and sampler according to
        given pixel or line of sight.

        Parameters
        ==========
        pix_idx : int
            pixel index value to create the distribution for

        Returns
        =======
        None : None
            just updates these functions at new pixel values
        """
        self.distance_pdf = lambda r: self.distnorm[pix_idx] * norm(
            loc=self.distmu[pix_idx], scale=self.distsigma[pix_idx]
        ).pdf(r)
        pdfs = self.rs**2 * self.distance_pdf(self.rs)
        cdfs = np.cumsum(pdfs) / np.sum(pdfs)
        self.distance_icdf = interp1d(cdfs, self.rs)

    @staticmethod
    def _check_norm(array):
        """
        static method to check if array is properly normalized and if not to normalize it.

        Parameters
        ==========
        array : array_like
            input array we want to renormalize if not already normalized

        Returns
        =======
        normed_array : array_like
            returns input array normalized
        """
        norm = np.linalg.norm(array, ord=1)
        if norm == 0:
            norm = np.finfo(array.dtype).eps
        return array / norm

    def _sample(self, size, *, random_state=None, **kwargs):
        """
        Overwrites the _sample method of BaseJoint Prior. Picks a pixel value according to their probabilities, then
        uniformly samples ra, and decs that are contained in chosen pixel. If the PriorDist includes distance it then
        updates the distance distributions and will sample according to the conditional distance distribution along a
        given line of sight

        Parameters
        ==========
        size : int
            number of samples we want to draw
        kwargs : dict
            kwargs are all passed to be used

        Returns
        =======
        sample : array_like
            sample of ra, and dec (and distance if 3D=True)
        """
        rng = random.resolve_random_state(random_state)
        xp = random.random_array_module(rng)

        sample_pix = rng.choice(self.npix, size=size, p=self.prob, replace=True)
        sample = xp.empty((size, self.num_vars))
        for samp in range(size):
            theta, ra = self.hp.pix2ang(self.nside, sample_pix[samp])
            dec = 0.5 * np.pi - theta
            if self.distance:
                self.update_distance(sample_pix[samp])
                dist = self.draw_distance(sample_pix[samp], random_state=rng)
                ra, dec = self.draw_from_pixel(
                    ra, dec, sample_pix[samp], random_state=rng
                )
                new = [ra, dec, dist]
            else:
                new = self.draw_from_pixel(ra, dec, sample_pix[samp])
            sample = xpx.at(sample, samp).set(xp.asarray(new))
        return xp.asarray(sample.reshape((-1, self.num_vars)))

    def draw_distance(self, pix, *, random_state=None):
        """
        Method to recursively draw a distance value from the given set distance distribution and check that it is in
        the bounds

        Parameters
        ==========

        pix : int
            integer for pixel to draw a distance from

        Returns
        =======
        dist : float
            sample drawn from the distance distribution at set pixel index
        """
        rng = random.resolve_random_state(random_state)

        if self.distmu[pix] == np.inf or self.distmu[pix] <= 0:
            return 0
        dist = self.distance_icdf(rng.uniform(0, 1))
        name = self.names[-1]
        if (dist > self.bounds[name][1]) | (dist < self.bounds[name][0]):
            self.draw_distance(pix, random_state=rng)
        else:
            return dist

    def draw_from_pixel(self, ra, dec, pix, *, random_state=None):
        """
        Recursive function to uniformly draw ra, and dec values that are located in the given pixel

        Parameters
        ==========
        ra : float, int
            value drawn for rightascension
        dec : float, int
            value drawn for declination
        pix : int
            pixel index for given pixel we want to get ra, and dec from

        Returns
        =======
        ra_dec : tuple
            this returns a tuple of ra, and dec sampled uniformly that are in the pixel given
        """
        rng = random.resolve_random_state(random_state)

        if not self.check_in_pixel(ra, dec, pix):
            self.draw_from_pixel(ra, dec, pix, random_state=rng)
        return np.array(
            [
                rng.uniform(ra - self.pixel_length, ra + self.pixel_length),
                rng.uniform(dec - self.pixel_length, dec + self.pixel_length),
            ]
        )

    def check_in_pixel(self, ra, dec, pix):
        """
        Method that checks if given rightacension and declination values are within the given pixel index and the bounds

        Parameters
        ==========
        ra : float, int
            rightascension value to check
        dec : float, int
            declination value to check
        pix : int
            index for pixel we want to check in

        Returns
        =======
        bool :
            returns True if values inside pixel, False if not
        """
        for val, name in zip([ra, dec], self.names):
            if (val < self.bounds[name][0]) or (val > self.bounds[name][1]):
                return False
        phi, theta = ra, 0.5 * np.pi - dec
        pixel = self.hp.ang2pix(self.nside, theta, phi)
        return pix == pixel

    @xp_wrap
    def _ln_prob(self, samp, lnprob, outbounds, *, xp=None):
        """
        Overwrites the _lnprob method of BaseJoint Prior

        Parameters
        ==========
        samp : array_like
            samples of ra, dec to evaluate the lnprob at
        lnprob : array_like
            array of correct length we want to populate with lnprob values
        outbounds : boolean array
            boolean array that flags samples that are out of the given bounds

        Returns
        =======
        lnprob : array_like
            lnprob values at each sample
        """
        for i in range(samp.shape[0]):
            if not outbounds[i]:
                if self.distance:
                    phi, dec, dist = samp[0]
                else:
                    phi, dec = samp[0]
                theta = 0.5 * np.pi - dec
                pixel = self.hp.ang2pix(self.nside, theta, phi)
                lnprob = xpx.at(lnprob, i).set(
                    xp.log(xp.asarray(self.prob[pixel] / self.pixel_area))
                )
                if self.distance:
                    self.update_distance(pixel)
                    lnprob = xpx.at(lnprob, i).set(
                        lnprob[i]
                        + xp.log(xp.asarray(self.distance_pdf(dist) * dist**2))
                    )
        lnprob = xp.where(xp.asarray(outbounds), -np.inf, lnprob)
        return lnprob

    def __eq__(self, other):
        skip_keys = ["_all_interped", "inverse_cdf", "distance_pdf", "distance_icdf"]
        if self.__class__ != other.__class__:
            return False
        if sorted(self.__dict__.keys()) != sorted(other.__dict__.keys()):
            return False
        for key in self.__dict__:
            if key in skip_keys:
                continue
            if key == "hp_file":
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            elif isinstance(self.__dict__[key], (np.ndarray, list)):
                thisarr = np.asarray(self.__dict__[key])
                otherarr = np.asarray(other.__dict__[key])
                if thisarr.dtype == float and otherarr.dtype == float:
                    fin1 = np.isfinite(np.asarray(self.__dict__[key]))
                    fin2 = np.isfinite(np.asarray(other.__dict__[key]))
                    if not np.array_equal(fin1, fin2):
                        return False
                    if not np.allclose(thisarr[fin1], otherarr[fin2], atol=1e-15):
                        return False
                else:
                    if not np.array_equal(thisarr, otherarr):
                        return False
            else:
                if not self.__dict__[key] == other.__dict__[key]:
                    return False
        return True


class HealPixPrior(JointPrior):
    """
    A prior distribution that follows a user-provided HealPix map for one
    parameter.

    See :code:`bilby.gw.prior.HealPixMapPriorDist` for more details of how to
    instantiate the prior.
    """

    def __init__(self, dist, name=None, latex_label=None, unit=None):
        """

        Parameters
        ----------
        dist: bilby.gw.prior.HealPixMapPriorDist
            The base joint probability.
        name: str
            The name of the parameter, it should be contained in the map.
            One of ["ra", "dec", "luminosity_distance"].
        latex_label: str
            Latex label used for plotting, will be read from default values if
            not provided.
        unit: str
            The unit of the parameter.
        """
        if not isinstance(dist, HealPixMapPriorDist):
            raise JointPriorDistError(
                "dist object must be instance of HealPixMapPriorDist"
            )
        super(HealPixPrior, self).__init__(
            dist=dist, name=name, latex_label=latex_label, unit=unit
        )
