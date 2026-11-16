#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a precessing binary
neutron star by sampling directly in efficient intrinsic parameters,

    (mu1, mass_ratio, chi_eff, chi_2, chi_1_in_plane, chi_2_in_plane, phi_jl, phi_12),

instead of the usual intrinsic parameters,

    (chirp_mass, mass_ratio, a_1, a_2, tilt_1, tilt_2, phi_jl, phi_12).

mu1 is a linear combination of the 0PN, 1PN and 1.5PN phase coefficients and is
nearly uncorrelated with the other parameters. Its prior (Mu1Prior) corresponds
to a prior on the chirp mass proportional to the chirp mass. The spin priors
correspond to isotropic spins with magnitudes uniform in [0, a_max]:
chi_eff given mass_ratio, chi_2 given (chi_eff, mass_ratio) and the in-plane
spins given the aligned spins (chi_1 = (1 + q) chi_eff - q chi_2).

Only these 8 intrinsic parameters are sampled; the extrinsic parameters are
fixed to their injected values.
"""

import os

import bilby
import numpy as np
from bilby.core.prior import ConditionalPriorDict, DeltaFunction, Uniform
from bilby.core.utils.random import seed
from bilby.gw.conversion import (
    chi_1_from_chi_eff_and_chi_2,
    convert_to_lal_binary_neutron_star_parameters,
    generate_all_bns_parameters,
    mu1_approx,
    mu1_to_chirp_mass,
)
from bilby.gw.prior import (
    ConditionalChiInPlane,
    ConditionalIsotropicChi2zPrior,
    ConditionalIsotropicChieffPrior,
    Mu1Prior,
    UniformInComponentsMassRatio,
    load_or_build_tables,
)

# Sets seed of bilby's generator "rng" to "123" to ensure reproducibility
seed(123)

outdir = "outdir"
label = "mu1_isotropic_spin"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Prior ranges
minimum_chirp_mass, maximum_chirp_mass = 1.19, 1.21
minimum_mass_ratio, maximum_mass_ratio = 0.5, 1.0
a_max = 0.5  # maximum spin magnitude

# mu1 is linearized in the chirp mass around Mcref, and its PN coefficients are
# evaluated at fref (the rotation to mu1 was derived for fref = 200 Hz).
# fref is independent of the reference frequency of the waveform.
Mcref = 0.5 * (minimum_chirp_mass + maximum_chirp_mass)
fref = 200.0

duration = 256
sampling_frequency = 4096
minimum_frequency = 20
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=fref,
)


# The waveform generator passes only the parameters to the conversion function,
# so Mcref and fref are set in a module-level function (unlike functools.partial,
# it can be stored in the result file).  The conversion computes chi_1 from
# (chi_eff, chi_2) and the chirp mass from mu1.
def convert_to_lal_parameters(parameters):
    return convert_to_lal_binary_neutron_star_parameters(
        parameters, Mcref=Mcref, fref=fref
    )


# mu1 is linear in the chirp mass and the aligned spins, so its extremes over the
# prior lie at the edges of the chirp-mass and aligned-spin ranges.
mass_ratio_grid = np.linspace(minimum_mass_ratio, maximum_mass_ratio, 1000)
spin_corners = [-a_max, a_max]
grid = np.meshgrid(
    [minimum_chirp_mass, maximum_chirp_mass],
    mass_ratio_grid,
    spin_corners,
    spin_corners,
    indexing="ij",
)
mu1_values = mu1_approx(*grid, Mcref=Mcref, fref=fref)
minimum_mu1, maximum_mu1 = float(mu1_values.min()), float(mu1_values.max())

# The rectangular mu1 prior allows slightly smaller chirp masses than
# minimum_chirp_mass; the smallest one is the reference chirp mass of the
# multi-banded likelihood.
grid = np.meshgrid(
    [minimum_mu1, maximum_mu1],
    mass_ratio_grid,
    spin_corners,
    spin_corners,
    indexing="ij",
)
reference_chirp_mass = float(mu1_to_chirp_mass(*grid, Mcref=Mcref, fref=fref).min())

# Injection: a precessing binary neutron star with spins given by their aligned
# (chi_i) and in-plane (chi_i_in_plane) components.
injection_parameters = dict(
    chirp_mass=1.2,
    mass_ratio=0.8,
    chi_1=0.05,
    chi_2=-0.03,
    chi_1_in_plane=0.1,
    chi_2_in_plane=0.05,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=150.0,
    theta_jn=0.4,
    psi=0.659,
    phase=1.3,
    geocent_time=1187008882.4,
    ra=3.44616,
    dec=-0.408084,
)
injection_parameters["chi_eff"] = (
    injection_parameters["chi_1"]
    + injection_parameters["mass_ratio"] * injection_parameters["chi_2"]
) / (1 + injection_parameters["mass_ratio"])
injection_parameters["mu1"] = float(
    mu1_approx(
        injection_parameters["chirp_mass"],
        injection_parameters["mass_ratio"],
        injection_parameters["chi_1"],
        injection_parameters["chi_2"],
        Mcref=Mcref,
        fref=fref,
    )
)

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])
for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - duration + 2,
)
injection_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_neutron_star,
    parameter_conversion=convert_to_lal_binary_neutron_star_parameters,
    waveform_arguments=dict(waveform_arguments),
)
ifos.inject_signal(
    waveform_generator=injection_waveform_generator, parameters=injection_parameters
)

# Inverse-CDF tables of the chi_eff and chi_2 priors.  They are built and saved
# on the first run (taking a few seconds) and loaded afterwards.
chi_eff_table, chi_2_table = load_or_build_tables(
    minimum_mass_ratio,
    maximum_mass_ratio,
    a_max,
    os.path.join(outdir, "chi_eff_table.npz"),
    os.path.join(outdir, "chi_2_table.npz"),
)

priors = ConditionalPriorDict()
priors["mu1"] = Mu1Prior(
    minimum=minimum_mu1,
    maximum=maximum_mu1,
    Mcref=Mcref,
    minimum_chirp_mass=minimum_chirp_mass,
    maximum_chirp_mass=maximum_chirp_mass,
    fref=fref,
)
priors["mass_ratio"] = UniformInComponentsMassRatio(
    minimum=minimum_mass_ratio, maximum=maximum_mass_ratio, name="mass_ratio"
)
priors["chi_eff"] = ConditionalIsotropicChieffPrior(table=chi_eff_table)
priors["chi_2"] = ConditionalIsotropicChi2zPrior(table=chi_2_table)
# chi_1 is not sampled, so the prior of chi_1_in_plane computes it from
# (mass_ratio, chi_eff, chi_2).
priors["chi_1_in_plane"] = ConditionalChiInPlane(
    minimum=0,
    maximum=a_max,
    name="chi_1_in_plane",
    latex_label=r"$\chi_{1\perp}$",
    aligned_spin_func=chi_1_from_chi_eff_and_chi_2,
)
priors["chi_2_in_plane"] = ConditionalChiInPlane(
    minimum=0,
    maximum=a_max,
    name="chi_2_in_plane",
    latex_label=r"$\chi_{2\perp}$",
)
priors["phi_12"] = Uniform(
    0, 2 * np.pi, name="phi_12", latex_label=r"$\phi_{12}$", boundary="periodic"
)
priors["phi_jl"] = Uniform(
    0, 2 * np.pi, name="phi_jl", latex_label=r"$\phi_{JL}$", boundary="periodic"
)
# The extrinsic parameters are fixed to their injected values.
for key in [
    "luminosity_distance",
    "theta_jn",
    "psi",
    "phase",
    "ra",
    "dec",
    "geocent_time",
]:
    priors[key] = DeltaFunction(injection_parameters[key], name=key)

# Multi-banded likelihood
search_waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.binary_neutron_star_frequency_sequence,
    parameter_conversion=convert_to_lal_parameters,
    waveform_arguments=dict(waveform_arguments),
)
likelihood = bilby.gw.likelihood.MBGravitationalWaveTransient(
    interferometers=ifos,
    waveform_generator=search_waveform_generator,
    priors=priors,
    reference_chirp_mass=reference_chirp_mass,
)

# Run the sampler.  We set `npool=16` to parallelize the analysis over 16 cores.
# generate_all_bns_parameters takes Mcref and fref from priors["mu1"] and adds
# chirp_mass, chi_1, a_1, tilt_1, ... to the posterior.
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=1000,
    sample="acceptance-walk",
    naccept=60,
    maxmcmc=50000,
    dlogz=0.1,
    npool=16,
    outdir=outdir,
    label=label,
    injection_parameters=injection_parameters,
    conversion_function=generate_all_bns_parameters,
    result_class=bilby.gw.result.CBCResult,
    check_point_delta_t=3600,
    exit_code=77,
)

result.plot_corner(
    parameters=[
        "mu1",
        "mass_ratio",
        "chi_eff",
        "chi_2",
        "chi_1_in_plane",
        "chi_2_in_plane",
    ]
)
