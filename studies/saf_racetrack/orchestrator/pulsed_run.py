"""Shared machinery for the finite-temperature pulsed-drive sweeps.

The per-sweep scripts
(`studies/saf_racetrack/scripts/sweep_pulse_shape_finiteT.py` and the
`studies/saf_racetrack/scripts/sweep_S4*_finiteT.py` family) each own their
grid and their configuration; everything they have in common lives here so
they do not carry four copies of it:

- `load_seed` reads a campaign seed state (the T=0 relaxed `m_eq` or a
  cached thermal `m_thermal`) and strips the leading frame axis.
- `make_pulse` builds one of the six compared shapes from a name, with
  the same conventions as the C++ driver `sweep_pulse_shape.cpp`.
- `run_pulsed_point` integrates one grid point and writes its trace.

This is the reference and pilot path. Production ensembles run through
the C++ driver: a 1 ns drive on 350x500 costs minutes there and hours
here, so Python is used to cross-check the C++ result and to scout
small grids, never to carry a full ensemble. The per-configuration
field-dump streams that the GIFs are built from also come from the C++
driver, which has the SnapshotBuffer; `run_one` here records at most a
single snapshot, enough for a spot check and not for an animation.

Functions
---------
load_seed
    Read an `m_eq` / `m_thermal` NPZ into a (m_top, m_bot) pair.
make_pulse
    Build a named pulse shape.
shape_names
    The six shape names, in the order the study reports them.
run_pulsed_point
    Integrate and save one (shape, peak J, T) trajectory.

Notes
-----
Every argument is explicit: there are no defaults that would let a
caller omit the box, the material, the seed or the pulse and still get
a run. A missing seed file raises rather than falling back to a fresh
initial condition, because a silently re-seeded run would look
plausible and be wrong.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import numpy as np
# Local
from studies.saf_racetrack.orchestrator.driver import run_one
from studies.saf_racetrack.orchestrator.integrators import (
    step_demag_deterministic, step_stochastic,
)
from studies.saf_racetrack.orchestrator.io import save_trace
from skyrmion_simulator.simulator.demag import precompute_demag_kernels
from skyrmion_simulator.simulator.parameters import _precompute, default_params
from skyrmion_simulator.simulator.pulses import (
    GaussianPulse, HalfSinePulse, SquarePulse, TrianglePulse,
)
from skyrmion_simulator.stochastic_llgs.joule_heating import T_of_j
from skyrmion_simulator.stochastic_llgs.parameters_thermal import attach_thermal

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def shape_names():
    """Names of the compared pulse shapes, in reporting order.

    Returns
    -------
    names : tuple[str]
        Shape identifiers accepted by `make_pulse`.
    """
    return ('square', 'halfsine', 'tri_sharprise', 'tri_sharpfall',
            'tri_symmetric', 'gaussian')


# -----------------------------------------------------------------------------
def make_pulse(shape, peak_j, t_pulse, gauss_fwhm):
    """Build one drive profile, matching the C++ driver's conventions.

    The pulse starts at t = 0 because the drive phase does, and every
    shape carries the same peak amplitude, so shapes are compared at
    matched peak and duration. Delivered charge and action then differ
    between shapes by construction and are recomputed from the profile
    during analysis.

    Parameters
    ----------
    shape : str
        One of `shape_names()`.
    peak_j : float
        Peak current density in A/m^2.
    t_pulse : float
        Pulse duration in seconds.
    gauss_fwhm : float
        Gaussian FWHM in seconds, independent of `t_pulse` so an FWHM
        sweep can vary it on its own. Used only by the Gaussian shape,
        but required always so a caller cannot select the Gaussian and
        leave its width implicit.

    Returns
    -------
    pulse : callable
        Pulse mapping time (s) to current density (A/m^2).
    """
    if t_pulse <= 0.0:
        raise RuntimeError(
            f'make_pulse: t_pulse must be positive, got {t_pulse}.')
    if shape == 'square':
        return SquarePulse(peak_j, 0.0, t_pulse)
    if shape == 'halfsine':
        return HalfSinePulse(peak_j, 0.0, t_pulse)
    if shape == 'tri_sharprise':
        # Peak at the leading edge: instantaneous rise, linear fall.
        return TrianglePulse(peak_j, 0.0, 0.0, t_pulse)
    if shape == 'tri_sharpfall':
        # Peak at the trailing edge: linear rise, instantaneous fall.
        return TrianglePulse(peak_j, 0.0, t_pulse, t_pulse)
    if shape == 'tri_symmetric':
        return TrianglePulse(peak_j, 0.0, 0.5*t_pulse, t_pulse)
    if shape == 'gaussian':
        # Centred in the window. Unlike the others the Gaussian has no
        # finite support, so a little current leaks past t_pulse; the
        # analysis integrates the profile over the whole simulated
        # window rather than assuming it stops there.
        if gauss_fwhm <= 0.0:
            raise RuntimeError(
                f'make_pulse: gauss_fwhm must be positive, got '
                f'{gauss_fwhm}.')
        return GaussianPulse(peak_j, 0.5*t_pulse, gauss_fwhm)
    raise RuntimeError(
        f'make_pulse: unknown shape {shape!r}; expected one of '
        f'{shape_names()}.')


# -----------------------------------------------------------------------------
def load_seed(path):
    """Read a campaign seed state into a (m_top, m_bot) pair.

    The campaign writes both layers as (1, ny, nx, 3); the leading
    frame axis is stripped here.

    Parameters
    ----------
    path : str
        Path to an `m_eq_*.npz` or `m_thermal_*.npz` file.

    Returns
    -------
    m_top, m_bot : numpy.ndarray(3d)
        Layer configurations, shape (ny, nx, 3).
    """
    if not os.path.isfile(path):
        raise RuntimeError(
            f'load_seed: {path!r} not found. Finite-T points need the '
            f'campaign equilibrate stage; T = 0 needs its relax '
            f'stage. Refusing to substitute a fresh initial '
            f'condition.')
    with np.load(path, allow_pickle=False) as d:
        for key in ('m_top', 'm_bot'):
            if key not in d.files:
                raise RuntimeError(
                    f'load_seed: {path!r} lacks {key!r}; '
                    f'keys={list(d.files)}.')
        m_top = np.asarray(d['m_top'], dtype=float)
        m_bot = np.asarray(d['m_bot'], dtype=float)
    if m_top.ndim != 4 or m_bot.ndim != 4:
        raise RuntimeError(
            f'load_seed: expected 4-D (1, ny, nx, 3) layers in '
            f'{path!r}, got {m_top.shape} and {m_bot.shape}.')
    return m_top[0].copy(), m_bot[0].copy()


# -----------------------------------------------------------------------------
def run_pulsed_point(cfg, point):
    """Integrate and save one (shape, peak J, T_sub) trajectory.

    Mirrors `sweep_pulse_shape.cpp` in seeds, pulse conventions and
    grid identity. The integrator differs at T = 0 only: this path uses
    the deterministic RK4 stepper there, because the stochastic closure
    requires a positive noise amplitude, while the C++ driver runs Heun
    at sigma = 0. Finite-T points use Heun in both, but their noise
    streams differ (PCG64 vs the C++ generator), so cross-language
    comparison is meaningful at T = 0 and statistical above it.

    Parameters
    ----------
    cfg : dict
        Shared configuration. Required keys: nx, ny, dt, D, K_top,
        demag_kind, demag_accuracy, demag_tol_conv, n_drive,
        sample_every, tol_norm, r_th, t_pulse, gauss_fwhm,
        seed_dir, out_dir, seed_base, sweep_tag.
    point : dict
        Grid point. Required keys: shape, peak_j, T_sub, ens,
        cell_idx.

    Returns
    -------
    summary : str
        One-line console summary for the parent process to print.
    """
    required_cfg = (
        'nx', 'ny', 'dt', 'D', 'K_top', 'demag_kind',
        'demag_accuracy', 'demag_tol_conv', 'n_drive', 'sample_every',
        'tol_norm', 'r_th', 't_pulse', 'gauss_fwhm', 'seed_dir',
        'out_dir', 'seed_base', 'sweep_tag')
    missing = [k for k in required_cfg if k not in cfg]
    if missing:
        raise RuntimeError(
            f'run_pulsed_point: cfg is missing {missing}.')
    required_point = ('shape', 'peak_j', 'T_sub', 'ens', 'cell_idx')
    missing = [k for k in required_point if k not in point]
    if missing:
        raise RuntimeError(
            f'run_pulsed_point: point is missing {missing}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fresh parameter namespace per point so mutations cannot leak
    # between grid points in a worker pool.
    p = default_params()
    p.nx = int(cfg['nx'])
    p.ny = int(cfg['ny'])
    p.dt = float(cfg['dt'])
    p.D = float(cfg['D'])
    p.K_top = float(cfg['K_top'])
    _precompute(p)
    kernels = precompute_demag_kernels(
        p, kind=str(cfg['demag_kind']),
        accuracy=float(cfg['demag_accuracy']),
        tol_conv=float(cfg['demag_tol_conv']))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Seed: the T=0 relaxed state for a deterministic point, else this
    # member's cached thermal state.
    t_sub = float(point['T_sub'])
    noise_free = (t_sub == 0.0)
    if noise_free:
        seed_path = os.path.join(
            cfg['seed_dir'], f'm_eq_{p.nx}x{p.ny}.npz')
    else:
        seed_path = os.path.join(
            cfg['seed_dir'],
            f'm_thermal_T{t_sub:05.1f}_ens{int(point["ens"]):03d}.npz')
    m_top_eq, m_bot_eq = load_seed(seed_path)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Thermal amplitude. T_of_j rejects a non-positive substrate
    # temperature, so the noise-free point bypasses it; Joule heating
    # cannot be represented there and a non-zero R_th is an error
    # rather than something to drop quietly.
    r_th = float(cfg['r_th'])
    if noise_free:
        if r_th != 0.0:
            raise RuntimeError(
                'run_pulsed_point: T_sub = 0 is the noise-free mode '
                'and cannot carry Joule heating; got R_th != 0.')
        t_eff = 0.0
    else:
        t_eff = T_of_j(float(point['peak_j']), T_sub=t_sub, R_th=r_th)
    # Same seed formula as sweep_pulse_shape.cpp, including its 2e9
    # stage offset, so a point is identified identically in both
    # languages. Note the noise STREAMS still differ: numpy's PCG64 and
    # the C++ generator produce different draws from the same seed, so
    # cross-language agreement is only expected at T = 0, where the
    # amplitude is zero and no draw affects the result.
    seed = (2000000000 + int(cfg['seed_base']) + 1000*int(point['ens'])
            + 1000000*int(point['cell_idx']))
    rng = np.random.default_rng(seed)
    attach_thermal(p, T=t_eff, R_th=r_th, seed=seed)
    # Stepper chosen explicitly per mode. sample_thermal_field rejects a
    # zero amplitude, so the noise-free point cannot go through the
    # stochastic closure and uses the deterministic RK4 stepper instead.
    # Consequence, stated so it is not mistaken for a bug: the C++
    # driver runs T = 0 as Heun at sigma = 0, so the two languages agree
    # there only to integrator truncation, not bitwise.
    if noise_free:
        step = step_demag_deterministic(kernels)
    else:
        step = step_stochastic(rng, p.sigma_noise,
                               float(cfg['tol_norm']), kernels)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # A point may override the window and the Gaussian width, which is
    # what the FWHM sweeps vary; absent an override the cfg value holds.
    t_pulse = float(point.get('t_pulse', cfg['t_pulse']))
    gauss_fwhm = float(point.get('gauss_fwhm', cfg['gauss_fwhm']))
    pulse = make_pulse(str(point['shape']), float(point['peak_j']),
                       t_pulse, gauss_fwhm)
    trace = run_one(
        p=p,
        pulse=pulse,
        n_relax=0,                 # seed is already equilibrated
        n_drive=int(cfg['n_drive']),
        sample_every=int(cfg['sample_every']),
        step_drive=step,
        step_relax=step,
        ic_factory=(lambda _p, _eq=(m_top_eq, m_bot_eq):
                    (_eq[0].copy(), _eq[1].copy())),
        record_snapshot_at=None,
        print_every=10000)
    metadata = {
        'sweep': str(cfg['sweep_tag']),
        'shape': str(point['shape']),
        'peak_j': float(point['peak_j']),
        'T_sub': t_sub,
        'T_effective': float(t_eff),
        'ens_idx': int(point['ens']),
        'nx': int(p.nx), 'ny': int(p.ny), 'dt': float(p.dt),
        'D': float(p.D), 'K_top': float(p.K_top),
        't_pulse': t_pulse,
        'gauss_fwhm': gauss_fwhm,
        'n_drive': int(cfg['n_drive']),
        'sample_every': int(cfg['sample_every']),
        'demag_kind': str(cfg['demag_kind']),
        'noise_free': bool(noise_free),
        'sigma_noise': float(p.sigma_noise),
        'seed': int(p.seed),
        'seed_path': seed_path,
    }
    os.makedirs(cfg['out_dir'], exist_ok=True)
    out_name = (f'{point["shape"]}_T{t_sub:05.1f}_'
                f'j{float(point["peak_j"]):.2e}_'
                f'ens{int(point["ens"]):03d}.npz')
    save_trace(os.path.join(cfg['out_dir'], out_name), trace, metadata)
    dx = float(trace['cx_top'][-1] - trace['cx_top'][0])
    dy = float(trace['cy_top'][-1] - trace['cy_top'][0])
    return (f'  {point["shape"]:14s} T_sub={t_sub:5.1f}  '
            f'peak_j={float(point["peak_j"]):.2e}  '
            f'ens={int(point["ens"]):03d}  '
            f'dx={dx*1e9:8.3f} nm  dy={dy*1e9:7.3f} nm')
