"""Single-layer skyrmion collapse trajectory (Rohart 2016).

Helper for the skyrmion Neel-Arrhenius benchmark (#8). Builds a
SINGLE ferromagnetic layer carrying an isolated Neel skyrmion,
applies a uniform out-of-plane destabilizing field, and runs a
Stratonovich-Heun stochastic trajectory until the skyrmion
collapses (topological charge falls through `q_threshold`) or
the observation window ends.

This reuses the original simulator / stochastic code with NO
duplication: parameters come from the production
`skyrmion_simulator.simulator.params_helper.make_params`, the genuine
single-layer time step is the production
`skyrmion_simulator.stochastic_llgs.integrator_sllg.heun_stochastic_step` in
its single-layer mode (`m_bot=None`, `kernels=None`), the
initial condition is `skyrmion_profile`, the order parameter is
`topological_charge`, and the noise / collapse detection are
`sample_thermal_field` / `detect_annihilation`. Validating this
exact path is the purpose of the benchmark.

This mirrors the Langevin-dynamics protocol of

    S. Rohart, J. Miltat, A. Thiaville, Phys. Rev. B 93,
    214412 (2016), "Path to collapse for an isolated Neel
    skyrmion" (arXiv:1601.02875),

whose Fig. 2(c) Arrhenius fit gives Delta_E = 26 +/- 4 meV and
tau_0 = 0.22 +/- 0.1 ns under a 250 mT destabilizing field at
T ~ 68-101 K. Rohart's model is ATOMISTIC (Co/Pt(111)
monolayer); ours is finite-difference micromagnetics. The
absolute barrier is grid-dependent and is NOT expected to match
across model classes -- the benchmark validates the Arrhenius
FORM, the sub-ns tau_0 order of magnitude, and the T-trend (see
test_skyrmion_arrhenius).

Functions
---------
run_collapse_trajectory
    Build a single-layer skyrmion, settle it under the
    destabilizing field, run to thermal collapse; return the
    first-passage time and survival flag.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import numpy as np
# Local
from skyrmion_simulator.simulator.params_helper import make_params
from skyrmion_simulator.simulator.initial_conditions import skyrmion_profile
from skyrmion_simulator.simulator.main import topological_charge
from skyrmion_simulator.simulator.pulses import ConstantPulse
from skyrmion_simulator.stochastic_llgs.diagnostics import detect_annihilation
from skyrmion_simulator.stochastic_llgs.integrator_sllg import (
    heun_stochastic_step,
)
from skyrmion_simulator.stochastic_llgs.parameters_thermal import attach_thermal
from skyrmion_simulator.stochastic_llgs.thermal_field import \
    sample_thermal_field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
_MU0 = 4.0e-7 * np.pi


def run_collapse_trajectory(config):
    """Run one single-layer skyrmion collapse trajectory.

    Seeds a Neel skyrmion (core m_z = -1), settles it under the
    destabilizing field with a deterministic relaxation (noise
    off, excluded from the collapse clock), then runs the
    stochastic observation window sampling the topological
    charge; collapse is the first-passage of |Q| through
    `q_threshold` for `k_consecutive` samples.

    Parameters
    ----------
    config : dict
        Required keys (no defaults):
            T_sub, alpha, gamma, Ms, A_ex, D, K_eff, H_ext,
            a, t_Co, nx, ny, skyrmion_R, skyrmion_dw, dt,
            n_relax, n_drive, sample_every, seed, tol_norm,
            q_threshold, k_consecutive.

    Returns
    -------
    payload : dict
        t_sample, Q, T_effective, sigma_noise, alive_at_end,
        flip_index, t_collapse (seconds; NaN if survived).
    """
    required = (
        'T_sub', 'alpha', 'gamma', 'Ms', 'A_ex', 'D', 'K_eff',
        'H_ext', 'a', 't_Co', 'nx', 'ny', 'skyrmion_R',
        'skyrmion_dw', 'dt', 'n_relax', 'n_drive',
        'sample_every', 'seed', 'tol_norm', 'q_threshold',
        'k_consecutive')
    missing = [k for k in required if k not in config]
    if missing:
        raise RuntimeError(
            f'run_collapse_trajectory: config missing keys: '
            f'{missing}.')
    T_sub = float(config['T_sub'])
    Ms = float(config['Ms'])
    K_eff = float(config['K_eff'])
    dt = float(config['dt'])
    n_relax = int(config['n_relax'])
    n_drive = int(config['n_drive'])
    sample_every = int(config['sample_every'])
    seed = int(config['seed'])
    tol_norm = float(config['tol_norm'])
    q_threshold = float(config['q_threshold'])
    k_consecutive = int(config['k_consecutive'])
    nx = int(config['nx'])
    ny = int(config['ny'])
    H_ext = np.asarray(config['H_ext'], dtype=float)
    if K_eff <= 0.0:
        raise RuntimeError(
            f'run_collapse_trajectory: K_eff must be a positive '
            f'easy-axis anisotropy, got {K_eff!r}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameters via the production make_params. make_params
    # folds the thin-film shape term into the anisotropy
    # prefactor: C_anis = 2 K_top / Ms - mu0 Ms. Rohart's
    # effective anisotropy K_eff already includes shape and
    # demag is off, so choose K_top so the NET prefactor equals
    # the bare 2 K_eff / Ms:  K_top = K_eff + 0.5 mu0 Ms^2.
    K_top = K_eff + 0.5 * _MU0 * Ms * Ms
    p = make_params(
        nx=nx, ny=ny, a=float(config['a']),
        t_Co=float(config['t_Co']), Ms=Ms,
        A_ex=float(config['A_ex']), D=float(config['D']),
        K_top=K_top, K_bot=K_top, alpha=float(config['alpha']),
        gamma=float(config['gamma']), H_ext=H_ext,
        pulse=ConstantPulse(0.0))
    # Single ferromagnetic layer: no RKKY partner.
    p.H_RKKY = 0.0
    p.dt = dt
    p.skyrmion_R = float(config['skyrmion_R'])
    p.skyrmion_dw = float(config['skyrmion_dw'])
    attach_thermal(p, T=T_sub, R_th=0.0, seed=seed)
    # IC: single Neel skyrmion, core m_z = -1 (polarity = +1).
    m = skyrmion_profile(
        nx, ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw, polarity=1)
    rng = np.random.default_rng(int(p.seed))
    sigma = float(p.sigma_noise)
    zero_h = np.zeros((ny, nx, 3), dtype=float)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Deterministic settle under the destabilizing field (noise
    # off) so the seeded analytic profile relaxes to the
    # metastable shape before the collapse clock starts. NOT
    # counted toward t_collapse.
    for _ in range(n_relax):
        m, _, _ = heun_stochastic_step(
            m, None, dt, p, None, zero_h, None, tol_norm, t=0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Stochastic observation window: sample Q every sample_every.
    # Early-stop once |Q| has stayed below q_threshold for
    # k_consecutive samples (the skyrmion has collapsed; further
    # integration of noise on the FM state is wasted) -- the
    # arrays are truncated to the samples actually taken.
    n_samples = (n_drive + sample_every - 1) // sample_every
    t_sample = np.empty(n_samples, dtype=float)
    Q_arr = np.empty(n_samples, dtype=float)
    s_idx = 0
    below_run = 0
    for step in range(1, n_drive + 1):
        h = sample_thermal_field(rng, (ny, nx), sigma, dt) \
            if sigma > 0.0 else zero_h
        m, _, _ = heun_stochastic_step(
            m, None, dt, p, None, h, None, tol_norm,
            t=step * dt)
        if step % sample_every == 0:
            t_sample[s_idx] = step * dt
            q = float(topological_charge(m, p.a))
            Q_arr[s_idx] = q
            s_idx += 1
            below_run = below_run + 1 \
                if abs(q) < q_threshold else 0
            if below_run >= k_consecutive:
                break
    t_sample = t_sample[:s_idx]
    Q_arr = Q_arr[:s_idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Collapse = first-passage of |Q| below q_threshold for
    # k_consecutive samples (seeded skyrmion has |Q| ~ 1).
    flip_index = detect_annihilation(
        np.abs(Q_arr), q_threshold=q_threshold,
        k_consecutive=k_consecutive)
    alive = bool(flip_index == -1)
    t_collapse = float('nan') if alive \
        else float(flip_index * sample_every * dt)
    payload = {
        't_sample': t_sample,
        'Q': Q_arr,
        'T_effective': float(p.T),
        'sigma_noise': sigma,
        'alive_at_end': alive,
        'flip_index': int(flip_index),
        't_collapse': t_collapse,
    }
    return payload
