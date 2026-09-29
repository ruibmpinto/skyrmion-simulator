"""Macrospin driver for the stochastic LLGS validation gates.

Builds a `(n_traj, 1)` SAF-like parameter namespace (the
bottom layer is present but decoupled: `H_RKKY = 0`,
`K_bot = 0`, `C_ex = 0`, `C_dmi = 0`) and runs `n_traj`
independent macrospin trajectories simultaneously by stacking
them along the lattice y-axis. With `C_ex = C_dmi = 0` and
`H_RKKY = 0`, the trajectories are uncoupled, so a single Heun
step advances all `n_traj` macrospins at numpy throughput.

Used by `test_langevin.py` (Langevin function in a Zeeman
field) and `test_brown_reversal.py` (mean first-passage time
across a uniaxial barrier).

Functions
---------
make_macrospin_params
    Construct a fully-validated `(n_traj, 1)` parameters
    namespace with exchange and DMI set to zero.
run_macrospin_ensemble
    Advance `n_traj` independent stochastic trajectories and
    return the top-layer history.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
# Third-party
import numpy as np
# Local
from src.simulator.parameters import default_params
from src.stochastic_llgs.integrator_sllg import heun_stochastic_step
from src.stochastic_llgs.parameters_thermal import attach_thermal
from src.stochastic_llgs.thermal_field import sample_thermal_field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def make_macrospin_params(T, alpha, H_ext, K, a, t_Co, Ms,
                          gamma, seed, n_traj):
    """Build a `(n_traj, 1)` parameters namespace.

    Parameters
    ----------
    T : float
        Temperature in Kelvin. Strictly positive.
    alpha : float
        Gilbert damping. Strictly positive.
    H_ext : numpy.ndarray(1d)
        External field in Tesla, shape (3,).
    K : float
        Uniaxial anisotropy in J/m^3 on the top layer. Use
        `K = 0` for the Langevin test, `K > 0` for the Brown
        reversal test. Non-negative.
    a : float
        Lattice constant in meters. Strictly positive.
    t_Co : float
        Layer thickness in meters. Strictly positive.
    Ms : float
        Saturation magnetization in A/m. Strictly positive.
    gamma : float
        Gyromagnetic ratio in rad/(s*T). Strictly positive.
    seed : int
        Integer master seed for noise sampling.
    n_traj : int
        Number of independent trajectories to advance in
        parallel (stacked along the y-axis). Strictly
        positive.

    Returns
    -------
    p : SimpleNamespace
        Parameters namespace with `nx=1`, `ny=n_traj`,
        `H_RKKY = 0`, `K_bot = 0`, `J_current = 0`,
        `C_ex = 0`, `C_dmi = 0`, anisotropy prefactor
        `C_anis_top = 2 K / Ms` (bare, no thin-film demag
        correction), and `attach_thermal` already applied so
        `p.sigma_noise` is populated.

    Notes
    -----
    With `C_ex = 0` and `C_dmi = 0`, neighbor lookups along
    the y-axis become irrelevant: each row of the lattice
    evolves independently as a macrospin. With `H_RKKY = 0`
    the top and bottom SAF layers are uncoupled; only the
    top layer is consumed downstream.
    """
    if not (np.isfinite(alpha) and alpha > 0.0):
        raise RuntimeError(
            f'make_macrospin_params: alpha must be finite '
            f'and strictly positive, got {alpha!r}.'
        )
    H_ext = np.asarray(H_ext, dtype=float)
    if H_ext.shape != (3,):
        raise RuntimeError(
            f'make_macrospin_params: H_ext must have shape '
            f'(3,), got {H_ext.shape}.'
        )
    if not (np.isfinite(K) and K >= 0.0):
        raise RuntimeError(
            f'make_macrospin_params: K must be finite and '
            f'non-negative, got {K!r}.'
        )
    for name, val in (('a', a), ('t_Co', t_Co), ('Ms', Ms),
                      ('gamma', gamma)):
        if not (np.isfinite(val) and val > 0.0):
            raise RuntimeError(
                f'make_macrospin_params: {name} must be '
                f'finite and strictly positive, got {val!r}.'
            )
    if not isinstance(n_traj, (int, np.integer)) \
            or n_traj <= 0:
        raise RuntimeError(
            f'make_macrospin_params: n_traj must be a '
            f'positive int, got {n_traj!r}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Start from defaults and override what we need
    p = copy.deepcopy(default_params())
    p.nx = 1
    p.ny = int(n_traj)
    p.alpha = float(alpha)
    p.gamma = float(gamma)
    p.Ms = float(Ms)
    p.a = float(a)
    p.t_Co = float(t_Co)
    p.H_ext = H_ext
    p.K_top = float(K)
    p.K_bot = 0.0
    p.H_RKKY = 0.0
    p.J_current = 0.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Zero spatial couplings so y-axis stacks independent
    # trajectories. NOTE: anisotropy prefactor uses BARE
    # 2K/Ms (no -mu0*Ms thin-film correction) -- macrospin
    # has no film geometry.
    p.C_ex = 0.0
    p.C_dmi = 0.0
    p.C_anis_top = 2.0 * p.K_top / p.Ms
    p.C_anis_bot = 0.0
    p.H_DL = 0.0
    p.H_FL = 0.0
    p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Attach thermal-noise fields (R_th=0: no Joule heating
    # in macrospin tests)
    attach_thermal(p, T=T, R_th=0.0, seed=seed)
    return p


# -----------------------------------------------------------------------------
def run_macrospin_ensemble(p, m0_top, dt, n_steps,
                           sample_every, tol_norm):
    """Run `p.ny` independent macrospin trajectories in
    parallel.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace from `make_macrospin_params`
        (must have `nx = 1` and `C_ex = 0`, `C_dmi = 0` to
        ensure y-axis trajectories are uncoupled).
    m0_top : numpy.ndarray(1d)
        Initial top-layer magnetization, shape (3,),
        broadcast across all trajectories.
    dt : float
        Time step in seconds.
    n_steps : int
        Total number of stochastic Heun steps.
    sample_every : int
        Subsample the trajectory: record m_top once every
        `sample_every` steps. Strictly positive.
    tol_norm : float
        Norm-drift tolerance forwarded to the stepper.

    Returns
    -------
    times : numpy.ndarray(1d)
        Time at each sample, in seconds, shape (n_samples,).
    m_top_history : numpy.ndarray(3d)
        Top-layer magnetization at each sample, shape
        (n_samples, n_traj, 3).
    """
    if p.C_ex != 0.0 or p.C_dmi != 0.0:
        raise RuntimeError(
            f'run_macrospin_ensemble: requires C_ex = 0 and '
            f'C_dmi = 0 so trajectories are uncoupled; got '
            f'C_ex = {p.C_ex!r}, C_dmi = {p.C_dmi!r}.'
        )
    if p.H_RKKY != 0.0:
        raise RuntimeError(
            f'run_macrospin_ensemble: requires H_RKKY = 0 '
            f'so the two SAF layers are uncoupled; got '
            f'H_RKKY = {p.H_RKKY!r}.'
        )
    if p.nx != 1:
        raise RuntimeError(
            f'run_macrospin_ensemble: requires p.nx = 1, '
            f'got {p.nx!r}.'
        )
    if not isinstance(n_steps, (int, np.integer)) \
            or n_steps <= 0:
        raise RuntimeError(
            f'run_macrospin_ensemble: n_steps must be a '
            f'positive int, got {n_steps!r}.'
        )
    if not isinstance(sample_every, (int, np.integer)) \
            or sample_every <= 0:
        raise RuntimeError(
            f'run_macrospin_ensemble: sample_every must be '
            f'a positive int, got {sample_every!r}.'
        )
    m0_top = np.asarray(m0_top, dtype=float)
    if m0_top.shape != (3,):
        raise RuntimeError(
            f'run_macrospin_ensemble: m0_top must have '
            f'shape (3,), got {m0_top.shape}.'
        )
    n0 = float(np.linalg.norm(m0_top))
    if not np.isclose(n0, 1.0, atol=1e-10):
        raise RuntimeError(
            f'run_macrospin_ensemble: m0_top must have unit '
            f'norm, got ||m0_top|| = {n0!r}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initialize spins; broadcast m0_top across all trajectories
    n_traj = int(p.ny)
    m_top = np.empty((n_traj, 1, 3), dtype=float)
    m_top[..., :] = m0_top[np.newaxis, np.newaxis, :]
    m_bot = np.empty((n_traj, 1, 3), dtype=float)
    m_bot[..., :] = np.array([0.0, 0.0, 1.0])[
        np.newaxis, np.newaxis, :,
    ]
    rng = np.random.default_rng(int(p.seed))
    sigma = float(p.sigma_noise)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_samples = (n_steps + sample_every - 1) // sample_every
    times = np.empty(n_samples, dtype=float)
    m_top_history = np.empty((n_samples, n_traj, 3), dtype=float)
    sample_idx = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for step in range(n_steps):
        h_top = sample_thermal_field(
            rng, (n_traj, 1), sigma, dt,
        )
        h_bot = sample_thermal_field(
            rng, (n_traj, 1), sigma, dt,
        )
        m_top, m_bot, _ = heun_stochastic_step(
            m_top, m_bot, dt, p, None,
            h_top, h_bot, tol_norm, t=step * dt,
        )
        if step % sample_every == 0:
            times[sample_idx] = step * dt
            m_top_history[sample_idx, :, :] = m_top[:, 0, :]
            sample_idx += 1
    return times[:sample_idx], m_top_history[:sample_idx, :, :]
