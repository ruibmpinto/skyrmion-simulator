"""Brown reversal-time macrospin validation gate.

Uniaxial macrospin with anisotropy K (no field), `alpha`
moderate. Each trajectory starts aligned with `+z` and runs
until either `m_z` flips sign (detected as `m_z` falling below
`mz_flip_threshold`) or `t_max` is reached. The ensemble mean
first-passage time is compared to Brown's high-barrier formula

    tau = (1 + alpha**2) / (alpha * gamma)
          * sqrt(pi / Delta) * exp(Delta),
    Delta = K * V_cell / (k_B * T).

The temperatures are chosen to span barriers `Delta in {3, 5,
8}`; higher barriers become uncomputably slow.

Pass criterion: |log10(tau_sim / tau_brown)| < tol_log10 at
each barrier.

Run with:
    python -m src.stochastic_llgs.validation.test_brown_reversal
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import os
import time
# Third-party
import numpy as np
# Local
from src.stochastic_llgs.parameters_thermal import attach_thermal
from src.stochastic_llgs.thermal_field import sample_thermal_field
from src.stochastic_llgs.integrator_sllg import heun_stochastic_step
from src.stochastic_llgs.validation.macrospin import (
    make_macrospin_params,
)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def brown_tau(delta, alpha, gamma):
    """High-barrier Brown reversal time (zero field).

    Parameters
    ----------
    delta : float
        Barrier height in units of k_B T,
        Delta = K * V_cell / (k_B T).
    alpha : float
        Gilbert damping.
    gamma : float
        Gyromagnetic ratio in rad/(s * T).

    Returns
    -------
    tau : float
        Mean reversal time in seconds.
    """
    return (
        (1.0 + alpha * alpha) / (alpha * gamma)
        * math.sqrt(math.pi / delta)
        * math.exp(delta)
    )


# -----------------------------------------------------------------------------
def run_brown_ensemble(p, m0_top, dt, n_steps, tol_norm,
                      mz_flip_threshold):
    """Run an ensemble of uniaxial trajectories until each
    flips below `mz_flip_threshold` or `n_steps` elapses.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace from `make_macrospin_params`.
    m0_top : numpy.ndarray(1d)
        Initial top-layer magnetization, shape (3,).
    dt : float
        Time step in seconds.
    n_steps : int
        Maximum number of stochastic Heun steps.
    tol_norm : float
        Norm-drift tolerance forwarded to the stepper.
    mz_flip_threshold : float
        Trajectory is considered reversed the first step at
        which `m_z < mz_flip_threshold`. Typically a small
        negative number (e.g. -0.5) to suppress thermal
        false positives near the barrier.

    Returns
    -------
    flip_step : numpy.ndarray(1d)
        Per-trajectory first-passage step index. Trajectories
        that never reverse get `n_steps`.
    flipped : numpy.ndarray(1d, bool)
        Mask of trajectories that reversed within `n_steps`.
    """
    # Broadcast IC across trajectories; bottom layer is inert.
    n_traj = int(p.ny)
    m_top = np.empty((n_traj, 1, 3), dtype=float)
    m_top[..., :] = m0_top[np.newaxis, np.newaxis, :]
    m_bot = np.empty((n_traj, 1, 3), dtype=float)
    m_bot[..., :] = np.array([0.0, 0.0, 1.0])[
        np.newaxis, np.newaxis, :,
    ]
    rng = np.random.default_rng(int(p.seed))
    sigma = float(p.sigma_noise)
    # Default flip_step = n_steps marks never-reversed (censored).
    flip_step = np.full(n_traj, n_steps, dtype=np.int64)
    flipped = np.zeros(n_traj, dtype=bool)
    for step in range(n_steps):
        h_top = sample_thermal_field(
            rng, (n_traj, 1), sigma, dt,
        )
        h_bot = sample_thermal_field(
            rng, (n_traj, 1), sigma, dt,
        )
        m_top, m_bot, _ = heun_stochastic_step(
            m_top, m_bot, dt, p, None,
            h_top, h_bot, tol_norm,
        )
        # Record first-passage step for trajectories crossing
        # the threshold this step; stop once all have reversed.
        mz = m_top[:, 0, 2]
        new_flip = (mz < mz_flip_threshold) & (~flipped)
        if np.any(new_flip):
            flip_step[new_flip] = step
            flipped[new_flip] = True
            if flipped.all():
                break
    return flip_step, flipped


# -----------------------------------------------------------------------------
def main():
    # Run the Brown gate: at each barrier Delta set T accordingly,
    # run the ensemble to reversal, estimate tau, compare to
    # Brown's formula via |log10(tau_sim/tau_brown)|.
    # =========================== User Configuration =========================
    # Default trims Delta=8 (it costs ~100 min wall on its
    # own at dt=5e-14, vectorized 256 traj). Enable by adding
    # 8.0 to `delta_list`.
    delta_list      = [3.0, 5.0]        # barrier in kT units
    alpha           = 0.5
    K_top           = 1.294e6           # J/m^3 (Co/Pt SAF top)
    a               = 2.0e-9
    t_co            = 1.3e-9
    ms              = 1.43e6
    gamma           = 194.8e9
    dt              = 5.0e-14
    n_traj          = 256
    seed_base       = 31
    tol_norm        = 5.0e-3
    mz_flip_thresh  = -0.5
    # tau ~ exp(Delta); pick n_steps_per_tau enough trajs
    # actually finish. Use 30 tau_brown per Delta.
    n_steps_per_tau = 30
    tol_log10       = 0.5
    out_dir         = (
        'output/stochastic_llgs/validation'
    )
    out_npz         = 'brown_reversal.npz'
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    k_B = 1.380649e-23
    V_cell = a * a * t_co
    n_D = len(delta_list)
    tau_sim = np.full(n_D, np.nan)
    tau_th = np.full(n_D, np.nan)
    fraction_flipped = np.zeros(n_D)
    flip_steps_all = []
    flip_masks_all = []
    print('Brown reversal macrospin gate')
    print('-' * 56)
    t_start = time.time()
    for i, delta in enumerate(delta_list):
        # Set T so the barrier in kT units equals the target Delta;
        # size the window to a multiple of the expected tau.
        T = K_top * V_cell / (delta * k_B)
        tau_b = brown_tau(delta, alpha, gamma)
        n_steps = int(n_steps_per_tau * tau_b / dt)
        seed = int(seed_base + 1000 * i)
        p = make_macrospin_params(
            T=T, alpha=alpha,
            H_ext=np.array([0.0, 0.0, 0.0]),
            K=K_top, a=a, t_Co=t_co, Ms=ms,
            gamma=gamma, seed=seed, n_traj=n_traj,
        )
        m0 = np.array([0.0, 0.0, 1.0])
        flip_step, flipped = run_brown_ensemble(
            p, m0, dt, n_steps, tol_norm, mz_flip_thresh,
        )
        n_flipped = int(flipped.sum())
        if n_flipped < 0.5 * n_traj:
            # Censored estimator unreliable below ~50%; warn.
            print(
                f'  WARN: Delta={delta:.1f}: only '
                f'{n_flipped}/{n_traj} trajectories flipped '
                f'within {n_steps_per_tau:d} * tau_brown.',
                flush=True,
            )
        # MLE for exponential under right-censoring at t_max
        # = n_steps * dt: tau_hat = sum(t_i) / n_flipped
        t_obs = flip_step * dt
        tau_hat = float(t_obs.sum() / max(n_flipped, 1))
        tau_sim[i] = tau_hat
        tau_th[i] = tau_b
        fraction_flipped[i] = n_flipped / n_traj
        flip_steps_all.append(flip_step)
        flip_masks_all.append(flipped)
        log10_ratio = math.log10(tau_hat / tau_b)
        print(
            f'  Delta={delta:.1f}  T={T:7.2f} K  '
            f'tau_Brown={tau_b:.2e} s  '
            f'tau_sim={tau_hat:.2e} s  '
            f'log10(ratio)={log10_ratio:+.2f}  '
            f'flipped={n_flipped}/{n_traj}',
            flush=True,
        )
    log10_ratio = np.log10(tau_sim / tau_th)
    max_log10 = float(np.max(np.abs(log10_ratio)))
    out_path = os.path.join(out_dir, out_npz)
    np.savez_compressed(
        out_path,
        delta=np.array(delta_list, dtype=float),
        tau_sim=tau_sim, tau_brown=tau_th,
        log10_ratio=log10_ratio, max_log10=max_log10,
        tol_log10=tol_log10,
        fraction_flipped=fraction_flipped,
        n_traj=n_traj, dt=dt,
    )
    dt_wall = time.time() - t_start
    print('-' * 56)
    print(
        f'max |log10(tau_sim/tau_Brown)| = {max_log10:.2f}; '
        f'gate tol = {tol_log10:.2f}; wall {dt_wall:.1f} s'
    )
    print(f'Saved {out_path}')
    if max_log10 >= tol_log10:
        raise RuntimeError(
            f'Brown reversal gate FAILED: '
            f'max|log10(tau_sim/tau_Brown)| = {max_log10:.2f} '
            f'>= tol {tol_log10:.2f}.'
        )
    print('Brown reversal gate PASSED.')


# =============================================================================
if __name__ == '__main__':
    main()
