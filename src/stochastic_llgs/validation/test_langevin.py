"""Langevin-function macrospin validation gate.

Macrospin in a Zeeman field B_z * z_hat, no anisotropy,
alpha = 1 (over-damped for fast thermalization). For each
temperature T, an ensemble of macrospin trajectories is run.
The time- and ensemble-averaged <m_z> is compared to the
exact Langevin function

    L(x) = coth(x) - 1/x,   x = mu * B_z / (k_B * T),
    mu = Ms * V_cell.

Pass criterion: RMS(<m_z>_sim - L(x)) / L(x) < tol_rms across
all (T, B_z) cells. Failure here usually means the noise
amplitude `sigma_noise` is wrong or the Stratonovich
predictor-corrector is not picking up the Ito-Stratonovich
drift correctly.

Run with:
    python -m src.stochastic_llgs.validation.test_langevin
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import time
# Third-party
import numpy as np
# Local
from src.stochastic_llgs.validation.macrospin import (
    make_macrospin_params,
    run_macrospin_ensemble,
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


def langevin_function(x):
    """Evaluate L(x) = coth(x) - 1/x with a safe x -> 0 limit.

    Parameters
    ----------
    x : float or numpy.ndarray
        Argument.

    Returns
    -------
    L : same type as x
        Langevin function. The series at x = 0 is L = x/3,
        used for |x| < 1e-4 to avoid catastrophic
        cancellation.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-4
    out[small] = x[small] / 3.0
    big = ~small
    out[big] = 1.0 / np.tanh(x[big]) - 1.0 / x[big]
    return out


# -----------------------------------------------------------------------------
def main():
    # =========================== User Configuration =========================
    t_kelvin_list   = [50.0, 100.0, 200.0, 300.0, 500.0]
    b_z_list        = [0.25, 0.5]                # Tesla
    alpha           = 1.0
    a               = 2.0e-9
    t_co            = 1.3e-9
    ms              = 1.43e6
    gamma           = 194.8e9
    # dt = 1e-14 chosen so that (gamma' * sigma)**2 * dt stays
    # below tol_norm = 5e-3 at T = 500 K with alpha = 1.0.
    # Coarser dt fires the heun-stepper norm-drift guard.
    dt              = 1.0e-14
    n_steps         = 200000         # 200000 * 1e-14 = 2 ns
    n_relax_steps   = 20000          # discard 200 ps
    sample_every    = 100
    n_traj          = 256
    seed_base       = 17
    tol_norm        = 5.0e-3
    tol_rms         = 0.05           # 5% RMS gate
    out_dir         = (
        'output/stochastic_llgs/validation'
    )
    out_npz         = 'langevin.npz'
    # ======================= End User Configuration =========================
    if n_relax_steps >= n_steps:
        raise RuntimeError(
            f'n_relax_steps ({n_relax_steps}) must be less '
            f'than n_steps ({n_steps}).'
        )
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_T = len(t_kelvin_list)
    n_B = len(b_z_list)
    mz_sim = np.full((n_T, n_B), np.nan)
    mz_se = np.full((n_T, n_B), np.nan)
    mz_lan = np.full((n_T, n_B), np.nan)
    x_vals = np.full((n_T, n_B), np.nan)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Langevin macrospin gate')
    print('-' * 56)
    print(
        f'n_traj={n_traj}, n_steps={n_steps}, '
        f'dt={dt:.2e}, sample_every={sample_every}'
    )
    t_start = time.time()
    for i, T in enumerate(t_kelvin_list):
        for j, B in enumerate(b_z_list):
            seed = int(seed_base + 1000 * i + j)
            p = make_macrospin_params(
                T=T, alpha=alpha,
                H_ext=np.array([0.0, 0.0, B]),
                K=0.0, a=a, t_Co=t_co, Ms=ms,
                gamma=gamma, seed=seed, n_traj=n_traj,
            )
            mu_per_cell = p.Ms * p.V_cell
            x = mu_per_cell * B / (p.k_B * T)
            L = float(langevin_function(np.array([x]))[0])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Start aligned with B to avoid waiting for
            # thermalization across the barrier (there is no
            # barrier here; this just sets a meaningful sign
            # for the Langevin comparison).
            m0 = np.array([0.0, 0.0, 1.0])
            times, m_hist = run_macrospin_ensemble(
                p, m0, dt, n_steps, sample_every, tol_norm,
            )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Drop relaxation portion
            relax_samples = (
                n_relax_steps + sample_every - 1
            ) // sample_every
            m_post = m_hist[relax_samples:, :, 2]
            mz_per_traj = m_post.mean(axis=0)
            mz_mean = float(mz_per_traj.mean())
            mz_se_ij = float(
                mz_per_traj.std(ddof=1)
                / np.sqrt(mz_per_traj.size)
            )
            mz_sim[i, j] = mz_mean
            mz_se[i, j] = mz_se_ij
            mz_lan[i, j] = L
            x_vals[i, j] = x
            print(
                f'  T={T:6.1f} K  B={B:.2f} T  x={x:6.3f}  '
                f'L={L:+.4f}  sim={mz_mean:+.4f} +/- '
                f'{mz_se_ij:.4f}',
                flush=True,
            )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pass criterion
    rel_err = (mz_sim - mz_lan) / mz_lan
    rms = float(np.sqrt(np.mean(rel_err ** 2)))
    max_abs = float(np.max(np.abs(rel_err)))
    out_path = os.path.join(out_dir, out_npz)
    np.savez_compressed(
        out_path,
        t_kelvin=np.array(t_kelvin_list, dtype=float),
        b_z=np.array(b_z_list, dtype=float),
        mz_sim=mz_sim, mz_se=mz_se,
        mz_langevin=mz_lan, x_vals=x_vals,
        rel_err=rel_err, rms=rms, max_abs=max_abs,
        tol_rms=tol_rms,
        n_traj=n_traj, n_steps=n_steps, dt=dt,
        sample_every=sample_every,
        n_relax_steps=n_relax_steps,
    )
    dt_wall = time.time() - t_start
    print('-' * 56)
    print(
        f'RMS relative error = {rms:.4f}, '
        f'max|rel_err| = {max_abs:.4f}; gate '
        f'tol_rms = {tol_rms:.4f}; wall {dt_wall:.1f} s'
    )
    print(f'Saved {out_path}')
    if rms >= tol_rms:
        raise RuntimeError(
            f'Langevin gate FAILED: RMS relative error '
            f'{rms:.4f} >= tol_rms {tol_rms:.4f}.'
        )
    print('Langevin gate PASSED.')


# =============================================================================
if __name__ == '__main__':
    main()
