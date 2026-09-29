"""Phase 3a: T = 0 reproducibility against the deterministic
RK4 stepper, no demag.

Runs a SAF skyrmion under SOT drive twice: once with the
existing `skyrmion_simulator.simulator.integrator.rk4_step` (no demag, no
noise), and once with `heun_stochastic_step` at `T = 0`
(`sigma_noise = 0`, hence the predictor-corrector degenerates
to a deterministic Heun integrator). Compares equilibrium
diameter, drift velocity, Hall angle, and topological charge.

Heun is order 2 and RK4 is order 4, so a small discretization
mismatch is expected. Pass criterion: relative deviation
< `tol_rel` on each observable. Failure here indicates a bug
in the stepper (e.g., wrong RHS, wrong renormalization, wrong
SOT inclusion) rather than a noise-amplitude problem.

Run with:
    python -m \\
        skyrmion_simulator.stochastic_llgs.validation.test_t0_limit_nodemag
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
import os
import time
# Third-party
import numpy as np
# Local
from skyrmion_simulator.simulator.analysis import (
    skyrmion_center,
    skyrmion_diameter,
)
from skyrmion_simulator.simulator.initial_conditions import saf_skyrmion
from skyrmion_simulator.simulator.integrator import rhs_local_keff, rk4_step
from skyrmion_simulator.simulator.main import topological_charge
from skyrmion_simulator.simulator.parameters import default_params
from skyrmion_simulator.simulator.pulses import ConstantPulse
from skyrmion_simulator.stochastic_llgs.integrator_sllg import \
    heun_stochastic_step
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


def run_deterministic(p, n_relax, n_drive, dt):
    """Run analysis.py-style relax + drive with rk4_step.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace.
    n_relax : int
        Number of relaxation steps (J = 0).
    n_drive : int
        Number of current-driven steps (J = p.J_current).
    dt : float
        Time step in seconds.

    Returns
    -------
    track : dict
        Final-state observables `diameter, cx, cy, Q,
        velocity, hall_deg`. Velocity is averaged over the
        last quarter of the drive phase.
    """
    m_top, m_bot = saf_skyrmion(
        p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Relax with J = 0
    # Swap pulse for the deterministic integrator that now reads
    # p.pulse(t) at each substep.
    H_DL_save = p.H_DL
    H_FL_save = p.H_FL
    pulse_save = getattr(p, 'pulse', None)
    p.H_DL = 0.0
    p.H_FL = 0.0
    p.pulse = ConstantPulse(0.0)
    t = 0.0
    for step in range(n_relax):
        m_top, m_bot = rk4_step(rhs_local_keff, m_top, m_bot, t, dt, p)
        t += dt
    p.H_DL = H_DL_save
    p.H_FL = H_FL_save
    if pulse_save is not None:
        p.pulse = pulse_save
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Drive; record center at quarter-points
    sample_steps = (
        int(0.75 * n_drive), n_drive,
    )
    track_x = []
    track_y = []
    track_t = []
    t = 0.0
    for step in range(1, n_drive + 1):
        m_top, m_bot = rk4_step(rhs_local_keff, m_top, m_bot, t, dt, p)
        t += dt
        if step in sample_steps:
            cx, cy = skyrmion_center(m_top, p.a, core_polarity=+1)
            track_x.append(cx)
            track_y.append(cy)
            track_t.append(step * dt)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    vx = (track_x[1] - track_x[0]) / (track_t[1] - track_t[0])
    vy = (track_y[1] - track_y[0]) / (track_t[1] - track_t[0])
    return {
        'diameter': float(skyrmion_diameter(m_top, p.a, core_polarity=+1)),
        'cx': float(track_x[1]),
        'cy': float(track_y[1]),
        'Q': float(topological_charge(m_top, p.a)),
        'velocity': float(np.sqrt(vx ** 2 + vy ** 2)),
        'hall_deg': float(
            np.degrees(np.arctan2(abs(vy), abs(vx)))
        ),
    }


# -----------------------------------------------------------------------------
def run_stochastic_t0(p, n_relax, n_drive, dt, tol_norm):
    """Same workflow as `run_deterministic` but with
    `heun_stochastic_step` at `sigma_noise = 0`.
    """
    m_top, m_bot = saf_skyrmion(
        p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw,
    )
    rng = np.random.default_rng(int(p.seed))
    sigma = float(p.sigma_noise)
    if sigma != 0.0:
        raise RuntimeError(
            f'run_stochastic_t0: sigma_noise must be 0 at '
            f'T = 0, got {sigma!r}.'
        )
    H_DL_save = p.H_DL
    H_FL_save = p.H_FL
    pulse_save = getattr(p, 'pulse', None)
    p.H_DL = 0.0
    p.H_FL = 0.0
    # The pulse is what llgs_rhs reads; zeroing only the H_DL/H_FL
    # diagnostics left this relax phase fully DRIVEN (the RK4 path
    # relaxes at J = 0), which broke the integrator-parity gate.
    p.pulse = ConstantPulse(0.0)
    for step in range(n_relax):
        # Zero-noise sampling: amplitude 0, but the sampler
        # rejects sigma <= 0. Allocate zero arrays directly
        # at T = 0.
        h_top = np.zeros((p.ny, p.nx, 3), dtype=float)
        h_bot = np.zeros((p.ny, p.nx, 3), dtype=float)
        m_top, m_bot, _ = heun_stochastic_step(
            m_top, m_bot, dt, p, None,
            h_top, h_bot, tol_norm, t=step * dt,
        )
    p.H_DL = H_DL_save
    p.H_FL = H_FL_save
    if pulse_save is not None:
        p.pulse = pulse_save
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    sample_steps = (
        int(0.75 * n_drive), n_drive,
    )
    track_x = []
    track_y = []
    track_t = []
    for step in range(1, n_drive + 1):
        h_top = np.zeros((p.ny, p.nx, 3), dtype=float)
        h_bot = np.zeros((p.ny, p.nx, 3), dtype=float)
        # Start-of-step time, matching the RK4 drive clock.
        m_top, m_bot, _ = heun_stochastic_step(
            m_top, m_bot, dt, p, None,
            h_top, h_bot, tol_norm, t=(step - 1) * dt,
        )
        if step in sample_steps:
            cx, cy = skyrmion_center(m_top, p.a, core_polarity=+1)
            track_x.append(cx)
            track_y.append(cy)
            track_t.append(step * dt)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    vx = (track_x[1] - track_x[0]) / (track_t[1] - track_t[0])
    vy = (track_y[1] - track_y[0]) / (track_t[1] - track_t[0])
    return {
        'diameter': float(skyrmion_diameter(m_top, p.a, core_polarity=+1)),
        'cx': float(track_x[1]),
        'cy': float(track_y[1]),
        'Q': float(topological_charge(m_top, p.a)),
        'velocity': float(np.sqrt(vx ** 2 + vy ** 2)),
        'hall_deg': float(
            np.degrees(np.arctan2(abs(vy), abs(vx)))
        ),
    }


# -----------------------------------------------------------------------------
def main():
    """Run the T=0 no-demag gate.

    Runs the same relax+drive via RK4 and via zero-noise Heun,
    then gates on the per-observable relative error.
    """
    # =========================== User Configuration =========================
    # 256x256 lattice. Use shorter durations than
    # src/skyrmion_simulator/simulator/analysis.py (500 ps relax + 1 ns drive)
    # to keep the gate wall-time manageable. Increase if the
    # comparison drifts due to incomplete relaxation.
    n_relax     = 6000        # 300 ps at dt=5e-14
    n_drive     = 10000       # 500 ps at dt=5e-14
    dt          = 5.0e-14
    seed        = 1
    tol_norm    = 5.0e-3
    tol_rel     = 0.02        # 2% on each observable
    out_dir     = 'output/stochastic_llgs/validation'
    out_npz     = 't0_limit_nodemag.npz'
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    print('Phase 3a: T = 0 reproducibility (no demag)')
    print('-' * 56)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reference: deterministic RK4 from src/skyrmion_simulator/simulator
    p_det = default_params()
    p_det.dt = dt
    print(
        f'Running deterministic RK4 reference '
        f'({n_relax} relax + {n_drive} drive steps)...'
    )
    t0 = time.time()
    ref = run_deterministic(p_det, n_relax, n_drive, dt)
    dt_ref = time.time() - t0
    print(
        f'  RK4 done in {dt_ref:.1f} s: '
        f'd={ref["diameter"]*1e9:.1f} nm, '
        f'v={ref["velocity"]:.1f} m/s, '
        f'theta={ref["hall_deg"]:.2f} deg, '
        f'Q={ref["Q"]:.4f}'
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Simulation: zero-noise Heun (T = 0)
    p_sim = copy.deepcopy(default_params())
    p_sim.dt = dt
    attach_thermal(p_sim, T=0.0, R_th=0.0, seed=seed)
    print(
        f'Running stochastic Heun at T = 0 '
        f'(sigma_noise={p_sim.sigma_noise:.2e})...'
    )
    t0 = time.time()
    sim = run_stochastic_t0(
        p_sim, n_relax, n_drive, dt, tol_norm,
    )
    dt_sim = time.time() - t0
    print(
        f'  Heun done in {dt_sim:.1f} s: '
        f'd={sim["diameter"]*1e9:.1f} nm, '
        f'v={sim["velocity"]:.1f} m/s, '
        f'theta={sim["hall_deg"]:.2f} deg, '
        f'Q={sim["Q"]:.4f}'
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Relative deviation per observable; collect those over tol.
    rel = {}
    failures = []
    for key in ('diameter', 'velocity', 'Q'):
        ref_val = ref[key]
        sim_val = sim[key]
        if ref_val == 0.0:
            rel[key] = float('inf') if sim_val != 0.0 else 0.0
        else:
            rel[key] = abs(sim_val - ref_val) / abs(ref_val)
        if rel[key] >= tol_rel:
            failures.append(
                f'{key}: ref={ref_val:.4e}, sim={sim_val:.4e}, '
                f'rel={rel[key]:.3f}'
            )
    # Hall angle: compare absolute deviation in degrees
    dtheta = abs(sim['hall_deg'] - ref['hall_deg'])
    rel['hall_deg_abs_diff'] = dtheta
    out_path = os.path.join(out_dir, out_npz)
    np.savez_compressed(
        out_path,
        ref_diameter=ref['diameter'],
        ref_velocity=ref['velocity'],
        ref_hall_deg=ref['hall_deg'],
        ref_Q=ref['Q'],
        sim_diameter=sim['diameter'],
        sim_velocity=sim['velocity'],
        sim_hall_deg=sim['hall_deg'],
        sim_Q=sim['Q'],
        rel_diameter=rel['diameter'],
        rel_velocity=rel['velocity'],
        rel_Q=rel['Q'],
        hall_abs_diff=dtheta,
        tol_rel=tol_rel,
    )
    print('-' * 56)
    for k, v in rel.items():
        print(f'  {k:24s} {v:.4f}')
    print(f'Saved {out_path}')
    if failures:
        raise RuntimeError(
            'Phase 3a (T=0 no-demag) FAILED:\n  '
            + '\n  '.join(failures)
        )
    print('Phase 3a PASSED.')


# =============================================================================
if __name__ == '__main__':
    main()
