"""Skyrmion breathing-mode test: perturb the relaxed equilibrium
and watch it ring back. Soft-mode signature near Bogdanov-Hubert
D_c is a low-frequency, slow-decay breathing oscillation.

For each D value:
  1. Relax with K_eff (alpha=1, over-damped quench) to obtain
     the equilibrium (m_top_eq, m_bot_eq).
  2. Apply a small radial perturbation: scale m_z by ~0.97
     uniformly and renormalize. This dilates the skyrmion in
     a near-pure breathing eigenmode.
  3. Integrate free LLG dynamics (J=0, paper alpha) for a few
     hundred ps, sampling d_top(t) every ~2 ps.
  4. Save the trace + analytic D_c metadata to disk.

The companion `plot_breathing.py` fits each d_top(t) to a damped
sinusoid d_eq + A*exp(-t/tau)*cos(2*pi*f*t + phi) and reports
(f, tau) vs D. Local-worker (multiprocessing.Pool) parallelism
via SWEEP_NPROC.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import multiprocessing as mp
import os
# Third-party
import numpy as np
# Local
from src.phase_diagram.relaxation import relax as relax_demag
from src.simulator.analysis import skyrmion_diameter
from src.simulator.demag import precompute_demag_kernels
from src.simulator.fields import effective_field
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.integrator import (
    normalize,
    rhs_demag,
    rhs_local_keff,
    rk4_step,
)
from src.simulator.parameters import _precompute, default_params
from src.simulator.pulses import ConstantPulse
from src.orchestrator.io import save_trace

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _relax_keff(p, m_top, m_bot, max_steps, tol_torque,
                check_every, alpha_relax):
    """Over-damped K_eff relax. Returns
    (m_top, m_bot, conv, n_steps, tau_max)."""
    _save = (p.alpha, p.gamma_p, p.H_DL, p.H_FL,
             getattr(p, 'pulse', None))
    p.alpha = float(alpha_relax)
    p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
    p.H_DL = 0.0
    p.H_FL = 0.0
    p.pulse = ConstantPulse(0.0)
    tau_max = float('inf')
    conv = False
    n_used = max_steps
    try:
        t = 0.0
        for k in range(1, max_steps + 1):
            m_top, m_bot = rk4_step(
                rhs_local_keff, m_top, m_bot, t, p.dt, p)
            t += p.dt
            if k % check_every == 0:
                H_t = effective_field(
                    m_top, m_bot, p.C_ex, p.C_dmi,
                    p.C_anis_top, p.H_ext, p.H_RKKY)
                H_b = effective_field(
                    m_bot, m_top, p.C_ex, p.C_dmi,
                    p.C_anis_bot, p.H_ext, p.H_RKKY)
                tau_t = np.cross(m_top, np.cross(m_top, H_t))
                tau_b = np.cross(m_bot, np.cross(m_bot, H_b))
                tau_max = float(max(
                    np.sqrt((tau_t * tau_t).sum(-1)).max(),
                    np.sqrt((tau_b * tau_b).sum(-1)).max()))
                if tau_max < tol_torque:
                    conv = True
                    n_used = k
                    break
    finally:
        (p.alpha, p.gamma_p, p.H_DL, p.H_FL, p.pulse) = _save
    return m_top, m_bot, conv, n_used, tau_max


def _radial_perturb(m, eps):
    """Multiply m_z by (1 - eps) and renormalize. For a SAF
    skyrmion this expands the m_z=0 ring slightly: a near-pure
    radial breathing perturbation."""
    m2 = m.copy()
    m2[..., 2] *= (1.0 - eps)
    return normalize(m2)


def _run_one_D(args):
    """Relax + perturb + free-evolve at one D; saves d_top(t) and
    returns a one-line summary. Picklable for multiprocessing."""
    cfg, D = args
    field_kind = cfg['field_kind']
    nx = cfg['nx']
    ny = cfg['ny']
    dt = cfg['dt']
    p = default_params()
    p.D = D
    p.nx = nx
    p.ny = ny
    p.dt = dt
    _precompute(p)
    print(f'D = {D*1e3:.3f} mJ/m^2 ({field_kind}): start',
          flush=True)
    m_top, m_bot = saf_skyrmion(
        nx, ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Relax to equilibrium. K_eff path uses the local inline
    # relax (no FFT kernel); newell/slab path uses the
    # demag-aware relax with a precomputed kernel.
    if field_kind == 'keff':
        kernels = None
        m_top_eq, m_bot_eq, conv, n_relax, tau = _relax_keff(
            p, m_top.copy(), m_bot.copy(),
            max_steps=cfg['relax_max_steps'],
            tol_torque=cfg['relax_tol_torque'],
            check_every=cfg['relax_check_every'],
            alpha_relax=cfg['relax_alpha'])
        # K_eff relax has no energy-trend check; emit nan for
        # schema uniformity with the demag path.
        E_final = float('nan')
    else:
        kernels = precompute_demag_kernels(
            p, kind=field_kind,
            accuracy=cfg['demag_newell_accuracy'],
            tol_conv=cfg['demag_newell_tol_conv'])
        (m_top_eq, m_bot_eq,
         conv, n_relax, E_final, tau) = relax_demag(
            m_top=m_top.copy(), m_bot=m_bot.copy(),
            p=p, kernels=kernels,
            max_steps=cfg['relax_max_steps'],
            alpha_relax=cfg['relax_alpha'],
            tol_torque=cfg['relax_tol_torque'],
            tol_dE=cfg['relax_tol_dE'],
            check_every=cfg['relax_check_every'],
            print_every=0)
    d_eq = float(skyrmion_diameter(
        m_top_eq, p.a, core_polarity=+1))
    print(f'  D={D*1e3:.3f} ({field_kind}): relaxed '
          f'(conv={conv}, n={n_relax}, tau={tau:.2e}), '
          f'd_eq={d_eq*1e9:.1f} nm', flush=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Perturb and free-evolve. Same RHS choice as the relax;
    # only alpha and the SOT are temporarily overridden.
    eps = float(cfg['perturb_eps'])
    m_top_p = _radial_perturb(m_top_eq, eps)
    m_bot_p = _radial_perturb(m_bot_eq, eps)
    _save = (p.alpha, p.gamma_p, p.H_DL, p.H_FL,
             getattr(p, 'pulse', None))
    p.alpha = float(cfg['free_alpha'])
    p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
    p.H_DL = 0.0
    p.H_FL = 0.0
    p.pulse = ConstantPulse(0.0)
    # Bind the right RHS for free evolution.
    if field_kind == 'keff':
        rhs_pair = rhs_local_keff
    else:
        rhs_pair = rhs_demag(kernels)
    n_free = int(cfg['n_free'])
    sample_every = int(cfg['sample_every'])
    times = []
    d_top_arr = []
    d_bot_arr = []
    try:
        t = 0.0
        for k in range(n_free + 1):
            if k % sample_every == 0:
                d_top_arr.append(
                    float(skyrmion_diameter(
                        m_top_p, p.a, core_polarity=+1)))
                d_bot_arr.append(
                    float(skyrmion_diameter(
                        m_bot_p, p.a, core_polarity=-1)))
                times.append(t)
            if k == n_free:
                break
            m_top_p, m_bot_p = rk4_step(
                rhs_pair, m_top_p, m_bot_p, t, p.dt, p)
            t += p.dt
    finally:
        (p.alpha, p.gamma_p, p.H_DL, p.H_FL, p.pulse) = _save
    trace = {
        't': np.asarray(times, dtype=float),
        'd_top': np.asarray(d_top_arr, dtype=float),
        'd_bot': np.asarray(d_bot_arr, dtype=float),
    }
    metadata = {
        'figure': 'S41_breathing',
        'field_kind': str(field_kind),
        'demag': ('local_keff' if field_kind == 'keff'
                  else field_kind),
        'D': float(D), 'd_eq': float(d_eq),
        'perturb_eps': float(eps),
        'free_alpha': float(cfg['free_alpha']),
        'relax_alpha': float(cfg['relax_alpha']),
        'relax_converged': bool(conv),
        'relax_tau_max_final': float(tau),
        'relax_E_final': float(E_final),
        'n_relax': int(n_relax),
        'n_free': int(n_free),
        'sample_every': int(sample_every),
        'nx': int(nx), 'ny': int(ny), 'dt': float(dt),
    }
    # Filename: breathing_{field_kind}_{T}ns_{D}mJm2.npz, with
    # 'p' as the decimal separator in D so the result is safe for
    # filesystems and globs (e.g. 0.620 -> 0p620).
    _T_ns_total = cfg['n_free'] * dt * 1.0e9
    if abs(_T_ns_total - round(_T_ns_total)) < 1e-6:
        _T_tag = f'{int(round(_T_ns_total))}ns'
    else:
        _T_tag = f'{_T_ns_total:.1f}'.replace('.', 'p') + 'ns'
    _D_mJ = D * 1.0e3
    _D_tag = f'{_D_mJ:.3f}'.replace('.', 'p') + 'mJm2'
    out_path = os.path.join(
        cfg['out_dir'],
        f'breathing_{field_kind}_{_T_tag}_{_D_tag}.npz')
    save_trace(path=out_path, trace=trace, metadata=metadata)
    return (
        f'  D = {D*1e3:.3f} mJ/m^2 ({field_kind}): '
        f'd_eq={d_eq*1e9:.1f} nm, '
        f'd(t=0)={d_top_arr[0]*1e9:.1f} nm, '
        f'd(t=end)={d_top_arr[-1]*1e9:.1f} nm '
        f'-> {out_path}')


def main():
    """Sweep the DMI constant `D`, perturbing each relaxed
    skyrmion and integrating free LLG to record the breathing
    trace `d_top(t)` to `output/sweeps_S41_S49/S41_breathing`.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    D_values = [
        0.62e-3, 0.72e-3, 0.80e-3,
        0.85e-3, 0.90e-3, 0.95e-3, 1.00e-3,
    ]
    # Field model. 'keff' uses the local K_eff path (no FFT
    # convolution; slab demag folded into anisotropy).
    # 'newell' or 'slab' use bare K plus an explicit FFT demag
    # kernel; expect a ~30s precompute per task and ~5-10x
    # larger per-step cost.
    field_kind = 'keff'
    demag_newell_accuracy = 4.0
    demag_newell_tol_conv = 0.02
    # Perturbation amplitude: 3% scaling of m_z (small enough to
    # remain in the linear-response regime, large enough to be
    # well above numerical noise).
    perturb_eps = 0.03
    # Free LLG dynamics: paper Set A alpha = 0.14, no SOT.
    free_alpha = 0.14
    # Free-evolution window: 1 ns at dt = 5e-14 => 20_000 steps.
    free_time = 1.0e-9
    nx = 256
    ny = 256
    dt = 5.0e-14
    n_free = int(math.ceil(free_time / dt))
    # Sample d(t) every ~2 ps to resolve sub-30-GHz breathing.
    sample_dt = 2.0e-12
    sample_every = int(math.ceil(sample_dt / dt))
    # Relax to equilibrium (alpha=1 over-damped).
    relax_max_steps = 200_000
    relax_alpha = 1.0
    relax_tol_torque = 1.0e-5
    relax_tol_dE = 1.0e-8
    relax_check_every = 1_000
    out_dir = 'output/sweeps_S41_S49/S41_breathing'
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg = {
        'nx': int(nx), 'ny': int(ny), 'dt': float(dt),
        'field_kind': str(field_kind),
        'demag_newell_accuracy': float(demag_newell_accuracy),
        'demag_newell_tol_conv': float(demag_newell_tol_conv),
        'perturb_eps': float(perturb_eps),
        'free_alpha': float(free_alpha),
        'n_free': int(n_free),
        'sample_every': int(sample_every),
        'relax_max_steps': int(relax_max_steps),
        'relax_alpha': float(relax_alpha),
        'relax_tol_torque': float(relax_tol_torque),
        'relax_tol_dE': float(relax_tol_dE),
        'relax_check_every': int(relax_check_every),
        'out_dir': out_dir,
    }
    # SLURM array dispatch: each task does one D. Falls back to
    # the full sweep locally when SLURM_ARRAY_TASK_ID is unset.
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(D_values):
            raise RuntimeError(
                f'sweep_breathing: SLURM_ARRAY_TASK_ID={task_id} '
                f'out of range [0, {len(D_values) - 1}].')
        D_values = [D_values[idx]]
    args_list = [(cfg, D) for D in D_values]
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    print(f'sweep_breathing: field_kind={field_kind}, '
          f'{len(args_list)} D values, n_proc={n_proc}, '
          f'perturb_eps={perturb_eps}, free_alpha={free_alpha}, '
          f'free_time={free_time*1e9:.1f} ns',
          flush=True)
    if n_proc <= 1:
        for args in args_list:
            print(_run_one_D(args), flush=True)
    else:
        with mp.Pool(n_proc) as pool:
            for r in pool.imap_unordered(_run_one_D, args_list):
                print(r, flush=True)


# =============================================================================
if __name__ == '__main__':
    main()
