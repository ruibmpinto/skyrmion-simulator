"""Skyrmion collapse Arrhenius DMI sweep (benchmark #8b).

Companion to `scan_skyrmion_arrhenius` that adds the DMI axis:
runs the single-layer collapse Arrhenius scan at several DMI
values D, so the activation barrier Delta_E(D) can be extracted
per D and compared to the steep barrier-vs-DMI trend of

    Rohart, Miltat, Thiaville, Phys. Rev. B 93, 214412 (2016),
    Fig. 5(b) -- Delta_E rises strongly with the DMI strength.

The destabilizing field is fixed at the same sub-critical value
as the single-D scan (95 mT); as D increases the skyrmion grows
and its collapse field rises, so a fixed field probes a larger
barrier (the Rohart Fig. 5b mechanism). Each D keeps D < D_c so
the isolated skyrmion is metastable (FM ground state).

Grid is (D, T_sub, ens_idx); each trajectory writes its own NPZ
tagged by D. Downstream `test_skyrmion_arrhenius_dtrend` fits
Delta_E per D and checks the trend. Mirrors the single-D scan's
execution model (serial / SLURM array).

Run with:
    python -m src.stochastic_llgs.validation.scan_skyrmion_arrhenius_dsweep
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import os
# Third-party
import numpy as np
# Local
from src.stochastic_llgs.validation.rohart_skyrmion import (
    run_collapse_trajectory,
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


def _run_one_point(args):
    """Run one collapse trajectory at a given (D, T) and save."""
    cfg, point = args
    D = float(point['D'])
    T_sub = float(point['T_sub'])
    ens_idx = int(point['ens_idx'])
    d_idx = int(point['d_idx'])
    cell_idx = int(point['cell_idx'])
    seed = int(cfg['seed_base']) + 1000 * ens_idx \
        + 1000_000 * cell_idx + 100_000_000 * d_idx
    # Per-D seed: Eq.(18) radius estimate, wall width = Delta.
    A_ex = float(cfg['A_ex'])
    K_eff = float(cfg['K_eff'])
    Delta = math.sqrt(A_ex / K_eff)
    Dc = (4.0 / math.pi) * math.sqrt(A_ex * K_eff)
    skyrmion_R = Delta / math.sqrt(2.0 * (1.0 - D / Dc))
    traj_config = {
        'T_sub': T_sub, 'alpha': float(cfg['alpha']),
        'gamma': float(cfg['gamma']), 'Ms': float(cfg['Ms']),
        'A_ex': A_ex, 'D': D, 'K_eff': K_eff,
        'H_ext': np.asarray(cfg['H_ext'], dtype=float),
        'a': float(cfg['a']), 't_Co': float(cfg['t_Co']),
        'nx': int(cfg['nx']), 'ny': int(cfg['ny']),
        'skyrmion_R': skyrmion_R, 'skyrmion_dw': Delta,
        'dt': float(cfg['dt']), 'n_relax': int(cfg['n_relax']),
        'n_drive': int(cfg['n_drive']),
        'sample_every': int(cfg['sample_every']),
        'seed': seed, 'tol_norm': float(cfg['tol_norm']),
        'q_threshold': float(cfg['q_threshold']),
        'k_consecutive': int(cfg['k_consecutive']),
    }
    payload = run_collapse_trajectory(traj_config)
    d_tag = f'D{int(round(D * 1e6)):04d}u'
    out_name = f'{d_tag}_T{T_sub:06.1f}_ens{ens_idx:04d}.npz'
    out_path = os.path.join(cfg['out_dir'], out_name)
    np.savez_compressed(
        out_path, D=D, T_sub=T_sub, ens_idx=ens_idx,
        t_collapse=payload['t_collapse'],
        alive_at_end=payload['alive_at_end'],
        flip_index=payload['flip_index'],
        T_effective=payload['T_effective'],
        sigma_noise=payload['sigma_noise'],
        skyrmion_R=skyrmion_R, Delta=Delta, Dc=Dc,
        H_ext=np.asarray(cfg['H_ext'], dtype=float),
        dt=float(cfg['dt']), n_drive=int(cfg['n_drive']),
        sample_every=int(cfg['sample_every']))
    tag = 'survived' if payload['alive_at_end'] \
        else f't_collapse={payload["t_collapse"]*1e9:.2f}ns'
    return (f'  D={D*1e3:.2f} mJ/m^2  T={T_sub:6.1f} K  '
            f'ens={ens_idx:04d}  {tag}')


# -----------------------------------------------------------------------------
def main():
    # =========================== User Configuration =========================
    A_ex            = 16.0e-12       # J/m
    K_eff           = 0.50e6         # J/m^3
    Ms              = 1.1e6          # A/m
    alpha           = 0.3
    gamma           = 1.760e11       # rad/(s T)
    a               = 2.0e-9         # m
    t_Co            = 0.6e-9         # m
    nx              = 128
    ny              = 128
    # Fixed sub-critical destabilizing field (same as the
    # single-D scan); the barrier grows with D at fixed field.
    H_ext           = [0.0, 0.0, 0.095]   # Tesla
    # DMI sweep, narrow grid inside the tau-observable window at
    # the fixed 95 mT field (a wider grid leaves low-D athermal
    # and high-D never-collapsing within the 60 ns window, since
    # H_c(D) rises with D). D/D_c = 0.80, 0.82, 0.83, 0.85, 0.86.
    d_list          = [2.90e-3, 2.95e-3, 3.00e-3,
                       3.05e-3, 3.10e-3]
    t_sub_list      = [35.0, 40.0, 45.0, 50.0, 55.0,
                       60.0, 65.0, 70.0, 75.0, 80.0]
    n_ens           = 64
    dt              = 2.5e-14        # s
    n_relax         = 8000           # 200 ps settle (no clock)
    n_drive         = 2_400_000      # 60 ns observation window
    sample_every    = 400            # 10 ps cadence
    seed_base       = 7331
    tol_norm        = 5.0e-3
    q_threshold     = 0.5
    k_consecutive   = 10
    out_dir         = ('output/stochastic_llgs/validation/'
                       'skyrmion_arrhenius_dsweep')
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    Dc = (4.0 / math.pi) * math.sqrt(A_ex * K_eff)
    for D in d_list:
        if not (D < Dc):
            raise RuntimeError(
                f'scan_..._dsweep: D={D*1e3:.3f} mJ/m^2 is not '
                f'sub-critical (D_c={Dc*1e3:.3f}); the isolated '
                f'skyrmion would not be metastable.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    grid = []
    cell_idx = 0
    for d_idx, D in enumerate(d_list):
        for T_sub in t_sub_list:
            for ens_idx in range(int(n_ens)):
                grid.append({
                    'D': float(D), 'T_sub': float(T_sub),
                    'ens_idx': int(ens_idx),
                    'd_idx': int(d_idx),
                    'cell_idx': int(cell_idx)})
            cell_idx += 1
    cfg = {
        'A_ex': A_ex, 'K_eff': K_eff, 'Ms': Ms, 'alpha': alpha,
        'gamma': gamma, 'a': a, 't_Co': t_Co, 'nx': nx,
        'ny': ny, 'H_ext': H_ext, 'dt': dt, 'n_relax': n_relax,
        'n_drive': n_drive, 'sample_every': sample_every,
        'seed_base': seed_base, 'tol_norm': tol_norm,
        'q_threshold': q_threshold,
        'k_consecutive': k_consecutive, 'out_dir': out_dir,
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(grid):
            raise RuntimeError(
                f'scan_..._dsweep: SLURM_ARRAY_TASK_ID={task_id}'
                f' out of range [0, {len(grid) - 1}].')
        grid = [grid[idx]]
    print(f'scan_skyrmion_arrhenius_dsweep: {len(grid)} '
          f'trajectories ({len(d_list)} D x {len(t_sub_list)} '
          f'T x {n_ens} ens, D_c={Dc*1e3:.2f} mJ/m^2, '
          f'|H|={H_ext[2]*1e3:.0f} mT)')
    for point in grid:
        print(_run_one_point((cfg, point)), flush=True)


# =============================================================================
if __name__ == '__main__':
    main()
