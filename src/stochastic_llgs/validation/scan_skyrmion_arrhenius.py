"""Skyrmion Neel-Arrhenius collapse scan (benchmark #8).

Drives the single-layer Rohart collapse trajectory
(`rohart_skyrmion.run_collapse_trajectory`) over a grid of
temperatures, with an ensemble per temperature, under a fixed
destabilizing out-of-plane field. Each trajectory writes its
own NPZ; the Arrhenius fit (ln tau vs 1/T) and the pass/fail
gate are performed downstream by `test_skyrmion_arrhenius`.

This is the benchmark driver (NOT the production
`scan_arrhenius`); it mirrors Rohart 2016's Langevin protocol
(250 mT destabilizing field, T ~ 68-101 K) with RT 2013 /
Rohart-compatible micromagnetic parameters. See
`rohart_skyrmion` for the model-class caveat.

Execution modes
---------------
- Serial: runs the whole grid in-process.
- SLURM array: each task runs `grid[SLURM_ARRAY_TASK_ID]`
  (one trajectory), via `scripts/submit_skyrmion_arrhenius.sh`.

Run with:
    python -m src.stochastic_llgs.validation.scan_skyrmion_arrhenius
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
    """Run one collapse trajectory and save its NPZ."""
    cfg, point = args
    T_sub = float(point['T_sub'])
    ens_idx = int(point['ens_idx'])
    cell_idx = int(point['cell_idx'])
    seed = int(cfg['seed_base']) + 1000 * ens_idx \
        + 1000_000 * cell_idx
    traj_config = {
        'T_sub': T_sub,
        'alpha': float(cfg['alpha']),
        'gamma': float(cfg['gamma']),
        'Ms': float(cfg['Ms']),
        'A_ex': float(cfg['A_ex']),
        'D': float(cfg['D']),
        'K_eff': float(cfg['K_eff']),
        'H_ext': np.asarray(cfg['H_ext'], dtype=float),
        'a': float(cfg['a']),
        't_Co': float(cfg['t_Co']),
        'nx': int(cfg['nx']),
        'ny': int(cfg['ny']),
        'skyrmion_R': float(cfg['skyrmion_R']),
        'skyrmion_dw': float(cfg['skyrmion_dw']),
        'dt': float(cfg['dt']),
        'n_relax': int(cfg['n_relax']),
        'n_drive': int(cfg['n_drive']),
        'sample_every': int(cfg['sample_every']),
        'seed': seed,
        'tol_norm': float(cfg['tol_norm']),
        'q_threshold': float(cfg['q_threshold']),
        'k_consecutive': int(cfg['k_consecutive']),
    }
    payload = run_collapse_trajectory(traj_config)
    out_name = f'T{T_sub:06.1f}_ens{ens_idx:04d}.npz'
    out_path = os.path.join(cfg['out_dir'], out_name)
    np.savez_compressed(
        out_path,
        T_sub=T_sub,
        ens_idx=ens_idx,
        t_collapse=payload['t_collapse'],
        alive_at_end=payload['alive_at_end'],
        flip_index=payload['flip_index'],
        T_effective=payload['T_effective'],
        sigma_noise=payload['sigma_noise'],
        t_sample=payload['t_sample'],
        Q=payload['Q'],
        D=float(cfg['D']),
        H_ext=np.asarray(cfg['H_ext'], dtype=float),
        dt=float(cfg['dt']),
        n_drive=int(cfg['n_drive']),
        sample_every=int(cfg['sample_every']),
    )
    tag = 'survived' if payload['alive_at_end'] \
        else f't_collapse={payload["t_collapse"]*1e9:.2f}ns'
    return (
        f'  T={T_sub:6.1f} K  ens={ens_idx:04d}  '
        f'flip_index={payload["flip_index"]:6d}  {tag}')


# -----------------------------------------------------------------------------
def main():
    # =========================== User Configuration =========================
    # RT 2013 / Rohart-compatible micromagnetic parameters
    # (A from the J*sqrt(3)/2 atomistic mapping = 16 pJ/m).
    A_ex            = 16.0e-12       # J/m
    K_eff           = 0.50e6         # J/m^3 (easy-axis, eff.)
    Ms              = 1.1e6          # A/m (RT 2013 Co/Pt)
    D               = 3.0e-3         # J/m^2 (D/D_c = 0.83 < 1)
    alpha           = 0.3
    gamma           = 1.760e11       # rad/(s T)
    a               = 2.0e-9         # m (validated dt stability;
    #                                  Delta ~ 5.7 nm => ~3 cells)
    t_Co            = 0.6e-9         # m (ultrathin Co; FDT vol.)
    nx              = 128            # 256 nm box (~Rohart 200 nm)
    ny              = 128
    # Destabilizing field opposite the core (core m_z = -1).
    # Rohart's atomistic 250 mT is SUPERCRITICAL here (H_c ~ 135
    # mT for this micromagnetic skyrmion -> athermal instant
    # collapse). 95 mT is sub-critical: calibration gives a
    # clean activated ladder tau ~ 1-7 ns over T = 40-85 K
    # (saturates at the ~0.9 ns dynamical floor above ~85 K).
    H_ext           = [0.0, 0.0, 0.095]  # Tesla (sub-critical)
    # Seed skyrmion: Delta = sqrt(A/K); R_s = Eq.(18) estimate.
    Delta           = math.sqrt(A_ex / K_eff)
    Dc              = (4.0 / math.pi) * math.sqrt(A_ex * K_eff)
    skyrmion_R      = Delta / math.sqrt(2.0 * (1.0 - D / Dc))
    skyrmion_dw     = Delta
    # Temperatures in the activated regime calibrated at 95 mT:
    # tau falls monotonically ~12->1 ns over 35-85 K, then
    # saturates at the dynamical floor above ~85 K. Fine 5 K
    # grid over 35-80 K (capped below the floor) so the
    # Arrhenius fit is well sampled and not floor-biased.
    t_sub_list      = [35.0, 40.0, 45.0, 50.0, 55.0,
                       60.0, 65.0, 70.0, 75.0, 80.0]
    n_ens           = 128
    dt              = 2.5e-14        # s (drift < tol)
    n_relax         = 8000           # 200 ps settle (no clock)
    n_drive         = 2_400_000      # 60 ns observation window
    sample_every    = 400            # 10 ps cadence
    seed_base       = 4099
    tol_norm        = 5.0e-3
    q_threshold     = 0.5
    k_consecutive   = 10
    out_dir         = ('output/stochastic_llgs/validation/'
                       'skyrmion_arrhenius')
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    if not (D < Dc):
        raise RuntimeError(
            f'scan_skyrmion_arrhenius: need D < D_c for a '
            f'metastable isolated skyrmion (FM ground state); '
            f'got D = {D*1e3:.3f} mJ/m^2, D_c = {Dc*1e3:.3f} '
            f'mJ/m^2.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    grid = []
    for cell_idx, T_sub in enumerate(t_sub_list):
        for ens_idx in range(int(n_ens)):
            grid.append({
                'T_sub': float(T_sub),
                'ens_idx': int(ens_idx),
                'cell_idx': int(cell_idx),
            })
    cfg = {
        'A_ex': A_ex, 'K_eff': K_eff, 'Ms': Ms, 'D': D,
        'alpha': alpha, 'gamma': gamma, 'a': a, 't_Co': t_Co,
        'nx': nx, 'ny': ny, 'H_ext': H_ext,
        'skyrmion_R': skyrmion_R, 'skyrmion_dw': skyrmion_dw,
        'dt': dt, 'n_relax': n_relax, 'n_drive': n_drive,
        'sample_every': sample_every, 'seed_base': seed_base,
        'tol_norm': tol_norm, 'q_threshold': q_threshold,
        'k_consecutive': k_consecutive, 'out_dir': out_dir,
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(grid):
            raise RuntimeError(
                f'scan_skyrmion_arrhenius: SLURM_ARRAY_TASK_ID'
                f' = {task_id} out of range '
                f'[0, {len(grid) - 1}].')
        grid = [grid[idx]]
    print(
        f'scan_skyrmion_arrhenius: {len(grid)} trajectories '
        f'(lattice={nx}x{ny} @ a={a*1e9:.1f} nm, '
        f'D={D*1e3:.2f} mJ/m^2, D/D_c={D/Dc:.2f}, '
        f'R_s={skyrmion_R*1e9:.1f} nm, '
        f'|H|={H_ext[2]*1e3:.0f} mT, n_ens={n_ens})')
    for point in grid:
        print(_run_one_point((cfg, point)), flush=True)


# =============================================================================
if __name__ == '__main__':
    main()
