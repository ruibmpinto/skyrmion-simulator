"""Finite-T skyrmion-size scan in a dot (Tomasello 2018, #9b).

The stochastic companion to the deterministic
`test_skyrmion_size_vs_T_tomasello2018` (#9a). Reproduces the
symbols (mean +/- std) of Tomasello 2018 Fig. 1(b): the
skyrmion diameter as a function of temperature, including
thermal fluctuations, at H_ext = 0, 25, 50 mT.

For each (H, T, seed) the micromagnetic parameters are scaled
by the reduced magnetization m(T) (same Callen-Callen exponents
and digitized m(T) as #9a), an isolated skyrmion is seeded in a
free-BC dot, thermalised under Brown noise, then its core
radius R_sk(t) = 0.5 * skyrmion_diameter is sampled over the
observation window. Downstream `test_skyrmion_size_thermal`
pools the samples to <R_sk> +/- std per (H, T) and gates the
H = 0 expansion ratio.

Production reuse: the single FM layer is stepped by the
production `heun_stochastic_step` in single-layer mode
(`m_bot=None`) with the free-BC `mask` kwarg; parameters via
`_make_rt_params`; the FDT noise via `attach_thermal` /
`sample_thermal_field`. The noise is zeroed outside the mask so
the frozen exterior does not random-walk.

Run with:
    python -m \
    skyrmion_simulator.phase_diagram.validation.scan_skyrmion_size_thermal
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
from skyrmion_simulator.phase_diagram.validation \
    .test_confined_skyrmion_radius_rt2013 import _make_rt_params
from skyrmion_simulator.phase_diagram.validation \
    .test_skyrmion_size_vs_T_tomasello2018 import _ALPHA_A, _BETA_D, \
    _GAMMA_K, _m_of_T
from skyrmion_simulator.simulator.analysis import skyrmion_diameter
from skyrmion_simulator.simulator.initial_conditions import skyrmion_profile
from skyrmion_simulator.simulator.lattice import disk_mask
from skyrmion_simulator.stochastic_llgs.integrator_sllg import \
    heun_stochastic_step
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
_HALF_MU0 = 0.5 * (4.0e-7 * math.pi)


def _scaled_params(cfg, T, H, seed):
    """Build a dot parameter namespace with m(T)-scaled
    constants and attached thermal noise."""
    m = _m_of_T(T)
    Ms = float(cfg['Ms0']) * m
    A = float(cfg['A0']) * m ** _ALPHA_A
    D = float(cfg['D0']) * m ** _BETA_D
    Ku = float(cfg['Ku0']) * m ** _GAMMA_K
    # Thin-film shape fold: effective anisotropy = uniaxial Ku
    # minus the 0.5 mu0 Ms^2 demag term.
    K_eff = Ku - _HALF_MU0 * Ms * Ms
    p = _make_rt_params(
        A_ex=A, D=D, K_eff=K_eff, Ms=Ms,
        alpha=float(cfg['alpha']), gamma=float(cfg['gamma']),
        nx=int(cfg['nx']), ny=int(cfg['ny']),
        a=float(cfg['a']), dt=float(cfg['dt']))
    p.t_Co = float(cfg['t_Co'])
    p.H_ext = np.array([0.0, 0.0, float(H)])
    attach_thermal(p, T=float(T), R_th=0.0, seed=int(seed))
    return p, m, K_eff


def _run_one_trajectory(args):
    """Thermalise one dot skyrmion and sample R_sk(t)."""
    cfg, point = args
    H = float(point['H'])
    T = float(point['T'])
    seed = int(point['seed'])
    p, m, K_eff = _scaled_params(cfg, T, H, seed)
    nx, ny, a = p.nx, p.ny, p.a
    dt = float(cfg['dt'])
    mask = disk_mask(nx=nx, ny=ny, a=a, R=float(cfg['R_dot']))
    # Broadcastable mask used to zero the noise outside the dot.
    m3 = mask[..., np.newaxis]
    m_top = skyrmion_profile(
        nx, ny, a=a, R=float(cfg['R_init']),
        dw=math.sqrt(p.A_ex / K_eff), polarity=+1)
    m_top[~mask, :] = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(int(p.seed))
    sigma = float(p.sigma_noise)
    tol_norm = float(cfg['tol_norm'])
    n_relax = int(cfg['n_relax'])
    n_drive = int(cfg['n_drive'])
    sample_every = int(cfg['sample_every'])
    zero_h = np.zeros((ny, nx, 3), dtype=float)
    # Thermalisation (samples discarded). Noise zeroed outside
    # the dot so the frozen exterior stays put.
    for _ in range(n_relax):
        h = (sample_thermal_field(rng, (ny, nx), sigma, dt) * m3
             if sigma > 0.0 else zero_h)
        m_top, _b, _d = heun_stochastic_step(
            m_top, None, dt, p, None, h, None, tol_norm,
            t=0.0, mask=mask)
    # Observation window: sample R_sk every sample_every.
    n_samples = (n_drive + sample_every - 1) // sample_every
    R_sk = np.empty(n_samples, dtype=float)
    s_idx = 0
    for step in range(1, n_drive + 1):
        h = (sample_thermal_field(rng, (ny, nx), sigma, dt) * m3
             if sigma > 0.0 else zero_h)
        m_top, _b, _d = heun_stochastic_step(
            m_top, None, dt, p, None, h, None, tol_norm,
            t=step * dt, mask=mask)
        if step % sample_every == 0:
            R_sk[s_idx] = 0.5 * skyrmion_diameter(
                m_top, a, core_polarity=+1)
            s_idx += 1
    out_name = (
        f'H{H*1e3:05.1f}_T{T:06.1f}_ens{seed:04d}.npz')
    out_path = os.path.join(cfg['out_dir'], out_name)
    np.savez_compressed(
        out_path, H=H, T=T, seed=seed, m=m, R_sk=R_sk,
        R_sk_mean=float(np.mean(R_sk)),
        R_sk_std=float(np.std(R_sk, ddof=1)),
        T_effective=float(p.T), sigma_noise=sigma,
        R_dot=float(cfg['R_dot']))
    return (f'  H={H*1e3:4.0f} mT  T={T:6.1f} K  '
            f'seed={seed:04d}  <R_sk>={np.mean(R_sk)*1e9:5.1f}'
            f'+/-{np.std(R_sk)*1e9:4.1f} nm')


def main():
    # Build the (H, T, seed) grid and run one trajectory per
    # point (or one SLURM-array task per point).
    # =========================== User Configuration =========================
    # Tomasello 2018 T = 0 set (same as #9a deterministic).
    Ms0 = 0.60e6             # A/m
    A0 = 20.0e-12            # J/m
    Ku0 = 0.60e6             # J/m^3
    D0 = 3.0e-3              # J/m^2
    R_dot = 141.0e-9         # m
    t_Co = 0.8e-9            # m (dot thickness; FDT volume)
    alpha = 0.3
    gamma = 1.760e11
    a = 2.5e-9
    nx = 128
    ny = 128
    R_init = 30.0e-9
    H_list = [0.0, 25.0e-3, 50.0e-3]
    T_list = [0., 50., 100., 150., 200., 250., 300.]
    n_seed = 8
    dt = 1.25e-14
    n_relax = 16000          # 200 ps thermalisation (discarded)
    n_drive = 80000          # 1 ns sampling window
    sample_every = 800       # 10 ps cadence (100 samples/traj)
    seed_base = 9001
    tol_norm = 5.0e-3
    out_dir = ('output/phase_diagram/validation/'
               'skyrmion_size_thermal')
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    cfg = {
        'Ms0': Ms0, 'A0': A0, 'Ku0': Ku0, 'D0': D0,
        'R_dot': R_dot, 't_Co': t_Co, 'alpha': alpha,
        'gamma': gamma, 'a': a, 'nx': nx, 'ny': ny,
        'R_init': R_init, 'dt': dt, 'n_relax': n_relax,
        'n_drive': n_drive, 'sample_every': sample_every,
        'tol_norm': tol_norm, 'out_dir': out_dir,
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Flatten (H, T, seed) into one task list; the seed offset
    # keeps every trajectory's RNG stream distinct.
    grid = []
    cell_idx = 0
    for H in H_list:
        for T in T_list:
            for s in range(int(n_seed)):
                grid.append({
                    'H': float(H), 'T': float(T),
                    'seed': int(seed_base) + 1000 * s
                    + 1000_000 * cell_idx})
            cell_idx += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SLURM-array mode: run only the single task this index maps
    # to; otherwise run every trajectory in serial below.
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(grid):
            raise RuntimeError(
                f'scan_skyrmion_size_thermal: '
                f'SLURM_ARRAY_TASK_ID={task_id} out of range '
                f'[0, {len(grid) - 1}].')
        grid = [grid[idx]]
    print(f'scan_skyrmion_size_thermal: {len(grid)} '
          f'trajectories ({len(H_list)} H x {len(T_list)} T x '
          f'{n_seed} seed, dot R={R_dot*1e9:.0f} nm, '
          f'{nx}x{ny} @ a={a*1e9:.1f} nm)')
    for point in grid:
        print(_run_one_trajectory((cfg, point)), flush=True)


# =============================================================================
if __name__ == '__main__':
    main()
