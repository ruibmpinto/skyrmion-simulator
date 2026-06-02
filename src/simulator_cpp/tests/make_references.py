"""Generate references.npz with Python-computed expected outputs for
every parity test in src/simulator_cpp/tests/.

The C++ tests load this file and compare their results element-wise.

All test cases share a single input lattice (32 x 32, a=2 nm) and a
single Params namespace (default Co/Pt SAF with nx=ny=32 patched).

Functions
---------
main
    Build references.npz and print a summary.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pathlib
import sys
from types import SimpleNamespace
# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

# Repo root: tests/ -> simulator_cpp/ -> src/ -> repo_root.
_REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

# Local (after path setup)
from src.simulator.parameters import default_params
from src.simulator.pulses import (
    ConstantPulse, SquarePulse, GaussianPulse, SuperpositionPulse,
)
from src.simulator.lattice import lattice_positions
from src.simulator.initial_conditions import (
    skyrmion_profile, uniform_state, saf_skyrmion,
)
from src.simulator import energy as energy_mod
from src.simulator.demag import precompute_demag_kernels, demag_field
from src.simulator.demag_newell import precompute_demag_kernels_newell
from src.simulator.fields import (
    effective_field, effective_field_demag_pair, bare_anis_prefactors,
)
from src.simulator.integrator import (
    normalize, llgs_rhs, rhs_local_keff, rhs_demag, rk4_step,
)
from src.simulator.main import topological_charge
from src.simulator.analysis import (
    skyrmion_center, skyrmion_diameter, skyrmion_ellipse, dw_angle,
)
from src.stochastic_llgs.diagnostics import (
    skyrmion_center_pbc,
    largest_core_mask_pbc,
    skyrmion_diameter_lcc,
    skyrmion_center_lcc_pbc,
    unwrap_trajectory,
    detect_annihilation,
    hall_angle,
)
from src.orchestrator.observers import observe_state
from src.stochastic_llgs.parameters_thermal import attach_thermal
from src.stochastic_llgs.joule_heating import T_of_j
from src.stochastic_llgs.integrator_sllg import heun_stochastic_step
from src.stochastic_llgs.experiments.pair_potential import (
    skyrmion_at_position, two_skyrmion_pair_ic,
)
from src.stochastic_llgs.validation.test_langevin import (
    langevin_function as py_langevin,
)
from src.stochastic_llgs.validation.test_brown_reversal import (
    brown_tau as py_brown_tau,
)
from src.stochastic_llgs.validation.test_equipartition import (
    magnon_stiffness_grid as py_magnon,
)


def main():
    """Build references.npz next to this script."""
    out_dir = pathlib.Path(__file__).parent / 'reference'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'references.npz'
    refs = {}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # State (shared lattice + magnetization).
    nx = 32
    ny = 32
    a = 2.0e-9
    R = 30.0e-9
    dw = 15.0e-9
    rng = np.random.default_rng(seed=20260526)
    refs.update({
        'state_nx': np.int64(nx),
        'state_ny': np.int64(ny),
        'state_a': float(a),
        'state_R': float(R),
        'state_dw': float(dw),
    })
    # Canonical SAF state for most tests.
    m_top, m_bot = saf_skyrmion(nx, ny, a, R, dw=dw)
    refs['state_m_top'] = m_top
    refs['state_m_bot'] = m_bot
    # Non-normalized perturbed state for the normalize() test.
    m_perturbed = m_top + 0.05 * rng.standard_normal(m_top.shape)
    refs['state_m_perturbed'] = m_perturbed
    refs['state_pos_top'] = lattice_positions(nx, ny, a)
    pos_bot = lattice_positions(nx, ny, a)
    refs['state_pos_bot'] = pos_bot
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Params (defaults patched with smaller lattice).
    p = default_params()
    p.nx = nx
    p.ny = ny
    p.a = a
    # Re-precompute prefactors after lattice change (parameters._precompute
    # depends on p.a via C_ex).
    from src.simulator.parameters import _precompute as _precompute_inner
    _precompute_inner(p)
    # Scalar params for the C++ side to instantiate Params.
    for k in [
        'Ms', 'A_ex', 'D', 'K_top', 'K_bot', 'alpha', 't_Co', 'd_Ru',
        'mu0', 'gamma', 'H_RKKY', 'DL_SOT', 'FL_SOT', 'J_current',
        'lambda_sq', 'P', 'mu_B_over_q_e', 'dt',
    ]:
        refs[f'params_{k}'] = float(getattr(p, k))
    refs['params_H_ext'] = np.asarray(p.H_ext, dtype=np.float64)
    refs['params_p_hat'] = np.asarray(p.p_hat, dtype=np.float64)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pulses
    t_grid = np.linspace(-1e-9, 2e-9, 17)
    refs['pulses_t_grid'] = t_grid
    refs['pulses_const_J0'] = float(p.J_current)
    refs['pulses_const_out'] = np.array(
        [ConstantPulse(p.J_current)(t) for t in t_grid])
    sq_J0, sq_ts, sq_te = float(p.J_current), 0.0, 5e-10
    refs['pulses_square_J0'] = sq_J0
    refs['pulses_square_t_start'] = sq_ts
    refs['pulses_square_t_end'] = sq_te
    refs['pulses_square_out'] = np.array(
        [SquarePulse(sq_J0, sq_ts, sq_te)(t) for t in t_grid])
    g_J0, g_tc, g_fwhm = float(p.J_current), 5e-10, 3e-10
    refs['pulses_gauss_J0'] = g_J0
    refs['pulses_gauss_t_center'] = g_tc
    refs['pulses_gauss_fwhm'] = g_fwhm
    refs['pulses_gauss_out'] = np.array(
        [GaussianPulse(g_J0, g_tc, g_fwhm)(t) for t in t_grid])
    sup = SuperpositionPulse(
        [SquarePulse(sq_J0, sq_ts, sq_te),
         GaussianPulse(g_J0, g_tc, g_fwhm)])
    refs['pulses_super_out'] = np.array([sup(t) for t in t_grid])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameters precompute (scalar prefactors)
    refs['precompute_C_ex'] = float(p.C_ex)
    refs['precompute_C_dmi'] = float(p.C_dmi)
    refs['precompute_C_anis_top'] = float(p.C_anis_top)
    refs['precompute_C_anis_bot'] = float(p.C_anis_bot)
    refs['precompute_H_DL'] = float(p.H_DL)
    refs['precompute_H_FL'] = float(p.H_FL)
    refs['precompute_gamma_p'] = float(p.gamma_p)
    C_top_bare, C_bot_bare = bare_anis_prefactors(p)
    refs['precompute_bare_C_top'] = float(C_top_bare)
    refs['precompute_bare_C_bot'] = float(C_bot_bare)
    ea = energy_mod.effective_anisotropy(p)
    refs['precompute_K_eff_top'] = float(ea['K_eff_top'])
    refs['precompute_K_eff_bot'] = float(ea['K_eff_bot'])
    refs['precompute_K_eff_avg'] = float(ea['K_eff_avg'])
    refs['precompute_mu0_Ms2_over_2'] = float(ea['mu0_Ms2_over_2'])
    refs['precompute_D_c'] = float(energy_mod.critical_dmi(p))
    refs['precompute_H_K'] = float(energy_mod.pma_anisotropy_field(p))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Lattice
    refs['lattice_pos'] = lattice_positions(nx, ny, a)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial conditions
    refs['init_skyrmion_pol_plus'] = skyrmion_profile(
        nx, ny, a, R, dw=dw, polarity=+1)
    refs['init_skyrmion_pol_minus'] = skyrmion_profile(
        nx, ny, a, R, dw=dw, polarity=-1)
    udir = np.array([0.3, -0.4, 0.5])
    refs['init_uniform_dir'] = udir
    refs['init_uniform'] = uniform_state(nx, ny, direction=udir)
    saf_top, saf_bot = saf_skyrmion(nx, ny, a, R, dw=dw)
    refs['init_saf_top'] = saf_top
    refs['init_saf_bot'] = saf_bot
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Demag (slab kernel) - full (ny, nx) spectrum with 1/(ny*nx) baked
    # in to match the C++ DemagKernels storage layout.
    norm_fft = 1.0 / (ny * nx)
    K_slab = precompute_demag_kernels(p, kind='slab',
                                      accuracy=None, tol_conv=None)
    for k in ['Nxx_self', 'Nyy_self', 'Nxy_self', 'Nzz_self',
              'Nxx_inter', 'Nyy_inter', 'Nxy_inter', 'Nzz_inter',
              'Nxz_inter', 'Nyz_inter']:
        refs[f'demag_slab_{k}'] = K_slab[k].astype(np.complex128) * norm_fft
    K_newell = precompute_demag_kernels_newell(p,
                                               accuracy=8.0, tol_conv=2.0e-2)
    for k in ['Nxx_self', 'Nyy_self', 'Nxy_self', 'Nzz_self',
              'Nxx_inter', 'Nyy_inter', 'Nxy_inter', 'Nzz_inter',
              'Nxz_inter', 'Nyz_inter']:
        refs[f'demag_newell_{k}'] = K_newell[k].astype(np.complex128) * norm_fft
    # Demag field applied via slab kernel.
    H_dt, H_db = demag_field(m_top, m_bot, K_slab)
    refs['demag_field_slab_H_top'] = H_dt
    refs['demag_field_slab_H_bot'] = H_db
    H_dt_n, H_db_n = demag_field(m_top, m_bot, K_newell)
    refs['demag_field_newell_H_top'] = H_dt_n
    refs['demag_field_newell_H_bot'] = H_db_n
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Effective field (local-K_eff path).
    H_top_keff = effective_field(
        m_top, m_bot, p.C_ex, p.C_dmi, p.C_anis_top, p.H_ext, p.H_RKKY)
    H_bot_keff = effective_field(
        m_bot, m_top, p.C_ex, p.C_dmi, p.C_anis_bot, p.H_ext, p.H_RKKY)
    refs['fields_H_top_keff'] = H_top_keff
    refs['fields_H_bot_keff'] = H_bot_keff
    # Effective field including demag (bare-K path, slab kernel).
    H_top_demag, H_bot_demag = effective_field_demag_pair(
        m_top, m_bot, p, K_slab)
    refs['fields_H_top_demag'] = H_top_demag
    refs['fields_H_bot_demag'] = H_bot_demag
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Energy
    refs['energy_E_slab'] = float(
        energy_mod.total_energy(m_top, m_bot, p, K_slab))
    refs['energy_E_newell'] = float(
        energy_mod.total_energy(m_top, m_bot, p, K_newell))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Integrator
    refs['integrator_normalized'] = normalize(m_perturbed)
    # llgs_rhs at t=0 with the local-K_eff effective field.
    refs['integrator_t_eval'] = 0.0
    refs['integrator_dmdt_top_keff'] = llgs_rhs(m_top, H_top_keff, p, 0.0)
    refs['integrator_dmdt_bot_keff'] = llgs_rhs(m_bot, H_bot_keff, p, 0.0)
    # rhs_local_keff: full layered RHS for both layers in one call.
    dt_top, dt_bot = rhs_local_keff(m_top, m_bot, p, 0.0)
    refs['integrator_rhs_keff_top'] = dt_top
    refs['integrator_rhs_keff_bot'] = dt_bot
    # rhs_demag (slab) RHS.
    rhs_dem = rhs_demag(K_slab)
    dt_top_d, dt_bot_d = rhs_dem(m_top, m_bot, p, 0.0)
    refs['integrator_rhs_demag_top'] = dt_top_d
    refs['integrator_rhs_demag_bot'] = dt_bot_d
    # One RK4 step.
    m_top_rk4, m_bot_rk4 = rk4_step(
        rhs_local_keff, m_top, m_bot, 0.0, p.dt, p)
    refs['integrator_rk4_keff_top'] = m_top_rk4
    refs['integrator_rk4_keff_bot'] = m_bot_rk4
    m_top_rk4d, m_bot_rk4d = rk4_step(
        rhs_dem, m_top, m_bot, 0.0, p.dt, p)
    refs['integrator_rk4_demag_top'] = m_top_rk4d
    refs['integrator_rk4_demag_bot'] = m_bot_rk4d
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Observables (top-layer convention: polarity = +1, core at m_z = -1).
    refs['observables_Q_top'] = float(topological_charge(m_top, a))
    refs['observables_Q_bot'] = float(topological_charge(m_bot, a))
    cx, cy = skyrmion_center(m_top, a, core_polarity=+1)
    refs['observables_cx_top'] = float(cx)
    refs['observables_cy_top'] = float(cy)
    refs['observables_diameter_top'] = float(
        skyrmion_diameter(m_top, a, core_polarity=+1))
    D1, D2, theta = skyrmion_ellipse(m_top, a, core_polarity=+1)
    refs['observables_D1_top'] = float(D1)
    refs['observables_D2_top'] = float(D2)
    refs['observables_theta_top'] = float(theta)
    refs['observables_psi_top'] = float(
        dw_angle(m_top, a, core_polarity=+1, mz_thresh=0.5))
    # PBC-safe centroid (top +1, bottom -1).
    cx_p, cy_p = skyrmion_center_pbc(m_top, a, +1)
    refs['observables_cx_top_pbc'] = float(cx_p)
    refs['observables_cy_top_pbc'] = float(cy_p)
    cxb_p, cyb_p = skyrmion_center_pbc(m_bot, a, -1)
    refs['observables_cx_bot_pbc'] = float(cxb_p)
    refs['observables_cy_bot_pbc'] = float(cyb_p)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Full sweep observe_state payload (16 scalars). Uses the PBC-safe
    # centroid, so its cx/cy differ from observables_cx_top above.
    obs = observe_state(m_top, m_bot, p)
    for k, v in obs.items():
        refs[f'observe_{k}'] = float(v)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Simulation parity: 50 relax steps + 20 drive steps, dump m_top
    # after each phase end. Mirrors the C++ run loop with dump_snapshots
    # disabled (we keep the final state only).
    from copy import deepcopy as _dc
    p2 = _dc(p)
    p2.dt = 5e-14
    n_relax = 50
    n_steps = 20
    # Phase 0: J = 0
    p2.H_DL = 0.0
    p2.H_FL = 0.0
    p2.pulse = ConstantPulse(0.0)
    mt = m_top.copy()
    mb = m_bot.copy()
    t = 0.0
    for _ in range(n_relax):
        mt, mb = rk4_step(rhs_local_keff, mt, mb, t, p2.dt, p2)
        t += p2.dt
    refs['simulation_m_top_after_relax'] = mt.copy()
    refs['simulation_m_bot_after_relax'] = mb.copy()
    # Phase 1: J = default
    p3 = _dc(p)
    p3.dt = 5e-14
    p3.pulse = ConstantPulse(p3.J_current)
    t = 0.0
    for _ in range(n_steps):
        mt, mb = rk4_step(rhs_local_keff, mt, mb, t, p3.dt, p3)
        t += p3.dt
    refs['simulation_m_top_after_drive'] = mt.copy()
    refs['simulation_m_bot_after_drive'] = mb.copy()
    refs['simulation_n_relax'] = np.int64(n_relax)
    refs['simulation_n_steps'] = np.int64(n_steps)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Stochastic thermal parameters (attach_thermal, T_of_j).
    pth = default_params()
    pth.nx = nx
    pth.ny = ny
    pth.a = a
    _precompute_inner(pth)
    attach_thermal(pth, T=100.0, R_th=0.0, seed=12345)
    refs['thermal_T'] = 100.0
    refs['thermal_sigma_noise'] = float(pth.sigma_noise)
    refs['thermal_V_cell'] = float(pth.V_cell)
    refs['thermal_T_of_j'] = float(
        T_of_j(4.0e11, T_sub=300.0, R_th=1.0e-20))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Heun stepper at T = 0 (zero noise => deterministic RK2). N steps
    # from the canonical SAF state under the default SOT drive, for the
    # local-K_eff (kernels=None) and slab-demag paths.
    n_heun = 20
    refs['heun_t0_N'] = np.int64(n_heun)
    zero = np.zeros((ny, nx, 3), dtype=float)
    tol_norm = 1.0
    ph = default_params()
    ph.nx = nx
    ph.ny = ny
    ph.a = a
    _precompute_inner(ph)
    ph.pulse = ConstantPulse(ph.J_current)
    hmt = m_top.copy()
    hmb = m_bot.copy()
    t_h = 0.0
    for _ in range(n_heun):
        hmt, hmb, _ = heun_stochastic_step(
            hmt, hmb, ph.dt, ph, None, zero, zero, tol_norm, t=t_h)
        t_h += ph.dt
    refs['heun_t0_nodemag_m_top'] = hmt
    refs['heun_t0_nodemag_m_bot'] = hmb
    K_heun = precompute_demag_kernels(
        ph, kind='slab', accuracy=None, tol_conv=None)
    hmt = m_top.copy()
    hmb = m_bot.copy()
    t_h = 0.0
    for _ in range(n_heun):
        hmt, hmb, _ = heun_stochastic_step(
            hmt, hmb, ph.dt, ph, K_heun, zero, zero, tol_norm, t=t_h)
        t_h += ph.dt
    refs['heun_t0_slab_m_top'] = hmt
    refs['heun_t0_slab_m_bot'] = hmb
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LCC diagnostics: mask, diameter, centre for the canonical state
    # (single dominant component) and a half-box-rolled state (core
    # straddles both wraps -> exercises the PBC union-find fusion).
    refs['lcc_mask_top'] = largest_core_mask_pbc(
        m_top, +1).astype(np.uint8)
    refs['lcc_mask_bot'] = largest_core_mask_pbc(
        m_bot, -1).astype(np.uint8)
    refs['lcc_diam_top'] = float(skyrmion_diameter_lcc(m_top, a, +1))
    cxl, cyl = skyrmion_center_lcc_pbc(m_top, a, +1)
    refs['lcc_cx_top'] = float(cxl)
    refs['lcc_cy_top'] = float(cyl)
    m_rolled = np.roll(m_top, shift=(ny // 2, nx // 2), axis=(0, 1))
    refs['lcc_rolled_m'] = m_rolled
    refs['lcc_mask_rolled'] = largest_core_mask_pbc(
        m_rolled, +1).astype(np.uint8)
    refs['lcc_diam_rolled'] = float(
        skyrmion_diameter_lcc(m_rolled, a, +1))
    cxr, cyr = skyrmion_center_lcc_pbc(m_rolled, a, +1)
    refs['lcc_cx_rolled'] = float(cxr)
    refs['lcc_cy_rolled'] = float(cyr)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Stochastic trajectory diagnostics: unwrap, hall, detect on a
    # synthetic moving-and-wrapping centre history (exact parity).
    rng_d = np.random.default_rng(seed=7)
    n_d = 24
    L_x = 64.0e-9
    L_y = 64.0e-9
    t_d = np.arange(n_d, dtype=float) * 5.0e-12
    # Linear drift + small jitter, wrapped into [0, L).
    cx_true = 5.0e-9 + 1.2e3 * t_d + 0.3e-9 * rng_d.standard_normal(n_d)
    cy_true = 8.0e-9 + 0.4e3 * t_d + 0.3e-9 * rng_d.standard_normal(n_d)
    cx_wrap = np.mod(cx_true, L_x)
    cy_wrap = np.mod(cy_true, L_y)
    refs['diag_t'] = t_d
    refs['diag_cx_wrap'] = cx_wrap
    refs['diag_cy_wrap'] = cy_wrap
    refs['diag_L_x'] = L_x
    refs['diag_L_y'] = L_y
    cxu, cyu = unwrap_trajectory(cx_wrap, cy_wrap, L_x, L_y)
    refs['diag_cx_unwrap'] = cxu
    refs['diag_cy_unwrap'] = cyu
    vx, vy, th = hall_angle(t_d, cxu, cyu, half=0.5)
    refs['diag_v_x'] = float(vx)
    refs['diag_v_y'] = float(vy)
    refs['diag_hall_deg'] = float(th)
    # Annihilation: |Q| drops below threshold and stays there.
    Q_hist = np.concatenate([
        np.full(10, 0.98), np.array([0.6, 0.4]),
        np.full(12, 0.1)]).astype(float)
    refs['diag_Q_hist'] = Q_hist
    refs['diag_q_threshold'] = 0.5
    refs['diag_k_consecutive'] = np.int64(10)
    refs['diag_flip_index'] = np.int64(
        detect_annihilation(Q_hist, 0.5, 10))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Two-skyrmion pair IC. Centres are placed off-grid (non-integer
    # multiples of a) so no lattice site sits at a core (r ~ 0), where
    # phi = atan2(y, x) is a coordinate singularity decided by sub-ULP
    # rounding. They are also asymmetric so the core-favouring m_z merge
    # tie-break is well-posed away from the (measure-zero) seam.
    p_nx, p_ny = 64, 64
    p_R, p_dw = 20.0e-9, 8.0e-9
    c1p = (31.0e-9, 27.0e-9)
    c2p = (95.0e-9, 73.0e-9)
    # Single off-centre skyrmion: bit-exact parity target (no merge,
    # no tie-break amplification).
    refs['pair_ic_single_m'] = skyrmion_at_position(
        p_nx, p_ny, a, p_R, p_dw, 1, c1p[0], c1p[1])
    pm_top, pm_bot = two_skyrmion_pair_ic(
        p_nx, p_ny, a, p_R, p_dw, polarity_top=1, c1=c1p, c2=c2p)
    refs['pair_ic_nx'] = np.int64(p_nx)
    refs['pair_ic_ny'] = np.int64(p_ny)
    refs['pair_ic_R'] = float(p_R)
    refs['pair_ic_dw'] = float(p_dw)
    refs['pair_ic_c1x'] = float(c1p[0])
    refs['pair_ic_c1y'] = float(c1p[1])
    refs['pair_ic_c2x'] = float(c2p[0])
    refs['pair_ic_c2y'] = float(c2p[1])
    refs['pair_ic_m_top'] = pm_top
    refs['pair_ic_m_bot'] = pm_bot
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Validation-gate analytic targets (deterministic, bit-checkable).
    val_lx = np.array([1.0e-5, 0.5, 2.0, 10.0])
    refs['val_langevin_x'] = val_lx
    refs['val_langevin_L'] = py_langevin(val_lx)
    val_b_alpha, val_b_gamma = 0.5, 194.8e9
    val_b_delta = np.array([3.0, 5.0, 8.0])
    refs['val_brown_alpha'] = float(val_b_alpha)
    refs['val_brown_gamma'] = float(val_b_gamma)
    refs['val_brown_delta'] = val_b_delta
    refs['val_brown_tau'] = np.array([
        py_brown_tau(d, val_b_alpha, val_b_gamma) for d in val_b_delta])
    mg_ny, mg_nx = 8, 8
    mg_Bz = 0.5
    mg_Cex = 2.0 * 16.0e-12 / (1.43e6 * (2.0e-9) ** 2)
    p_mag = SimpleNamespace(
        H_ext=np.array([0.0, 0.0, mg_Bz]), a=2.0e-9, C_ex=mg_Cex)
    refs['val_magnon_ny'] = np.int64(mg_ny)
    refs['val_magnon_nx'] = np.int64(mg_nx)
    refs['val_magnon_Bz'] = float(mg_Bz)
    refs['val_magnon_Cex'] = float(mg_Cex)
    refs['val_magnon_Hk'] = py_magnon(p_mag, mg_ny, mg_nx)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save
    np.savez(out_path, **refs)
    print(f'make_references: wrote {out_path}')
    print(f'  {len(refs)} entries')


# =============================================================================
if __name__ == '__main__':
    main()
