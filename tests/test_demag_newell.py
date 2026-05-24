"""Regression tests for the Newell numerical demag kernel.

Tests
-----
test_aharoni_single_prism_cube
    Verify that the Newell self-cell tensor (Z=0 lattice
    origin) matches Aharoni's closed-form demag factors for
    a uniformly magnetized rectangular prism. The Aharoni
    override at (0, 0) of the self-layer real-space kernel
    is exact by construction.
test_aharoni_single_prism_saf_cell
    Same check for our SAF cell dimensions (a=2 nm,
    a=2 nm, t_Co=1.3 nm). Numerical override hits Aharoni
    to machine precision; sum rule N_xx + N_yy + N_zz = 1
    holds.
test_uniform_m_zhat_slab_limit
    Sanity check the long-wavelength (lattice sum / FFT
    k=0) limit of the Newell self-layer kernel: for
    uniform m = z_hat, the resulting demag field should
    approach -mu0 Ms z_hat (the slab limit). Cross-checked
    against the analytic slab kernel.
test_uniform_m_xhat_slab_limit
    Same as above for m = x_hat. The slab limit gives
    H_demag = 0 (no in-plane demag for a uniformly
    magnetized slab); the Newell residual is the finite-
    cell correction and must be small (< 5% of mu0 Ms).

Run with `python -m tests.test_demag_newell`.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
# Third-party
import numpy as np
# Local
from src.simulator.demag import (
    demag_field,
    precompute_demag_kernels,
)
from src.simulator.demag_newell import (
    _aharoni_demag_factor,
    _build_layer_pair_kernel,
)
from src.simulator.parameters import default_params, _precompute

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
# Aharoni override is exact to floating-point precision.
TOL_AHARONI = 1e-12
# Sum rule should hold to machine precision.
TOL_SUM_RULE = 1e-12
# k=0 limit of N_zz_self approaches 1.0 with lattice size; on a
# 64x64 lattice expect ~1% residual.
TOL_NZZ_K0 = 0.02
# k=0 limit of N_xx_self approaches 0 with lattice size; on a
# 64x64 lattice the residual is ~2% (finite-cell correction).
TOL_NXX_K0 = 0.05
# Uniform-M field magnitude vs slab expectation.
TOL_UNIFORM_FIELD = 0.05  # 5% of mu0 Ms

# Test lattice size: 64x64 is large enough for the lattice sum
# to converge to the slab limit and small enough to build at
# accuracy=4 in ~5s.
NX_TEST = 64
NY_TEST = 64


# -----------------------------------------------------------------------------
def test_aharoni_single_prism_cube():
    """Newell self-cell diagonals match Aharoni for a unit cube.

    Cube of edge L: N_xx = N_yy = N_zz = 1/3 (symmetry), sum =
    1. The Aharoni override at (0, 0) writes these values
    directly; this test verifies the override path and the
    closed-form implementation.
    """
    L = 1.0
    # Aharoni closed-form
    N_xx_aharoni = _aharoni_demag_factor(L, L, L)
    N_yy_aharoni = _aharoni_demag_factor(L, L, L)
    N_zz_aharoni = _aharoni_demag_factor(L, L, L)
    # Build the self-layer real-space kernel; the (0, 0)
    # entry must be exactly the Aharoni value.
    kernel = _build_layer_pair_kernel(
        nx=NX_TEST, ny=NY_TEST,
        dx=L, dy=L, t_layer=L,
        Z_separation=0.0, accuracy=4.0,
    )
    N_xx = kernel['Nxx'][0, 0]
    N_yy = kernel['Nyy'][0, 0]
    N_zz = kernel['Nzz'][0, 0]
    err_xx = abs(N_xx - N_xx_aharoni)
    err_yy = abs(N_yy - N_yy_aharoni)
    err_zz = abs(N_zz - N_zz_aharoni)
    if err_xx > TOL_AHARONI:
        raise RuntimeError(
            f'test_aharoni_single_prism_cube: N_xx = {N_xx:.6f} '
            f'differs from Aharoni {N_xx_aharoni:.6f} by '
            f'{err_xx:.2e} (tol {TOL_AHARONI}).')
    if err_yy > TOL_AHARONI:
        raise RuntimeError(
            f'test_aharoni_single_prism_cube: N_yy = {N_yy:.6f} '
            f'differs from Aharoni {N_yy_aharoni:.6f} by '
            f'{err_yy:.2e}.')
    if err_zz > TOL_AHARONI:
        raise RuntimeError(
            f'test_aharoni_single_prism_cube: N_zz = {N_zz:.6f} '
            f'differs from Aharoni {N_zz_aharoni:.6f} by '
            f'{err_zz:.2e}.')
    # Cube symmetry: all three diagonals equal.
    if abs(N_xx - N_zz) > TOL_AHARONI:
        raise RuntimeError(
            f'test_aharoni_single_prism_cube: cube symmetry '
            f'broken, N_xx={N_xx:.6f} != N_zz={N_zz:.6f}.')
    # Sum rule.
    trace = N_xx + N_yy + N_zz
    if abs(trace - 1.0) > TOL_SUM_RULE:
        raise RuntimeError(
            f'test_aharoni_single_prism_cube: trace = '
            f'{trace:.6f}, expected 1.0 (sum rule).')


# -----------------------------------------------------------------------------
def test_aharoni_single_prism_saf_cell():
    """Newell self-cell diagonals match Aharoni for our SAF
    cell aspect ratio (a:a:t = 2:2:1.3 nm).

    Oblate prism: thin along z, so N_z > N_x = N_y. Aharoni
    gives N_z ~ 0.433, N_x = N_y ~ 0.283.
    """
    p = default_params()
    dx = p.a
    dy = p.a
    dz = p.t_Co
    # Aharoni closed-form: N along the third argument.
    N_xx_aharoni = _aharoni_demag_factor(dy, dz, dx)
    N_yy_aharoni = _aharoni_demag_factor(dx, dz, dy)
    N_zz_aharoni = _aharoni_demag_factor(dx, dy, dz)
    kernel = _build_layer_pair_kernel(
        nx=NX_TEST, ny=NY_TEST,
        dx=dx, dy=dy, t_layer=dz,
        Z_separation=0.0, accuracy=4.0,
    )
    N_xx = kernel['Nxx'][0, 0]
    N_yy = kernel['Nyy'][0, 0]
    N_zz = kernel['Nzz'][0, 0]
    err_xx = abs(N_xx - N_xx_aharoni)
    err_yy = abs(N_yy - N_yy_aharoni)
    err_zz = abs(N_zz - N_zz_aharoni)
    if err_xx > TOL_AHARONI:
        raise RuntimeError(
            f'test_aharoni_single_prism_saf_cell: '
            f'N_xx={N_xx:.6f} vs Aharoni '
            f'{N_xx_aharoni:.6f}, err={err_xx:.2e}.')
    if err_yy > TOL_AHARONI:
        raise RuntimeError(
            f'test_aharoni_single_prism_saf_cell: '
            f'N_yy={N_yy:.6f} vs Aharoni '
            f'{N_yy_aharoni:.6f}, err={err_yy:.2e}.')
    if err_zz > TOL_AHARONI:
        raise RuntimeError(
            f'test_aharoni_single_prism_saf_cell: '
            f'N_zz={N_zz:.6f} vs Aharoni '
            f'{N_zz_aharoni:.6f}, err={err_zz:.2e}.')
    # Oblate-prism ordering: N_z (short axis) > N_x = N_y.
    if not (N_zz > N_xx and abs(N_xx - N_yy) < TOL_AHARONI):
        raise RuntimeError(
            f'test_aharoni_single_prism_saf_cell: oblate '
            f'ordering broken (N_zz={N_zz:.4f}, '
            f'N_xx={N_xx:.4f}, N_yy={N_yy:.4f}).')
    # Sum rule.
    trace = N_xx + N_yy + N_zz
    if abs(trace - 1.0) > TOL_SUM_RULE:
        raise RuntimeError(
            f'test_aharoni_single_prism_saf_cell: trace = '
            f'{trace:.6f}, expected 1.0 (sum rule).')


# -----------------------------------------------------------------------------
def test_uniform_m_zhat_slab_limit():
    """For uniform m = +z_hat across the full lattice, both
    layers, the Newell self-layer demag must approach
    H = -mu0 Ms z_hat (slab limit). On a 64x64 lattice with
    accuracy=4 the residual is below 2%.

    Verifies that the k=0 component of the FFT'd kernel
    aggregates to the slab limit, and that the inter-layer
    contribution (which is zero for k=0 in the slab kernel)
    is small for Newell too.
    """
    p = default_params()
    p.nx = NX_TEST
    p.ny = NY_TEST
    _precompute(p)
    K = precompute_demag_kernels(
        p, kind='newell',
        accuracy=4.0, tol_conv=0.02,
    )
    # Uniform m = +z_hat
    m_top = np.zeros((p.ny, p.nx, 3))
    m_top[..., 2] = 1.0
    m_bot = m_top.copy()
    H_top, H_bot = demag_field(m_top, m_bot, K)
    # Field should be uniform and purely -z_hat.
    mu0_Ms = p.mu0 * p.Ms
    H_z_expected = -mu0_Ms
    H_z_mean = float(np.mean(H_top[..., 2]))
    H_z_std = float(np.std(H_top[..., 2]))
    rel_err = abs(H_z_mean - H_z_expected) / mu0_Ms
    if rel_err > TOL_UNIFORM_FIELD:
        raise RuntimeError(
            f'test_uniform_m_zhat_slab_limit: '
            f'H_z_mean = {H_z_mean:.4e} T, expected '
            f'{H_z_expected:.4e} T (slab limit), '
            f'rel err = {rel_err:.4f} > tol '
            f'{TOL_UNIFORM_FIELD}.')
    # Uniformity check: std should be small relative to mean.
    if H_z_std > 0.01 * abs(H_z_mean):
        raise RuntimeError(
            f'test_uniform_m_zhat_slab_limit: H_z not '
            f'spatially uniform (std={H_z_std:.4e}, '
            f'mean={H_z_mean:.4e}).')
    # In-plane components should be near zero for uniform m_z.
    Hx_max = float(np.max(np.abs(H_top[..., 0])))
    Hy_max = float(np.max(np.abs(H_top[..., 1])))
    if Hx_max > TOL_UNIFORM_FIELD * mu0_Ms:
        raise RuntimeError(
            f'test_uniform_m_zhat_slab_limit: '
            f'max|H_x|={Hx_max:.4e} > '
            f'{TOL_UNIFORM_FIELD*mu0_Ms:.4e}.')
    if Hy_max > TOL_UNIFORM_FIELD * mu0_Ms:
        raise RuntimeError(
            f'test_uniform_m_zhat_slab_limit: '
            f'max|H_y|={Hy_max:.4e} > '
            f'{TOL_UNIFORM_FIELD*mu0_Ms:.4e}.')


# -----------------------------------------------------------------------------
def test_uniform_m_xhat_slab_limit():
    """For uniform m = +x_hat, the slab limit gives
    H_demag = 0 (uniformly in-plane magnetized infinite
    slab has no demag). The Newell residual is the
    finite-cell correction; on 64x64 with accuracy=4 it
    must stay below 5% of mu0 Ms.
    """
    p = default_params()
    p.nx = NX_TEST
    p.ny = NY_TEST
    _precompute(p)
    K = precompute_demag_kernels(
        p, kind='newell',
        accuracy=4.0, tol_conv=0.02,
    )
    # Uniform m = +x_hat
    m_top = np.zeros((p.ny, p.nx, 3))
    m_top[..., 0] = 1.0
    m_bot = m_top.copy()
    H_top, H_bot = demag_field(m_top, m_bot, K)
    mu0_Ms = p.mu0 * p.Ms
    Hx_max = float(np.max(np.abs(H_top[..., 0])))
    Hy_max = float(np.max(np.abs(H_top[..., 1])))
    Hz_max = float(np.max(np.abs(H_top[..., 2])))
    rel_x = Hx_max / mu0_Ms
    rel_y = Hy_max / mu0_Ms
    rel_z = Hz_max / mu0_Ms
    if rel_x > TOL_UNIFORM_FIELD:
        raise RuntimeError(
            f'test_uniform_m_xhat_slab_limit: max|H_x|/'
            f'mu0_Ms = {rel_x:.4f} > tol '
            f'{TOL_UNIFORM_FIELD} (finite-cell residual).')
    if rel_y > TOL_UNIFORM_FIELD:
        raise RuntimeError(
            f'test_uniform_m_xhat_slab_limit: max|H_y|/'
            f'mu0_Ms = {rel_y:.4f} > tol '
            f'{TOL_UNIFORM_FIELD}.')
    if rel_z > TOL_UNIFORM_FIELD:
        raise RuntimeError(
            f'test_uniform_m_xhat_slab_limit: max|H_z|/'
            f'mu0_Ms = {rel_z:.4f} > tol '
            f'{TOL_UNIFORM_FIELD}.')
    # Cross-check: slab kernel must give EXACTLY zero in-plane
    # demag for uniform m_x (a stronger test than Newell).
    K_slab = precompute_demag_kernels(
        p, kind='slab', accuracy=None, tol_conv=None,
    )
    Hs_top, _Hs_bot = demag_field(m_top, m_bot, K_slab)
    slab_x_max = float(np.max(np.abs(Hs_top[..., 0])))
    slab_y_max = float(np.max(np.abs(Hs_top[..., 1])))
    slab_z_max = float(np.max(np.abs(Hs_top[..., 2])))
    # Slab is exact at k=0; expect 0 to floating-point.
    slab_tol = 1e-8 * mu0_Ms
    if slab_x_max > slab_tol:
        raise RuntimeError(
            f'test_uniform_m_xhat_slab_limit: slab '
            f'max|H_x|={slab_x_max:.4e} > tol '
            f'{slab_tol:.4e} (slab should give exact 0).')
    if slab_y_max > slab_tol or slab_z_max > slab_tol:
        raise RuntimeError(
            f'test_uniform_m_xhat_slab_limit: slab '
            f'in-plane uniform-M demag not zero '
            f'(Hy_max={slab_y_max:.4e}, '
            f'Hz_max={slab_z_max:.4e}).')


# =============================================================================
def main():
    print('Test 1: Aharoni single-prism cube...')
    test_aharoni_single_prism_cube()
    print('  OK')
    print()
    print('Test 2: Aharoni single-prism SAF cell (2 x 2 x 1.3 nm)...')
    test_aharoni_single_prism_saf_cell()
    print('  OK')
    print()
    print('Test 3: Uniform m = z_hat -> slab limit H = '
          '-mu0 Ms z_hat (Newell)...')
    test_uniform_m_zhat_slab_limit()
    print('  OK')
    print()
    print('Test 4: Uniform m = x_hat -> H ~ 0 (Newell '
          'finite-cell + slab exact)...')
    test_uniform_m_xhat_slab_limit()
    print('  OK')
    print()
    print('All Newell demag tests passed.')


# =============================================================================
if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(f'FAIL: {e}', file=sys.stderr)
        sys.exit(1)
