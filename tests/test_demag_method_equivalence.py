"""Equivalence check: closed-form vs Gauss-Legendre quadrature
for the Newell demag tensor.

The closed form (`_newell_tensor_closed`, OOMMF transcription)
and the mumax3-style quadrature (`_compute_one_pair_tensor`)
must agree per cell-pair as the quadrature density grows. We
compare the two **per-pair tensor functions directly** at a
set of representative offsets with a bounded quadrature
density (n_density per axis fixed, never accuracy-scaled), so
the test stays fast and never explodes the near-cell point
count. The self-cell is checked against the Aharoni closed
form, which both kernel builds override at (0, 0).

A separate static-demag check confirms both module `_METHOD`
settings produce the same uniform-m demag field through the
full FFT pipeline at a sane accuracy.

Functions
---------
test_pair_tensor_far_cells_match
    Closed-form vs quadrature per-pair tensor at well-
    separated offsets (strict tol).
test_pair_tensor_near_cells_converge
    Closed-form vs quadrature per-pair tensor at touching /
    near cells, quadrature density increased until it
    converges to the closed form.
test_self_cell_matches_aharoni
    Closed-form self-cell tensor equals the Aharoni demag
    factors.
test_static_demag_uniform_mz_freebc
    Uniform +z demag field equal for both `_METHOD` settings
    through the full kernel + FFT pipeline (sane accuracy).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
from types import SimpleNamespace
# Third-party
import numpy as np
import pytest
# Local
from src.simulator import demag_newell
from src.simulator.demag_newell import (
    _aharoni_demag_factor,
    _compute_one_pair_tensor,
    _newell_tensor_closed,
)
from src.simulator.demag import (
    precompute_demag_kernels, demag_field)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
# Cell geometry used for the per-pair comparisons (SP4-like
# thin-film cell: 5 x 5 x 3 nm).
_DX = 5.0e-9
_DY = 5.0e-9
_DZ = 3.0e-9


def _quad_tensor(X, Y, Z, n_per_axis):
    """Quadrature per-pair tensor with a fixed (not accuracy-
    scaled) node count, sign-flipped to the depolarizing-
    positive convention used by the closed form and slab
    kernel."""
    t = _compute_one_pair_tensor(
        X=X, Y=Y, Z=Z, cellsize=(_DX, _DY, _DZ),
        n_density=(n_per_axis, n_per_axis, n_per_axis))
    return -t


def _max_rel(a, b):
    """Max abs difference scaled by max |closed|."""
    scale = float(np.max(np.abs(a))) + 1e-30
    return float(np.max(np.abs(a - b))) / scale


# -----------------------------------------------------------------------------
def test_pair_tensor_far_cells_match():
    """At well-separated offsets the quadrature converges fast;
    closed-form and quadrature (12 pts/axis) agree to 1e-4."""
    tol = 1.0e-4
    offsets = [
        (2 * _DX, 0.0, 0.0),
        (3 * _DX, 2 * _DY, 0.0),
        (5 * _DX, 0.0, 0.0),
        (10 * _DX, 0.0, 0.0),
        (0.0, 0.0, _DZ + 100.0e-9),
        (2 * _DX, 2 * _DY, _DZ + 100.0e-9),
    ]
    for (X, Y, Z) in offsets:
        closed = _newell_tensor_closed(X, Y, Z, _DX, _DY, _DZ)
        quad = _quad_tensor(X, Y, Z, n_per_axis=12)
        rel = _max_rel(closed, quad)
        assert rel < tol, (
            f'offset ({X:.2e}, {Y:.2e}, {Z:.2e}): per-pair '
            f'tensor relative diff {rel:.3e} exceeds tol '
            f'{tol:.3e}')


# -----------------------------------------------------------------------------
def test_pair_tensor_near_cells_converge():
    """At touching / near offsets the quadrature is harder; we
    increase the node count and require the residual to
    decrease monotonically toward the closed form, which is
    the exact reference (it matches Aharoni to 1e-12 in
    `test_self_cell_matches_aharoni`). The absolute floor is
    set to 3e-2: at a touching cell (edge-to-edge distance 0)
    the 1/r^3 near-singularity caps Gauss-Legendre accuracy
    at a few percent even at n = 24; the monotone-decrease
    check is the real evidence the two formulas describe the
    same integral."""
    tol = 3.0e-2
    # Touching neighbours along each axis (hardest cases short
    # of the self-cell, which is Aharoni-overridden).
    offsets = [
        (_DX, 0.0, 0.0),
        (0.0, _DY, 0.0),
        (_DX, _DY, 0.0),
    ]
    for (X, Y, Z) in offsets:
        closed = _newell_tensor_closed(X, Y, Z, _DX, _DY, _DZ)
        prev_rel = None
        rel = None
        for n in (8, 16, 24):
            quad = _quad_tensor(X, Y, Z, n_per_axis=n)
            rel = _max_rel(closed, quad)
            if prev_rel is not None:
                # Residual must not grow as density increases.
                assert rel <= prev_rel + 1.0e-6, (
                    f'offset ({X:.2e}, {Y:.2e}, {Z:.2e}): '
                    f'quadrature residual grew from '
                    f'{prev_rel:.3e} to {rel:.3e} when n '
                    f'increased to {n}')
            prev_rel = rel
        assert rel < tol, (
            f'offset ({X:.2e}, {Y:.2e}, {Z:.2e}): per-pair '
            f'tensor relative diff {rel:.3e} (n=24) exceeds '
            f'tol {tol:.3e}')


# -----------------------------------------------------------------------------
def test_self_cell_matches_aharoni():
    """Closed-form self-cell diagonals match the Aharoni demag
    factors to 1e-12; off-diagonals vanish."""
    tol = 1.0e-12
    closed = _newell_tensor_closed(0.0, 0.0, 0.0, _DX, _DY, _DZ)
    # Aharoni cyclic-permutation convention used in
    # _build_layer_pair_kernel.
    nx_ref = _aharoni_demag_factor(_DY, _DZ, _DX)
    ny_ref = _aharoni_demag_factor(_DX, _DZ, _DY)
    nz_ref = _aharoni_demag_factor(_DX, _DY, _DZ)
    assert abs(closed[0, 0] - nx_ref) < tol
    assert abs(closed[1, 1] - ny_ref) < tol
    assert abs(closed[2, 2] - nz_ref) < tol
    assert abs(closed[0, 1]) < tol
    assert abs(closed[0, 2]) < tol
    assert abs(closed[1, 2]) < tol


# -----------------------------------------------------------------------------
def test_static_demag_uniform_mz_freebc():
    """Free-BC demag on uniform +z m: interior <H_z> equal for
    both `_METHOD` settings through the full FFT pipeline at a
    sane accuracy (=6, near-cell points bounded)."""
    nx, ny = 100, 25
    mu0 = 4.0e-7 * np.pi
    Ms = 8.0e5
    mu0_Ms = mu0 * Ms
    p = SimpleNamespace(
        nx=nx, ny=ny, a=5.0e-9, t_Co=3.0e-9,
        d_Ru=100.0e-9, Ms=Ms, mu0=mu0)
    tol = 2.0e-3
    saved_method = demag_newell._METHOD
    interiors = {}
    try:
        for method in ('closed', 'quadrature'):
            demag_newell._METHOD = method
            kernels = precompute_demag_kernels(
                p, kind='newell_freebc',
                accuracy=6.0, tol_conv=0.10)
            m_top = np.zeros((ny, nx, 3))
            m_top[..., 2] = 1.0
            m_bot = np.zeros_like(m_top)
            H, _ = demag_field(m_top, m_bot, kernels)
            interiors[method] = float(
                (H[..., 2] / mu0_Ms)[3:-3, 5:-5].mean())
    finally:
        demag_newell._METHOD = saved_method
    diff = abs(interiors['closed'] - interiors['quadrature'])
    assert diff < tol, (
        f'interior <H_z>/(mu0 Ms): closed='
        f'{interiors["closed"]:+.4f}, quadrature='
        f'{interiors["quadrature"]:+.4f}, diff {diff:.3e} '
        f'exceeds tol {tol:.3e}')


# =============================================================================
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
