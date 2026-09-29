"""Shared helpers for single-FM-layer validation benchmarks.

These thin wrappers reuse the existing simulator infrastructure
(`rk4_step` + `rhs_local_keff` + `default_params` +
`skyrmion_profile`) so the benchmark code path matches the
production code path exactly. Single-FM physics is obtained by
setting `p.H_RKKY = 0.0`, which decouples the two SAF layers and
makes the top layer evolve as an isolated 2D ferromagnet.

The K_eff convention from the production simulator (uniform-slab
demag folded into K) is used throughout: callers pass the
desired `K_eff` and `make_single_fm_params` back-converts it to
the bare `K_top, K_bot` that `_precompute` expects so that the
precomputed `C_anis_*` prefactors land on `2 K_eff / Ms`. This
keeps the benchmark numerics bit-equivalent to the production
path while letting tests reason in terms of `K_eff`.

Functions
---------
make_single_fm_params
    Build a parameter namespace for a single-FM benchmark by
    overriding `default_params()` with the supplied material
    constants and clearing the RKKY coupling.
relax_single_fm
    Over-damped descent to the local energy minimum, using
    `rk4_step` + `rhs_local_keff` with `alpha` temporarily
    overridden and the pulse forced to zero current.
integrate_single_fm
    Real-time LLG-Slonczewski integration with `rk4_step` +
    `rhs_local_keff`. Returns a time series of top-layer
    snapshots.
plot_ic_2d
    Save a PNG of m_z (heatmap) + (m_x, m_y) quiver overlay.
radial_profile
    Azimuthally average a 2D scalar field around the lattice
    centre.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.simulator.fields import effective_field
from src.simulator.integrator import rhs_local_keff, rk4_step
from src.simulator.parameters import _precompute, default_params
from src.simulator.pulses import ConstantPulse

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def make_single_fm_params(A_ex, D, K_eff, Ms, alpha, gamma,
                          H_ext, nx, ny, a, dt,
                          J_current=0.0, p_hat=None):
    """Build a `default_params()` namespace tuned for a single-FM
    benchmark.

    Parameters
    ----------
    A_ex : float
        Exchange stiffness (J/m).
    D : float
        Interfacial DMI constant (J/m^2).
    K_eff : float
        Target effective anisotropy (J/m^3). The bare `K_top`
        and `K_bot` are set to `K_eff + 0.5 mu_0 Ms^2` so that
        the precomputed `C_anis_*` prefactors equal
        `2 K_eff / Ms`, matching the simulator's K_eff
        convention.
    Ms : float
        Saturation magnetisation (A/m).
    alpha : float
        Gilbert damping for LLG.
    gamma : float
        Gyromagnetic ratio (rad / (s T)).
    H_ext : numpy.ndarray(1d)
        External field (Tesla), shape (3,).
    nx, ny : int
        Lattice size.
    a : float
        Lattice constant (m).
    dt : float
        Time step (s).
    J_current : float, default=0.0
        Steady current density for SOT (A/m^2). The default
        zero disables the SOT torque without changing the
        DL_SOT / FL_SOT prefactors.
    p_hat : {numpy.ndarray(1d), None}, default=None
        SOT polarisation unit vector, shape (3,). None keeps
        the `default_params()` value.

    Returns
    -------
    p : SimpleNamespace
        Parameter namespace with `_precompute` already called.
    """
    # Loud rejection of invalid Ms or alpha so the back-conversion
    # below cannot silently produce wrong K_top values.
    if not (Ms > 0.0):
        raise RuntimeError(
            f'make_single_fm_params: Ms must be positive, got {Ms!r}.')
    if not (alpha > 0.0):
        raise RuntimeError(
            f'make_single_fm_params: alpha must be positive, got '
            f'{alpha!r}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    p = default_params()
    # Material overrides.
    p.A_ex = float(A_ex)
    p.D = float(D)
    p.Ms = float(Ms)
    p.alpha = float(alpha)
    p.gamma = float(gamma)
    # Back-convert K_eff to bare K so _precompute lands on the
    # right C_anis_* prefactors.
    half_mu0_Ms2 = 0.5 * p.mu0 * p.Ms * p.Ms
    p.K_top = float(K_eff) + half_mu0_Ms2
    p.K_bot = float(K_eff) + half_mu0_Ms2
    # External and inter-layer fields.
    p.H_ext = np.asarray(H_ext, dtype=float).reshape(3)
    # Single-FM physics: the two SAF layers must be decoupled.
    p.H_RKKY = 0.0
    # Lattice + time step.
    p.nx = int(nx)
    p.ny = int(ny)
    p.a = float(a)
    p.dt = float(dt)
    # Drive pulse and SOT polarisation (no current by default).
    p.pulse = ConstantPulse(float(J_current))
    p.J_current = float(J_current)
    if p_hat is not None:
        p.p_hat = np.asarray(p_hat, dtype=float).reshape(3)
    # Topological spin Hall channel off (only the SOT terms are
    # needed for the validation benchmarks).
    p.lambda_sq = 0.0
    # Refresh derived prefactors (C_ex, C_dmi, C_anis_top/bot,
    # gamma_p, H_DL, H_FL, ...).
    _precompute(p)
    return p


# -----------------------------------------------------------------------------
def _uniform_up_like(m_top):
    """Return a uniform m_z = +1 array shaped like `m_top`.

    Used as the dummy bottom layer in the single-FM benchmarks:
    with `p.H_RKKY = 0` this layer evolves independently of the
    top one, and starting it in its energy minimum keeps the
    side-channel compute negligible.
    """
    m_bot = np.zeros_like(m_top)
    m_bot[..., 2] = 1.0
    return m_bot


# -----------------------------------------------------------------------------
def _tau_max(m, m_other, p):
    """Return max |m x (m x H_eff)| on the top layer (Tesla)."""
    H = effective_field(
        m, m_other,
        p.C_ex, p.C_dmi, p.C_anis_top,
        p.H_ext, p.H_RKKY)
    mxH = np.cross(m, H)
    mxmxH = np.cross(m, mxH)
    return float(np.max(np.linalg.norm(mxmxH, axis=-1)))


def relax_single_fm(m_top, p, alpha_relax=1.0, max_steps=200_000,
                    tol_torque=1.0e-5, check_every=1_000):
    """Over-damped descent of one FM layer to the local minimum.

    Reuses `rk4_step` + `rhs_local_keff`. `p.alpha`, `p.gamma_p`
    and `p.pulse` are temporarily overridden to make the LLG
    behave as an over-damped quench (alpha=alpha_relax,
    SOT-free) and restored on exit, regardless of whether
    convergence was reached.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Initial top-layer magnetisation, shape (ny, nx, 3).
    p : SimpleNamespace
        Simulator parameters (use `make_single_fm_params`).
    alpha_relax : float, default=1.0
        Damping override for the descent.
    max_steps : int, default=200000
        Safety cutoff on RK4 steps.
    tol_torque : float, default=1e-5
        Convergence threshold on max |m x (m x H_eff)| (Tesla).
    check_every : int, default=1000
        Stride at which the torque norm is sampled.

    Returns
    -------
    m_top : numpy.ndarray(3d)
        Final magnetisation.
    tau_max : float
        Final torque norm (Tesla).
    n_done : int
        Number of RK4 steps executed.
    """
    saved = {'alpha': p.alpha, 'gamma_p': p.gamma_p,
             'pulse': p.pulse, 'lambda_sq': p.lambda_sq}
    try:
        p.alpha = float(alpha_relax)
        p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
        p.pulse = ConstantPulse(0.0)
        p.lambda_sq = 0.0
        m_bot = _uniform_up_like(m_top)
        t = 0.0
        n_done = int(max_steps)
        tau_max = float('inf')
        for step in range(int(max_steps)):
            m_top, m_bot = rk4_step(
                rhs_local_keff, m_top, m_bot, t, p.dt, p)
            t += p.dt
            if (step + 1) % int(check_every) == 0:
                tau_max = _tau_max(m_top, m_bot, p)
                if tau_max < float(tol_torque):
                    n_done = step + 1
                    break
                n_done = step + 1
    finally:
        p.alpha = saved['alpha']
        p.gamma_p = saved['gamma_p']
        p.pulse = saved['pulse']
        p.lambda_sq = saved['lambda_sq']
    return m_top, tau_max, n_done


# -----------------------------------------------------------------------------
def integrate_single_fm(m_top, p, n_steps, sample_every, t0=0.0):
    """Real-time LLG-Slonczewski integration of one FM layer.

    Reuses `rk4_step` + `rhs_local_keff`. The pulse / SOT
    polarisation already set on `p` (via
    `make_single_fm_params`) drives the dynamics.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Initial top-layer magnetisation, shape (ny, nx, 3).
    p : SimpleNamespace
        Simulator parameters.
    n_steps : int
        Number of RK4 steps.
    sample_every : int
        Stride at which to record `m_top` snapshots.
    t0 : float, default=0.0
        Starting time (s).

    Returns
    -------
    times : numpy.ndarray(1d)
        Sampled times in seconds, length `n_steps // sample_every + 1`.
    m_trace : numpy.ndarray(4d)
        Sampled magnetisation snapshots, shape (n_samples, ny, nx, 3).
    """
    n_steps = int(n_steps)
    sample_every = int(max(1, sample_every))
    n_samples = n_steps // sample_every + 1
    times = np.empty(n_samples)
    m_trace = np.empty((n_samples, *m_top.shape))
    times[0] = float(t0)
    m_trace[0] = m_top
    m_bot = _uniform_up_like(m_top)
    t = float(t0)
    idx = 1
    for step in range(1, n_steps + 1):
        m_top, m_bot = rk4_step(
            rhs_local_keff, m_top, m_bot, t, p.dt, p)
        t += p.dt
        if step % sample_every == 0 and idx < n_samples:
            times[idx] = t
            m_trace[idx] = m_top
            idx += 1
    return times[:idx], m_trace[:idx]


# -----------------------------------------------------------------------------
def plot_ic_2d(m, a, out_path, title=None, q_stride=None):
    """Save a PNG snapshot of a 2D magnetisation configuration.

    Heatmap of m_z (RdBu, range [-1, +1]) with an in-plane
    (m_x, m_y) quiver overlay on a sub-sampled grid.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Shape (ny, nx, 3); unit magnetisation.
    a : float
        Lattice spacing (m), used to label the axes in nm.
    out_path : str
        Destination PNG path. Parent directory is created if
        missing.
    title : {str, None}, default=None
        Optional plot title.
    q_stride : {int, None}, default=None
        Quiver subsample stride. None auto-picks
        `max(1, min(nx, ny) // 16)`.
    """
    ny, nx, _ = m.shape
    if q_stride is None:
        q_stride = max(1, min(nx, ny) // 16)
    # Axes span the lattice centred on the origin, in nm.
    extent_nm = [
        -0.5 * (nx - 1) * a * 1e9,
        +0.5 * (nx - 1) * a * 1e9,
        -0.5 * (ny - 1) * a * 1e9,
        +0.5 * (ny - 1) * a * 1e9,
    ]
    fig, ax = plt.subplots(figsize=(5.0, 4.4), dpi=160)
    # m_z as a fixed [-1, 1] diverging map; origin='lower' so the
    # image y-axis matches the array row order.
    im = ax.imshow(
        m[..., 2], origin='lower', cmap='RdBu',
        vmin=-1.0, vmax=+1.0, extent=extent_nm,
        interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.04)
    cbar.set_label(r'$m_z$', rotation=0, labelpad=8)
    # Subsampled in-plane (m_x, m_y) quiver overlay.
    yy_idx, xx_idx = np.meshgrid(
        np.arange(ny)[::q_stride],
        np.arange(nx)[::q_stride],
        indexing='ij')
    xx_nm = (xx_idx - 0.5 * (nx - 1)) * a * 1e9
    yy_nm = (yy_idx - 0.5 * (ny - 1)) * a * 1e9
    mx_q = m[::q_stride, ::q_stride, 0]
    my_q = m[::q_stride, ::q_stride, 1]
    ax.quiver(
        xx_nm, yy_nm, mx_q, my_q,
        color='k', scale=30, width=0.003,
        pivot='middle')
    ax.set_xlabel(r'$x$ (nm)')
    ax.set_ylabel(r'$y$ (nm)')
    ax.set_aspect('equal')
    if title:
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


# -----------------------------------------------------------------------------
def radial_profile(field2d, a, n_bins):
    """Azimuthal average of `field2d` on `n_bins` radial bins.

    The lattice centre is the geometric centre of the array.
    Returns (r_bin_centres, mean_per_bin); empty bins yield NaN.
    """
    ny, nx = field2d.shape[:2]
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
        indexing='xy')
    # Per-cell radius from the geometric centre, flattened to pair
    # with the flattened field values.
    x = (jj - 0.5 * (nx - 1)) * a
    y = (ii - 0.5 * (ny - 1)) * a
    r = np.sqrt(x * x + y * y).ravel()
    vals = field2d.ravel()
    # Cap r_max at the inscribed-circle radius so no bin straddles
    # the array corners (which would undersample azimuthally).
    r_max = float(0.5 * min(nx, ny) * a)
    edges = np.linspace(0.0, r_max, int(n_bins) + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    # Bin mean via weighted-sum / count histograms over the radii.
    sums, _ = np.histogram(r, bins=edges, weights=vals)
    counts, _ = np.histogram(r, bins=edges)
    # Per-bin mean; empty bins are explicitly NaN. The masked
    # divide avoids the silent `np.maximum(counts, 1)` sentinel
    # while still leaving NaN for downstream filtering.
    mean = np.full_like(sums, np.nan, dtype=float)
    nonempty = counts > 0
    mean[nonempty] = sums[nonempty] / counts[nonempty]
    return centres, mean
