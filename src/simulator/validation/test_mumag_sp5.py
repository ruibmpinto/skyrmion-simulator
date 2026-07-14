"""muMAG Standard Problem #5 benchmark (STT-driven vortex).

NIST muMAG Standard Problem #5 (Najafi et al., J. Appl. Phys.
105, 113914 (2009); https://www.ctcms.nist.gov/~rdm/std5/
spec5.xhtml). A Permalloy 100 nm x 100 nm x 10 nm square
holds a magnetic vortex; a spatially uniform in-plane spin-
polarised current drives the vortex core to a new steady-
state position. The Zhang-Li adiabatic + non-adiabatic spin-
transfer torque (STT) is the driving mechanism.

Single layer + local Zhang-Li STT
---------------------------------
SP5 is a single ferromagnetic film. As in the SP4 test, only
`m_top` is evolved with the self-layer demag (the bottom
layer is held at zero so the inter-layer term vanishes). The
Zhang-Li STT term is implemented LOCALLY in this test's RHS
(the production `llgs_rhs` carries only spin-orbit and
topological-Hall torques, not the current-convection Zhang-Li
term), so no shared `src/simulator` code is modified.

LLGS with Zhang-Li STT (explicit form, Thiaville 2005)
------------------------------------------------------
    dm/dt = 1/(1+a^2) [ -g (m x H) - g a m x (m x H)
            - (1 + a b)(u . grad) m
            + (b - a) m x ((u . grad) m) ]
with u = u_T x_hat the spin-drift velocity, a = alpha,
b = beta = xi, g = gamma. The field/damping part is taken
from `llgs_rhs` (its gamma_p = gamma/(1+a^2) prefactor matches
the 1/(1+a^2) factor above); the two STT terms are added here.

Canonical specification
-----------------------
- Geometry:  100 nm x 100 nm x 10 nm, free boundaries.
- Material:  A = 1.3e-11 J/m, M_s = 8.0e5 A/m, K = 0,
             alpha = 0.1.
- Gyromag.:  gamma_0 = 2.211e5 m/(A s) -> gamma = 1.760e11
             rad/(s T) via gamma_0 / mu_0.
- Drive:     P J = 1e12 A/m^2 along +x. The spec tabulates
             the spin-drift speed u_T for each xi:
               xi = 0    -> u_T = -72.35 m/s
               xi = 0.05 -> u_T = -72.17 m/s  (this test)
               xi = 0.1  -> u_T = -71.64 m/s
               xi = 0.5  -> u_T = -57.88 m/s
- Initial:   vortex m ~ [-y, x, R] / |.|, R = 10 nm, centred.
- Output:    vortex-core trajectory; <m_x>(t), <m_y>(t).

Acceptance (scalar: Najafi 2009 steady-state core position)
-----------------------------------------------------------
NIST publishes only the time-less M_x-M_y spiral *image*
(no downloadable time-series), so a full-trajectory RMS gate
is not possible from the bundled references and OOMMF/mumax3
is not available locally to regenerate one. Instead we gate
on the primary-source scalar that all contributed solvers
(OOMMF, nmag, M3S, LLG, micromagus) agreed on:

    Najafi et al., J. Appl. Phys. 105, 113914 (2009), for
    xi = 0.05: the vortex core settles (after damped
    gyration) at
        Dx = x - x0 = -1.2 nm,   Dy = y - y0 = -14.7 nm,
    where (x0, y0) is the dot centre and the core is the
    m_z extremum.

The test drives the vortex, tracks the core via the m_z peak
with parabolic sub-cell interpolation, takes the steady-state
position as the time-average of the core over the late part
of the drive (the gyration-orbit centre), and compares
(Dx, Dy) to the Najafi reference within `core_tol`. The full
<m_x>(t)/<m_y>(t) spiral and the core trajectory are also
saved for visual comparison against the NIST spiral plot.

Functions
---------
main
    Build the vortex IC, drive with Zhang-Li STT, track the
    core, compare its steady-state displacement to Najafi
    2009.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.simulator.demag import precompute_demag_kernels
from src.simulator.fields import effective_field_demag_pair
from src.simulator.integrator import (
    llgs_rhs, rk4_step_single, zhang_li_torque)
from src.simulator.lattice import rect_mask
from src.simulator.parameters import _precompute, default_params
from src.simulator.pulses import ConstantPulse
from src.simulator.validation._helpers import plot_ic_2d

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
# Najafi 2009 steady-state core displacement (xi = 0.05).
NAJAFI_DX = -1.2e-9    # m
NAJAFI_DY = -14.7e-9   # m


def _make_sp5_params(alpha, nx, ny, a, dt):
    """Build a Permalloy SP5 parameter namespace."""
    p = default_params()
    p.A_ex = 1.3e-11           # J/m
    p.D = 0.0                  # no DMI
    p.Ms = 8.0e5               # A/m
    p.alpha = float(alpha)
    p.gamma = 1.760e11         # rad/(s T) = gamma_0 / mu_0
    p.K_top = 0.0              # K = 0 (bare convention)
    p.K_bot = 0.0
    p.H_RKKY = 0.0
    p.t_Co = 10.0e-9           # film thickness
    p.d_Ru = 100.0e-9          # irrelevant (single layer)
    p.nx = int(nx)
    p.ny = int(ny)
    p.a = float(a)
    p.dt = float(dt)
    p.H_ext = np.array([0.0, 0.0, 0.0])
    p.pulse = ConstantPulse(0.0)   # no SOT; STT added locally
    p.J_current = 0.0
    p.lambda_sq = 0.0
    _precompute(p)
    return p


def _single_layer_field(m_top, p, kernels, mask):
    """Single-layer effective field (exchange + self-demag;
    K = 0, D = 0, H_ext = 0). Bottom layer held at zero so
    the inter-layer demag term vanishes."""
    m_zero = np.zeros_like(m_top)
    H_top, _ = effective_field_demag_pair(
        m_top, m_zero, p, kernels, mask=mask)
    return H_top


def _zhang_li_rhs(m_top, p, kernels, mask, u_T, beta):
    """dm/dt for a single layer with Zhang-Li STT.

    Field/damping part from `llgs_rhs` (gamma_p = gamma /
    (1 + alpha^2)); the adiabatic and non-adiabatic STT terms
    are added with the same 1/(1+alpha^2) prefactor.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Magnetisation, shape (ny, nx, 3).
    p : SimpleNamespace
        Parameter namespace.
    kernels : dict
        Free-BC demag kernels.
    mask : numpy.ndarray(2d)
        Boolean (ny, nx) magnet mask.
    u_T : float
        Spin-drift speed along +x (m/s).
    beta : float
        Non-adiabaticity xi.

    Returns
    -------
    dmdt : numpy.ndarray(3d)
        Time derivative, shape (ny, nx, 3).
    """
    H_top = _single_layer_field(m_top, p, kernels, mask)
    # Precession + damping from the PRODUCTION llgs_rhs (SOT/TSH
    # branches inert: p.pulse = 0, p.lambda_sq = 0).
    dmdt = llgs_rhs(m_top, H_top, p, t=0.0)
    # Zhang-Li STT from the PRODUCTION integrator.zhang_li_torque
    # (carries the same 1/(1+alpha^2) prefactor as llgs_rhs).
    # Masked so the inert dummy +z cells outside the magnet
    # contribute no spurious convection.
    stt = zhang_li_torque(m_top, p.a, u_T, beta, p.alpha)
    dmdt = dmdt + stt * mask[..., np.newaxis]
    return dmdt


def _rk4_zhang_li(m_top, p, kernels, mask, u_T, beta, dt):
    """One single-layer Zhang-Li RK4 step via the PRODUCTION
    stepper `integrator.rk4_step_single`. Only the RHS wiring
    (production field + production Zhang-Li torque) is local."""
    def rhs(m, p_, tt):
        return _zhang_li_rhs(m, p_, kernels, mask, u_T, beta)
    return rk4_step_single(rhs, m_top, t=0.0, dt=dt, p=p)


def _relax_vortex(m_top, p, kernels, mask, max_steps,
                  tol_torque, check_every):
    """Over-damped relaxation of the vortex IC at zero current
    (no STT). Returns (m_top, tau_max, n_done)."""
    saved = {'alpha': p.alpha, 'gamma_p': p.gamma_p}
    try:
        p.alpha = 1.0
        p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
        t = 0.0
        tau_max = float('inf')
        n_done = int(max_steps)
        for step in range(int(max_steps)):
            # Zero-current step: u_T = 0 -> pure LLG relax.
            m_top = _rk4_zhang_li(
                m_top, p, kernels, mask, u_T=0.0, beta=0.0,
                dt=p.dt)
            t += p.dt
            if (step + 1) % int(check_every) == 0:
                H_top = _single_layer_field(
                    m_top, p, kernels, mask)
                mxH = np.cross(m_top, H_top)
                mxmxH = np.cross(m_top, mxH)
                tau_max = float(np.max(np.linalg.norm(
                    mxmxH * mask[..., np.newaxis], axis=-1)))
                n_done = step + 1
                if tau_max < tol_torque:
                    break
    finally:
        p.alpha = saved['alpha']
        p.gamma_p = saved['gamma_p']
    return m_top, tau_max, n_done


def _vortex_core(m_top, a, mask, cx0, cy0):
    """Vortex-core displacement (Dx, Dy) from the dot centre.

    The core is the out-of-plane (m_z) extremum. Its location
    is refined to sub-cell precision by a 1D parabolic fit of
    m_z along x and along y through the peak cell. Returns the
    displacement relative to (cx0, cy0) in metres.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Magnetisation, shape (ny, nx, 3).
    a : float
        Cell size (m).
    mask : numpy.ndarray(2d)
        Boolean (ny, nx) magnet mask (peak searched inside).
    cx0, cy0 : float
        Dot-centre coordinates (m).

    Returns
    -------
    dx, dy : float
        Core displacement from the centre (m).
    """
    mz = m_top[..., 2].copy()
    # Restrict the peak search to the magnet (outside is dummy
    # +z with m_z = 1, which would otherwise win the argmax).
    mz_masked = np.where(mask, mz, -np.inf)
    iy, ix = np.unravel_index(
        int(np.argmax(mz_masked)), mz.shape)
    ny, nx = mz.shape

    def _parabolic_shift(fm, f0, fp):
        # Sub-cell offset of the vertex of the parabola through
        # (-1, fm), (0, f0), (+1, fp), in cell units, clamped
        # to [-0.5, 0.5].
        denom = fm - 2.0 * f0 + fp
        if denom == 0.0:
            return 0.0
        s = 0.5 * (fm - fp) / denom
        return max(-0.5, min(0.5, s))

    # x-parabola (need interior neighbours; else no shift).
    if 0 < ix < nx - 1:
        sx = _parabolic_shift(mz[iy, ix - 1], mz[iy, ix],
                              mz[iy, ix + 1])
    else:
        sx = 0.0
    if 0 < iy < ny - 1:
        sy = _parabolic_shift(mz[iy - 1, ix], mz[iy, ix],
                              mz[iy + 1, ix])
    else:
        sy = 0.0
    x_core = (ix + sx) * a
    y_core = (iy + sy) * a
    return x_core - cx0, y_core - cy0


def main():
    """Run the NIST muMAG Standard Problem 5 (STT) validation.

    Relaxes a magnetic vortex in a single-layer permalloy square
    (100 x 100 x 10 nm), drives it with spin-transfer torque
    (xi = 0.05), tracks the vortex core trajectory, and writes IC,
    relaxed-state and core-trajectory figures plus the trajectory
    data. Passes when the steady-state core displacement matches
    the Najafi 2009 reference (-1.2, -14.7) nm within tolerance.

    Returns
    -------
    passed : bool
        True if the core displacement is within `core_tol`.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Geometry (NIST SP5).
    Lx_mag = 100.0e-9
    Ly_mag = 100.0e-9
    # Cell <= l_ex ~ 5.7 nm; 2.5 nm resolves the vortex core.
    a = 2.5e-9
    pad_factor = 1.4
    nx = int(round(pad_factor * Lx_mag / a))
    ny = int(round(pad_factor * Ly_mag / a))
    dt = 5.0e-14
    # STT drive: xi = 0.05 set.
    xi = 0.05
    u_T = -72.17               # m/s (spec, xi = 0.05)
    # Vortex IC core scale.
    R_core = 10.0e-9
    # Relaxation (vortex equilibrium before the current is on).
    max_steps_relax = 120_000
    tol_torque_relax = 1.0e-4
    check_every = 1_000
    # Drive: NIST estimates equilibrium ~ 14 ns. Sample at
    # 20 ps for a smooth spiral; drive_time configurable.
    drive_time = 8.0e-9
    sample_dt = 20.0e-12
    drive_n_steps = int(round(drive_time / dt))
    drive_sample_every = int(round(sample_dt / dt))
    out_dir = 'docs/figures/validation'
    ic_png = os.path.join(out_dir, 'sp5_initial.png')
    traj_txt = os.path.join(out_dir, 'sp5_trajectory.txt')
    traj_png = os.path.join(out_dir, 'sp5_trajectory.png')
    # Steady-state core position = time-average of the core
    # over the late part of the drive (the damped-gyration
    # orbit centre). Average over the last `tail_frac` of the
    # samples.
    tail_frac = 0.5
    # Acceptance: Euclidean distance of the steady-state core
    # displacement from the Najafi 2009 reference (-1.2, -14.7)
    # nm. Tolerance covers the 2.5 nm grid + sub-cell interp +
    # residual gyration; cross-solver spread in Najafi is ~1 nm.
    core_tol = 3.0e-9
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('test_mumag_sp5 (NIST muMAG SP#5, STT vortex, '
          'single layer, xi = 0.05):')
    print(
        f'  geometry: magnet {Lx_mag*1e9:.0f} x {Ly_mag*1e9:.0f} '
        f'x 10 nm; lattice {nx}x{ny} at a = {a*1e9:.1f} nm.')
    print(f'  drive: u_T = {u_T:.2f} m/s, xi = {xi}, '
          f'drive_time = {drive_time*1e9:.1f} ns.')
    p = _make_sp5_params(alpha=0.1, nx=nx, ny=ny, a=a, dt=dt)
    mask = rect_mask(nx=nx, ny=ny, a=a, Lx=Lx_mag, Ly=Ly_mag)
    n_inside = int(mask.sum())
    print(f'  mask: {n_inside} / {nx*ny} cells '
          f'({100*n_inside/(nx*ny):.1f}%).')
    print(f'  building free-BC Newell demag kernel '
          f'(2*ny x 2*nx = {2*ny}x{2*nx})...')
    kernels = precompute_demag_kernels(
        p, kind='newell_freebc', accuracy=4.0, tol_conv=0.05)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Vortex IC: m ~ [-(Y-Yc), (X-Xc), R] / |.| inside the
    # mask, dummy +z outside (inert: K=0, D=0, fields masked).
    xs = (np.arange(nx) - (nx - 1) / 2.0) * a
    ys = (np.arange(ny) - (ny - 1) / 2.0) * a
    Xg, Yg = np.meshgrid(xs, ys)
    m_top = np.zeros((ny, nx, 3))
    vx = -Yg
    vy = Xg
    vz = np.full_like(Xg, R_core)
    norm = np.sqrt(vx * vx + vy * vy + vz * vz)
    m_top[..., 0] = vx / norm
    m_top[..., 1] = vy / norm
    m_top[..., 2] = vz / norm
    m_top[~mask, :] = np.array([0.0, 0.0, 1.0])
    plot_ic_2d(
        m=m_top, a=a, out_path=ic_png,
        title='SP5 IC: Landau vortex (R_core = 10 nm)')
    print(f'  IC figure: {ic_png}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Relax the vortex to equilibrium (no current).
    m_top, tau_v, n_v = _relax_vortex(
        m_top, p, kernels, mask, max_steps=max_steps_relax,
        tol_torque=tol_torque_relax, check_every=check_every)
    print(f'  [vortex relax] {n_v} steps, tau_max = '
          f'{tau_v:.2e} T')
    os.makedirs(out_dir, exist_ok=True)
    plot_ic_2d(
        m=m_top, a=a,
        out_path=os.path.join(out_dir, 'sp5_relaxed.png'),
        title='SP5 relaxed vortex (pre-current)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Drive with Zhang-Li STT; record <m_x>, <m_y> and the
    # vortex-core displacement (Dx, Dy) trajectory.
    inside_w = mask[..., np.newaxis].astype(float)
    inside_count = float(mask.sum())
    # Dot centre (cell-grid centre, matches the vortex IC).
    cx0 = (nx - 1) / 2.0 * a
    cy0 = (ny - 1) / 2.0 * a
    dx0, dy0 = _vortex_core(m_top, a, mask, cx0, cy0)
    times = [0.0]
    mx_hist = [float((m_top[..., 0] * inside_w[..., 0]).sum()
                     / inside_count)]
    my_hist = [float((m_top[..., 1] * inside_w[..., 0]).sum()
                     / inside_count)]
    dx_hist = [dx0]
    dy_hist = [dy0]
    t = 0.0
    for step in range(1, drive_n_steps + 1):
        m_top = _rk4_zhang_li(
            m_top, p, kernels, mask, u_T=u_T, beta=xi, dt=dt)
        t += dt
        if step % drive_sample_every == 0:
            mx = float((m_top[..., 0] * inside_w[..., 0]).sum()
                       / inside_count)
            my = float((m_top[..., 1] * inside_w[..., 0]).sum()
                       / inside_count)
            dxc, dyc = _vortex_core(m_top, a, mask, cx0, cy0)
            times.append(t)
            mx_hist.append(mx)
            my_hist.append(my)
            dx_hist.append(dxc)
            dy_hist.append(dyc)
    times = np.array(times)
    mx_hist = np.array(mx_hist)
    my_hist = np.array(my_hist)
    dx_hist = np.array(dx_hist)
    dy_hist = np.array(dy_hist)
    # Save the trajectory (averaged m + core displacement).
    np.savetxt(
        traj_txt,
        np.column_stack([times, mx_hist, my_hist,
                         dx_hist, dy_hist]),
        header='t(s)  <m_x>/Ms  <m_y>/Ms  Dx(m)  Dy(m)')
    print(f'  trajectory data: {traj_txt}')
    # Plot: <m> spiral (left) + core trajectory with the
    # Najafi steady-state target (right).
    fig, (axs, axc) = plt.subplots(
        1, 2, figsize=(10, 4), dpi=150)
    axs.plot(mx_hist, my_hist, '-', lw=1.0)
    axs.plot(mx_hist[0], my_hist[0], 'go', label='start')
    axs.plot(mx_hist[-1], my_hist[-1], 'rs', label='end')
    axs.set_xlabel('<m_x>')
    axs.set_ylabel('<m_y>')
    axs.set_title('SP5 <m> spiral')
    axs.legend(frameon=False)
    axs.grid(alpha=0.3)
    axc.plot(dx_hist * 1e9, dy_hist * 1e9, '-', lw=1.0,
             label='core path')
    axc.plot(NAJAFI_DX * 1e9, NAJAFI_DY * 1e9, 'k*',
             ms=14, label='Najafi 2009')
    axc.set_xlabel('Dx (nm)')
    axc.set_ylabel('Dy (nm)')
    axc.set_title('SP5 vortex-core trajectory')
    axc.legend(frameon=False)
    axc.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(traj_png)
    plt.close(fig)
    print(f'  trajectory plot: {traj_png}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Steady-state core = time-average over the late tail (the
    # damped-gyration orbit centre).
    n_tail = max(1, int(round(tail_frac * len(dx_hist))))
    dx_ss = float(np.mean(dx_hist[-n_tail:]))
    dy_ss = float(np.mean(dy_hist[-n_tail:]))
    err = math.hypot(dx_ss - NAJAFI_DX, dy_ss - NAJAFI_DY)
    passed = err < core_tol
    print(
        f'  steady-state core (last {tail_frac*100:.0f}% avg): '
        f'Dx = {dx_ss*1e9:+.2f} nm, Dy = {dy_ss*1e9:+.2f} nm')
    print(
        f'  Najafi 2009 reference: Dx = {NAJAFI_DX*1e9:+.1f} '
        f'nm, Dy = {NAJAFI_DY*1e9:+.1f} nm; '
        f'distance = {err*1e9:.2f} nm (tol {core_tol*1e9:.1f} '
        f'nm)')
    print(f'  status: {"PASS" if passed else "FAIL"}')
    return passed


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
