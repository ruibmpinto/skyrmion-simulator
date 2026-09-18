"""muMAG Standard Problem #4 benchmark (Field 1, 170 deg).

NIST muMAG Standard Problem #4 (Eicke & McMichael, 2000;
https://www.ctcms.nist.gov/~rdm/std4/spec4.html). A
Permalloy-like 500 nm x 125 nm x 3 nm thin-film rectangle
is equilibrated into an S-state (saturating field along
[1,1,1] then slowly reduced to zero), then driven by a
sudden in-plane reversal field.

Single-layer simulation
-----------------------
SP4 is a single ferromagnetic film. This test simulates a
single layer directly: only the top magnetization `m_top`
is evolved, the demag uses the self-layer kernel alone (the
inter-layer term vanishes because there is no second layer:
`m_bot` is held identically zero, so `N_inter . m_bot = 0`).
There is no dummy bottom layer and no interlayer coupling.

Canonical specification
-----------------------
- Geometry:  L_x = 500 nm, L_y = 125 nm, thickness 3 nm,
             free boundaries.
- Material:  A = 1.3e-11 J/m, M_s = 8.0e5 A/m, K = 0.
- Damping:   alpha = 0.02 for the dynamics (Gilbert form);
             relaxation toward the S-state uses alpha = 1
             for a fast quench.
- Gyromag.:  gamma_0 = 2.211e5 m/(A s) (Gilbert form), equal
             to gamma = 1.760e11 rad/(s T) via gamma_0 / mu_0.
- Field 1:   mu_0 H = (-24.6, +4.3, 0) mT applied at t = 0.

S-state preparation (NIST recipe)
---------------------------------
The reference S-state is obtained by applying a saturating
field along [1,1,1] and slowly reducing it to zero. This
test reproduces that recipe: the magnetization is initialised
uniformly along [1,1,1]/sqrt(3), a saturating field along the
same direction is applied and ramped to zero over several
relaxation stages, then relaxed at zero field. The [1,1,1]
direction breaks the x<->-x and the C/S degeneracy so the
physically correct S-state is selected.

Reference value
---------------
- t( <m_x>(t) = 0 first crossing ) ~ 0.136 ns (Field 1).
  PROVENANCE: this is the M_x first-crossing value from the
  NIST contributed / OOMMF reference time-series (the de-facto
  SP4 regression target used by mumax3 / fidimag / ubermag).
  It is NOT printed in the bundled NIST spec PDF (which gives
  no number) nor readable from the bundled results PDF (which
  plots <M_y>, not <M_x>). Treat as an external reference
  pending download of the NIST averaged-magnetization data.

Acceptance
----------
The first zero crossing of `<m_x>(t)` matches the reference
0.136 ns within `rtol_t_cross`.

Functions
---------
main
    Build SP4 setup, equilibrate to S-state via [1,1,1]
    saturate-and-reduce, apply Field 1, integrate, locate
    m_x=0 crossing, compare.
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
from src.simulator.integrator import llgs_rhs, rk4_step_single
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


def _make_sp4_params(alpha, nx, ny, a, dt):
    """Build a Permalloy SP4 parameter namespace.

    Parameters
    ----------
    alpha : float
        Gilbert damping for the dynamics phase.
    nx, ny : int
        Lattice cell counts.
    a : float
        Cell size (m).
    dt : float
        Time step (s).

    Returns
    -------
    p : SimpleNamespace
        SP4 parameter namespace.
    """
    p = default_params()
    p.A_ex = 1.3e-11           # J/m  (NIST SP4)
    p.D = 0.0                  # no DMI in SP4
    p.Ms = 8.0e5               # A/m
    p.alpha = float(alpha)
    p.gamma = 1.760e11         # rad / (s T) = gamma_0 / mu_0
    # SP4 specifies K = 0. effective_field_demag_pair uses the
    # BARE K convention (anisotropy_field uses `2 K / Ms`
    # directly) with the demag added explicitly, so K = 0 here.
    p.K_top = 0.0
    p.K_bot = 0.0
    # Single layer: no interlayer exchange.
    p.H_RKKY = 0.0
    # Geometry. t_Co is the film thickness. d_Ru is irrelevant
    # for the single-layer run (the inter-layer demag term is
    # multiplied by m_bot = 0), but must be a sane positive
    # value for the kernel builder.
    p.t_Co = 3.0e-9
    p.d_Ru = 100.0e-9
    p.nx = int(nx)
    p.ny = int(ny)
    p.a = float(a)
    p.dt = float(dt)
    p.H_ext = np.array([0.0, 0.0, 0.0])
    p.pulse = ConstantPulse(0.0)
    p.J_current = 0.0
    p.lambda_sq = 0.0
    _precompute(p)
    return p


def _single_layer_field(m_top, p, kernels, mask):
    """Effective field on a single ferromagnetic layer.

    Calls `effective_field_demag_pair` with the bottom layer
    set to zero, so the inter-layer demag term vanishes
    (`N_inter . 0 = 0`) and the returned top-layer field is
    the exchange + DMI + anisotropy + Zeeman + self-demag of a
    standalone layer.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Single-layer magnetisation, shape (ny, nx, 3).
    p : SimpleNamespace
        Parameter namespace.
    kernels : dict
        Free-BC demag kernels.
    mask : numpy.ndarray(2d)
        Boolean (ny, nx) magnet mask.

    Returns
    -------
    H_top : numpy.ndarray(3d)
        Single-layer effective field (Tesla).
    """
    m_zero = np.zeros_like(m_top)
    H_top, _ = effective_field_demag_pair(
        m_top, m_zero, p, kernels, mask=mask)
    return H_top


def _rk4_single_layer(m_top, p, kernels, mask, t, dt):
    """One single-layer RK4 step via the PRODUCTION stepper.

    Delegates to `integrator.rk4_step_single` with a single-
    layer RHS closure built from the production
    `effective_field_demag_pair` (self-demag only; bottom
    layer absent) and `llgs_rhs`. No integrator logic is
    reimplemented here -- only the field wiring.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Single-layer magnetisation, shape (ny, nx, 3).
    p : SimpleNamespace
        Parameter namespace.
    kernels : dict
        Free-BC demag kernels.
    mask : numpy.ndarray(2d)
        Boolean (ny, nx) magnet mask.
    t : float
        Current time (s).
    dt : float
        Time step (s).

    Returns
    -------
    m_top_new : numpy.ndarray(3d)
        Updated single-layer magnetisation.
    """
    def rhs(m, p_, tt):
        H = _single_layer_field(m, p_, kernels, mask)
        return llgs_rhs(m, H, p_, tt)
    return rk4_step_single(rhs, m_top, t, dt, p)


def _relax_single_layer(m_top, p, kernels, mask, H_ext,
                        max_steps, tol_torque, check_every,
                        tag):
    """Over-damped single-layer descent to equilibrium under
    the supplied applied field.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Initial single-layer magnetisation.
    p : SimpleNamespace
        Parameter namespace.
    kernels : dict
        Free-BC demag kernels.
    mask : numpy.ndarray(2d)
        Boolean (ny, nx) magnet mask.
    H_ext : numpy.ndarray(1d)
        Applied field (Tesla), shape (3,).
    max_steps : int
        Maximum descent steps.
    tol_torque : float
        Convergence threshold on max |m x (m x H)| (Tesla).
    check_every : int
        Torque-check cadence (steps).
    tag : str
        Label for the progress print.

    Returns
    -------
    m_top : numpy.ndarray(3d)
        Relaxed magnetisation.
    tau_max : float
        Final max torque (Tesla).
    n_done : int
        Steps taken.
    """
    saved = {'alpha': p.alpha, 'gamma_p': p.gamma_p,
             'pulse': p.pulse, 'lambda_sq': p.lambda_sq,
             'H_ext': p.H_ext}
    try:
        p.alpha = 1.0
        p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
        p.pulse = ConstantPulse(0.0)
        p.lambda_sq = 0.0
        p.H_ext = np.asarray(H_ext, dtype=float)
        t = 0.0
        tau_max = float('inf')
        n_done = int(max_steps)
        for step in range(int(max_steps)):
            m_top = _rk4_single_layer(
                m_top, p, kernels, mask, t, p.dt)
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
        print(f'  [{tag}] relaxed in {n_done} steps, '
              f'tau_max = {tau_max:.2e} T')
    finally:
        p.alpha = saved['alpha']
        p.gamma_p = saved['gamma_p']
        p.pulse = saved['pulse']
        p.lambda_sq = saved['lambda_sq']
        p.H_ext = saved['H_ext']
    return m_top, tau_max, n_done


def main():
    """Run the NIST muMAG Standard Problem 4 (Field 1) validation.

    Prepares the S-state in a single-layer permalloy rectangle
    (500 x 125 x 3 nm) with free-BC Newell demag, applies the
    Field-1 reversal drive over the 1 ns NIST window, tracks the
    spatially averaged magnetisation, and writes IC, S-state and
    <m> vs t figures. Passes when the <m_x> = 0 crossing time
    matches the 0.136 ns NIST reference within tolerance.

    Returns
    -------
    passed : bool
        True if the crossing time is within `rtol_t_cross`.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Geometry (NIST SP4).
    Lx_mag = 500.0e-9
    Ly_mag = 125.0e-9
    # Cell size <= l_ex/2 ~ 2.5 nm (l_ex = sqrt(2 A / mu_0
    # Ms^2) = 5.08 nm for Py).
    a = 2.5e-9
    # Pad the lattice so the free-BC zero-pad has a vacuum band.
    pad_factor = 1.5
    Lx_box = pad_factor * Lx_mag
    Ly_box = pad_factor * Ly_mag
    nx = int(round(Lx_box / a))
    ny = int(round(Ly_box / a))
    # Time integration.
    dt = 5.0e-14               # 0.05 ps
    # S-state preparation: [1,1,1] saturate-and-reduce.
    # Saturating field magnitude (Tesla) and the ramp-down
    # schedule. mu_0 Ms = 1.0 T for Ms = 8e5; a 1.5 T field
    # along [1,1,1] saturates against the out-of-plane demag.
    H_sat_mag = 1.5
    H_ramp = [1.5, 1.0, 0.5, 0.25, 0.1, 0.0]
    max_steps_sat = 40_000
    max_steps_relax = 200_000
    tol_torque_relax = 1.0e-4  # Tesla
    check_every = 1_000
    # Dynamics (Field 1).
    field1_T = np.array([-24.6e-3, +4.3e-3, 0.0])
    drive_time = 1.0e-9        # 1 ns NIST window
    sample_dt = 5.0e-12        # 5 ps cadence
    drive_n_steps = int(round(drive_time / dt))
    drive_sample_every = int(round(sample_dt / dt))
    # Reference and acceptance.
    t_cross_ref = 0.136e-9     # NIST <m_x>=0, Field 1 (ext.)
    # Acceptance widened to +/-10%: the authoritative NIST
    # review (Porter & Donahue 2020, Fig 8) brackets the
    # Field-1 reversal at t = 125-150 ps via OOMMF snapshots;
    # 0.136 ns sits in that window. No published scalar pins
    # it tighter than the ~+/-10% spread.
    rtol_t_cross = 0.10
    ic_png = 'docs/figures/validation/sp4_initial.png'
    out_dir = 'docs/figures/validation'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('test_mumag_sp4 (NIST muMAG SP#4, Field 1 / 170 deg, '
          'single layer):')
    print(
        f'  geometry: magnet {Lx_mag*1e9:.0f} x {Ly_mag*1e9:.0f} '
        f'x 3 nm; lattice box {nx*a*1e9:.0f} x {ny*a*1e9:.0f} '
        f'nm at a = {a*1e9:.1f} nm (nx={nx}, ny={ny}).')
    p = _make_sp4_params(alpha=0.02, nx=nx, ny=ny, a=a, dt=dt)
    mask = rect_mask(nx=nx, ny=ny, a=a, Lx=Lx_mag, Ly=Ly_mag)
    n_inside = int(mask.sum())
    print(f'  mask: {n_inside} / {nx*ny} cells '
          f'({100*n_inside/(nx*ny):.1f}%).')
    print(f'  building free-BC Newell demag kernel '
          f'(2*ny x 2*nx = {2*ny}x{2*nx} grid)...')
    kernels = precompute_demag_kernels(
        p, kind='newell_freebc', accuracy=4.0, tol_conv=0.05)
    print(f'  kernel kind: {kernels.get("kind")!r}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # IC: uniform along [1,1,1]/sqrt(3) inside the mask.
    # Outside-mask cells get a unit dummy (+z) so |m| = 1
    # everywhere (the integrator's normalize() rejects zero-
    # magnitude spins). These cells are inert: with K = 0 and
    # D = 0, every field term (exchange, DMI, anisotropy,
    # Zeeman, demag) is masked to zero outside the magnet, so
    # H = 0 there and the dummy spins stay frozen at +z. The
    # free-BC demag masks m before the convolution, so the
    # dummy +z carries no magnetic charge.
    dir_111 = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    m_top = np.zeros((ny, nx, 3))
    m_top[mask, :] = dir_111
    m_top[~mask, :] = np.array([0.0, 0.0, 1.0])
    plot_ic_2d(
        m=m_top, a=a, out_path=ic_png,
        title=(f'SP4 IC: uniform [1,1,1] inside '
               f'{Lx_mag*1e9:.0f}x{Ly_mag*1e9:.0f} nm rect'))
    print(f'  IC figure: {ic_png}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # S-state via [1,1,1] saturate-and-reduce: apply a field
    # along [1,1,1] and ramp it to zero, relaxing at each stage.
    for stage, h_mag in enumerate(H_ramp):
        H_stage = h_mag * dir_111
        steps = (max_steps_sat if h_mag > 0.0
                 else max_steps_relax)
        m_top, tau_s, n_s = _relax_single_layer(
            m_top, p, kernels, mask, H_ext=H_stage,
            max_steps=steps, tol_torque=tol_torque_relax,
            check_every=check_every,
            tag=f'sat ramp |H|={h_mag:.2f} T')
    # Save the S-state for diagnostics.
    os.makedirs(out_dir, exist_ok=True)
    plot_ic_2d(
        m=m_top, a=a,
        out_path=os.path.join(out_dir, 'sp4_sstate.png'),
        title='SP4 S-state ([1,1,1] saturate + reduce)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Dynamics: instantaneously apply Field 1, integrate with
    # alpha = 0.02 (single-layer RK4).
    p.H_ext = field1_T.copy()
    p.alpha = 0.02
    p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
    times_ps = []
    m_avg_xyz = []
    t = 0.0
    inside_w = mask[..., np.newaxis].astype(float)
    inside_count = float(mask.sum())
    avg0 = (m_top * inside_w).sum(axis=(0, 1)) / inside_count
    times_ps.append(0.0)
    m_avg_xyz.append(avg0.copy())
    for step in range(1, drive_n_steps + 1):
        m_top = _rk4_single_layer(
            m_top, p, kernels, mask, t, p.dt)
        t += p.dt
        if step % drive_sample_every == 0:
            avg = (m_top * inside_w).sum(axis=(0, 1)) \
                / inside_count
            times_ps.append(t * 1e12)
            m_avg_xyz.append(avg.copy())
    times_ps = np.array(times_ps)
    m_avg_xyz = np.array(m_avg_xyz)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Locate first m_x = 0 crossing (with linear interp).
    mx = m_avg_xyz[:, 0]
    sgn = np.sign(mx)
    cross = np.where(np.diff(sgn) != 0)[0]
    if cross.size == 0:
        print('  <m_x>(t) does not cross zero in the '
              'integration window.')
        return False
    i0 = int(cross[0])
    t_lo, t_hi = times_ps[i0], times_ps[i0 + 1]
    mx_lo, mx_hi = mx[i0], mx[i0 + 1]
    t_cross_ps = float(
        t_lo - mx_lo * (t_hi - t_lo) / (mx_hi - mx_lo))
    t_cross_s = t_cross_ps * 1e-12
    rel_err = abs(t_cross_s - t_cross_ref) / t_cross_ref
    print(
        f'  first <m_x>=0 crossing at t = {t_cross_ps:.3f} ps '
        f'(reference {t_cross_ref*1e12:.0f} ps, '
        f'rel.err = {rel_err*100:.2f} %)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Save the <m_alpha>(t) trace.
    fig, ax = plt.subplots(figsize=(6, 4), dpi=160)
    ax.plot(times_ps, m_avg_xyz[:, 0], label='<m_x>')
    ax.plot(times_ps, m_avg_xyz[:, 1], label='<m_y>')
    ax.plot(times_ps, m_avg_xyz[:, 2], label='<m_z>')
    ax.axhline(0.0, color='0.7', ls=':')
    ax.axvline(t_cross_ref * 1e12, color='k', ls='--',
               label=f'ref t_x=0 = '
               f'{t_cross_ref*1e12:.0f} ps')
    ax.set_xlabel('t (ps)')
    ax.set_ylabel('<m>')
    ax.set_title('NIST muMAG SP#4 Field 1: m-avg(t)')
    ax.legend(loc='best', frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = os.path.join(out_dir, 'sp4_m_avg_vs_t.png')
    fig.savefig(plot_path)
    plt.close(fig)
    print(f'  m_avg(t) plot: {plot_path}')
    passed = rel_err < rtol_t_cross
    print(
        f'  status: {"PASS" if passed else "FAIL"} '
        f'(rtol = {rtol_t_cross*100:.1f}%)')
    return passed


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
