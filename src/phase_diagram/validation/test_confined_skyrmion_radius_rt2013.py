"""Confined-skyrmion radius R_s(D) in a finite dot (RT 2013).

Reproduces the Rohart & Thiaville 2013 (Phys. Rev. B 88,
184422) confined-skyrmion benchmark: an isolated skyrmion
relaxed at ZERO applied field in a circular nanodot. RT 2013
is an entirely zero-field study; its testable observable is
the skyrmion core radius R_s (the m_z = 0 ring) as a function
of D / D_c and dot radius R, plotted in their Fig. 4(d).

This replaces the earlier (mis-scoped) "collapse field"
test: RT 2013 contains no applied-field collapse benchmark,
so a collapse field H_coll cannot be validated against it.

Method / reference model
------------------------
RT 2013 uses the *local dipolar energy approximation* (their
Fig. 3 caption), i.e. the demag is folded into an effective
uniaxial anisotropy K_eff -- there is no spatially-resolved
magnetostatic field. We therefore relax with the no-demag
`effective_field` path (free-BC mask + RT/mumax3 DMI edge
condition) using K = K_eff, matching the reference model.

Canonical parameters (RT 2013)
------------------------------
- A = 16 pJ/m (from L_0 = 4 pi A / D: D=9 -> L_0=22.34 nm,
  D=4 -> L_0=50.26 nm both yield A = 16 pJ/m).
- K_eff = 0.50 MJ/m^3, so D_c = (4/pi) sqrt(A K_eff)
  = 3.61 mJ/m^2. This matches the Fig. 4 caption, which
  states D = 3 -> D/D_c = 0.83 and D = 4.5 -> D/D_c = 1.25
  (i.e. D_c = 3/0.83 = 3.61 mJ/m^2).
- Delta = sqrt(A / K_eff) = 5.66 nm.

Reference values read from Fig. 4(d)
------------------------------------
The confinement-dominated regime is D/D_c > 1, where the
infinite-film skyrmion would diverge / not exist as an
isolated object and the dot radius sets R_s (Fig. 4(c) and
caption: "the skyrmion radius is limited by the dot radius").
At D = 4.5 mJ/m^2 (D/D_c = 1.25), Fig. 4(d) gives:
    R = 50 nm dot  -> R_s ~ 25 nm  (this test)
    R = 75 nm dot  -> R_s ~ 46 nm
    R = 100 nm dot -> R_s ~ 68 nm
A R = 50 nm dot is used here: mid-range, figure-readable,
confinement-dominated, and distinct from the infinite-film
Bogdanov-Hubert test (Eq. 18 diverges at D/D_c = 1). The
D < D_c regime (e.g. D = 3) is NOT used because there R_s
collapses to the infinite-film value and the test would
merely duplicate the BH benchmark.

Acceptance
----------
Relaxed R_s (the m_z = 0 radius along +x from the dot centre)
within `rtol` of the Fig. 4(d) read value. The tolerance is
loose (figure-read uncertainty ~ +/- few nm). The infinite-
film Eq. (18) value is reported alongside to make the
confinement enhancement explicit.

Functions
---------
main
    Build the dot + skyrmion IC + RT BC, relax at H = 0,
    measure R_s, compare to RT 2013 Fig. 4(d).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
# Third-party
import numpy as np
# Local
from src.phase_diagram.relaxation import relax
from src.simulator.initial_conditions import skyrmion_profile
from src.simulator.lattice import disk_mask
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


def _make_rt_params(A_ex, D, K_eff, Ms, alpha, gamma, nx, ny,
                    a, dt):
    """RT 2013 dot parameter namespace.

    K is folded as K_eff (local dipolar approximation); the
    simulator's `_precompute` builds C_anis = 2 K_eff / Ms by
    storing K = K_eff + 0.5 mu0 Ms^2 (the +0.5 mu0 Ms^2 is
    removed again by the thin-film K_eff correction inside
    `_precompute`). No explicit demag is used.

    Parameters
    ----------
    A_ex : float
        Exchange stiffness (J/m).
    D : float
        DMI constant (J/m^2).
    K_eff : float
        Effective uniaxial anisotropy (J/m^3).
    Ms : float
        Saturation magnetisation (A/m).
    alpha : float
        Gilbert damping.
    gamma : float
        Gyromagnetic ratio (rad/(s T)).
    nx, ny : int
        Lattice cell counts.
    a : float
        Cell size (m).
    dt : float
        Time step (s).

    Returns
    -------
    p : SimpleNamespace
        Parameter namespace.
    """
    p = default_params()
    p.A_ex = float(A_ex)
    p.D = float(D)
    p.Ms = float(Ms)
    p.alpha = float(alpha)
    p.gamma = float(gamma)
    half_mu0_Ms2 = 0.5 * p.mu0 * p.Ms * p.Ms
    p.K_top = float(K_eff) + half_mu0_Ms2
    p.K_bot = float(K_eff) + half_mu0_Ms2
    p.H_ext = np.array([0.0, 0.0, 0.0])
    p.H_RKKY = 0.0
    p.nx = int(nx)
    p.ny = int(ny)
    p.a = float(a)
    p.dt = float(dt)
    p.pulse = ConstantPulse(0.0)
    p.J_current = 0.0
    p.lambda_sq = 0.0
    _precompute(p)
    return p


def _measure_Rs(m_top, ix_c, iy_c, a, mask):
    """Skyrmion core radius = first m_z = 0 crossing along +x
    from the dot centre, with linear interpolation.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Relaxed magnetisation.
    ix_c, iy_c : int
        Centre cell indices.
    a : float
        Cell size (m).
    mask : numpy.ndarray(2d)
        Boolean dot mask.

    Returns
    -------
    R_s : float or None
        Radius (m), or None if no crossing inside the mask.
    """
    mz_row = m_top[iy_c, ix_c:, 2]
    mask_row = mask[iy_c, ix_c:]
    for i in range(1, mz_row.size):
        if not mask_row[i]:
            return None
        if mz_row[i - 1] * mz_row[i] < 0.0:
            # Linear interpolation between cells i-1 and i.
            z0, z1 = mz_row[i - 1], mz_row[i]
            frac = z0 / (z0 - z1)
            return (i - 1 + frac) * a
    return None


def main():
    # Relax an isolated skyrmion at H=0 in the dot, measure R_s,
    # compare to RT 2013 Fig. 4(d).
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RT 2013 material and dot.
    A_ex = 16.0e-12          # J/m
    K_eff = 0.50e6           # J/m^3 (D_c = 3.61 mJ/m^2)
    Ms = 1.1e6               # A/m (RT 2013 Co/Pt; irrelevant
    # to R_s but set for completeness)
    D = 4.5e-3               # J/m^2 (D/D_c = 1.25)
    gamma = 1.760e11
    R_dot = 50.0e-9
    a = 2.0e-9
    nx = 60                  # 120 nm box (dot R=50 -> +10 nm)
    ny = 60
    dt = 5.0e-14
    # IC: skyrmion (core m_z = -1, polarity=+1) seeded near the
    # expected confined R_s (~25 nm).
    R_init = 25.0e-9
    Delta = math.sqrt(A_ex / K_eff)
    dw_init = Delta
    # Relaxation.
    max_steps = 120_000
    tol_torque = 1.0e-5
    check_every = 1_000
    # RT 2013 Fig. 4(d) read value (R = 50 nm dot, D = 4.5
    # mJ/m^2, D/D_c = 1.25): R_s ~ 25 nm. Tolerance is loose
    # (figure read uncertainty).
    Rs_ref = 25.0e-9
    rtol = 0.25
    ic_png = ('docs/figures/validation/'
              'confined_Rs_rt2013_initial.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Critical DMI D_c and the reduced DMI kappa = D / D_c
    # (kappa > 1 is the confinement-dominated regime probed here).
    Dc = (4.0 / math.pi) * math.sqrt(A_ex * K_eff)
    kappa = math.pi * D / (4.0 * math.sqrt(A_ex * K_eff))
    print('test_confined_skyrmion_radius_rt2013 (RT 2013 '
          'Fig. 4d, zero field):')
    print(
        f'  A={A_ex:.2e}, K_eff={K_eff:.2e}, '
        f'D={D*1e3:.2f} mJ/m^2 (D_c={Dc*1e3:.2f} mJ/m^2, '
        f'D/D_c={D/Dc:.2f}, kappa={kappa:.2f})')
    print(
        f'  dot R = {R_dot*1e9:.0f} nm, lattice {nx}x{ny} at '
        f'a = {a*1e9:.1f} nm; Delta = {Delta*1e9:.2f} nm.')
    # Eq. (18) only holds below threshold (D < D_c); above it
    # the infinite-film isolated skyrmion radius diverges and
    # the dot edge alone confines R_s -- which is exactly the
    # physics this test probes.
    if D < Dc:
        Rs_inf = Delta / math.sqrt(2.0 * (1.0 - D / Dc))
        print(
            f'  Eq.(18) infinite-film R_s = {Rs_inf*1e9:.1f} '
            f'nm (confinement expands this in the dot).')
    else:
        print(
            '  Eq.(18) infinite-film R_s diverges for D >= '
            'D_c; the dot radius alone sets R_s here.')
    p = _make_rt_params(
        A_ex=A_ex, D=D, K_eff=K_eff, Ms=Ms,
        alpha=0.3, gamma=gamma, nx=nx, ny=ny, a=a, dt=dt)
    mask = disk_mask(nx=nx, ny=ny, a=a, R=R_dot)
    ix_c = nx // 2
    iy_c = ny // 2
    # IC inside the dot, dummy +z outside (single FM layer).
    m_top0 = skyrmion_profile(
        nx=nx, ny=ny, a=a, R=R_init, dw=dw_init, polarity=+1)
    m_top0[~mask, :] = np.array([0.0, 0.0, 1.0])
    plot_ic_2d(
        m=m_top0, a=a, out_path=ic_png,
        title=(f'RT2013 confined R_s IC: dot R={R_dot*1e9:.0f} '
               f'nm, sk R_init={R_init*1e9:.0f} nm'))
    print(f'  IC figure: {ic_png}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Single-layer relaxation through the production relax()
    # (free-BC mask, no demag, over-damped quench).
    m_top, _m_bot, _conv, n_done, _E, tau_max = relax(
        m_top0, None, p, None, mask=mask, alpha_relax=1.0,
        tol_torque=tol_torque, max_steps=max_steps,
        check_every=check_every)
    R_s = _measure_Rs(m_top, ix_c, iy_c, a, mask)
    if R_s is None:
        print(f'  relax {n_done} steps, tau_max={tau_max:.2e} '
              f'T: NO m_z=0 ring (skyrmion did not survive).')
        print('  status: FAIL')
        return False
    rel_err = abs(R_s - Rs_ref) / Rs_ref
    passed = rel_err < rtol
    print(
        f'  relax {n_done} steps, tau_max={tau_max:.2e} T')
    print(
        f'  measured R_s = {R_s*1e9:.1f} nm '
        f'(RT 2013 Fig.4d ~ {Rs_ref*1e9:.0f} nm, '
        f'rel.err = {rel_err*100:.1f} %)')
    print(
        f'  status: {"PASS" if passed else "FAIL"} '
        f'(rtol = {rtol*100:.0f}%)')
    return passed


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
