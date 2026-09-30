"""Bogdanov-Hubert / Rohart-Thiaville skyrmion profile benchmark.

Reproduces the isolated-skyrmion radius reported in
Rohart & Thiaville, Phys. Rev. B 88, 184422 (2013) (RT 2013),
Section IV (skyrmion in nanodot) for the infinite-film limit.

RT 2013 canonical parameters (Sec. II, end of column 2)
-------------------------------------------------------
- A = 16 pJ/m
- K = 510 kJ/m^3  (treated as K_eff with the shape anisotropy
                   already folded in; RT explicitly omits the
                   dipolar coupling for the analytic case)
- mu_0 M_s = 1.1 T  =>  M_s = 8.755e5 A/m (used by RT in the
                        dipolar comparison; the analytic case
                        is M_s-independent because K is already
                        K_eff)
- Resulting domain-wall width  Delta = sqrt(A/K) = 5.601 nm
- Critical DMI                 D_c   = (4/pi) sqrt(A K)
                                     = 3.637 mJ/m^2
- Benchmark DMI                D     = 3 mJ/m^2
                                       (D/D_c = 0.825)

Reference value
---------------
RT 2013 Eq. (18), valid for D close to D_c on an infinite film,
predicts a skyrmion core radius

    R_s ~= Delta / sqrt(2 * (1 - D/D_c)) .

With the parameters above this gives R_s ~= 9.47 nm. RT's
Fig. 4(d) "infinite film limit" curve at D = 3 mJ/m^2 sits
near this value too. The acceptance tolerance below allows
~10% deviation because Eq. (18) is itself an asymptotic
expansion and because the numerical equilibrium on a finite
PBC box differs slightly from the infinite-film limit.

Functions
---------
main
    Build, relax (PBC + K_eff path, no FFT demag, no field),
    fit, compare to RT Eq. (18).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
# Third-party
import numpy as np
from scipy.optimize import curve_fit
# Local
from skyrmion_simulator.simulator.initial_conditions import skyrmion_profile
from skyrmion_simulator.simulator.pulses import ConstantPulse
from skyrmion_simulator.simulator.validation._helpers import (
    make_single_fm_params,
    plot_ic_2d,
    radial_profile,
    relax_single_fm,
)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def main():
    """Run the Rohart-Thiaville 2013 isolated-skyrmion validation.

    Relaxes an isolated Neel skyrmion in a large PBC film (infinite-
    film limit) via over-damped descent, then compares the
    equilibrium radius against the RT Eq. (18) prediction
    R_s = Delta / sqrt(2(1 - D/D_c)) and inspects the 360-degree
    profile, writing the IC figure. Passes when the relaxed radius
    matches Eq. (18) within tolerance.

    Returns
    -------
    passed : bool
        True if the relaxed skyrmion radius is within `rtol`.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Rohart-Thiaville 2013 canonical parameters (Sec. II).
    A_ex = 16.0e-12          # J/m
    K_eff = 510.0e3          # J/m^3   (already includes shape anis)
    Ms = 8.755e5             # A/m     (mu_0 M_s = 1.1 T)
    D = 3.0e-3               # J/m^2   (RT benchmark value)
    alpha = 0.14             # used only via LLG; relax uses alpha=1
    alpha_relax = 1.0
    gamma = 1.760e11         # rad / (s T) (free-electron value)
    H_ext = np.array([0.0, 0.0, 0.0])
    # Lattice: large PBC box approximates the RT "infinite film
    # limit". Box ~25*Delta gives R_s ~ Delta / sqrt(2(1-D/D_c))
    # ~ 9.5 nm with > 3 nm margin to the boundary.
    a = 1.0e-9
    nx = 96
    ny = 96
    # Time step. With alpha_relax = 1 the over-damped descent
    # rate scales as gamma * H_eff and the exchange prefactor
    # C_ex = 2 A / (Ms a^2) ~ 36 T sets the stiffest mode.
    # dt * gamma_p * C_ex ~ 0.06 stays well inside the RK4
    # imaginary-axis stability circle of radius ~2.8.
    dt = 2.0e-14
    # Relaxation tolerances. The torque convergence reads off
    # the dissipative torque amplitude in Tesla.
    max_steps = 200_000
    tol_torque = 1.0e-5
    check_every = 1_000
    # IC chosen close to the RT Eq. (18) prediction so the
    # descent does not have to traverse far. Starting too far
    # above R_s ~ 9.5 nm risks the descent overshooting
    # through R = 0 into the FM minimum.
    R_init = 10.0e-9
    dw_init = 5.6e-9
    n_bins = 64
    # Acceptance.
    rtol = 0.10
    ic_png = 'docs/figures/validation/bh_skyrmion_initial.png'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Derived analytics.
    Delta_an = math.sqrt(A_ex / K_eff)
    Dc_an = (4.0 / math.pi) * math.sqrt(A_ex * K_eff)
    D_over_Dc = D / Dc_an
    if D_over_Dc >= 1.0:
        raise RuntimeError(
            f'test_skyrmion_profile_bh: D >= D_c '
            f'(D={D*1e3:.3f}, D_c={Dc_an*1e3:.3f} mJ/m^2). '
            f'Eq. (18) only applies below D_c.')
    R_s_eq18 = Delta_an / math.sqrt(2.0 * (1.0 - D_over_Dc))
    print(
        f'test_skyrmion_profile_bh (RT 2013, infinite-film '
        f'limit, no demag):')
    print(
        f'  A={A_ex:.2e} J/m, K_eff={K_eff:.2e} J/m^3, '
        f'mu_0 M_s = {4.0*math.pi*1e-7*Ms:.3f} T, '
        f'D={D*1e3:.2f} mJ/m^2')
    print(
        f'  Delta = sqrt(A/K_eff) = {Delta_an*1e9:.3f} nm, '
        f'D_c = (4/pi) sqrt(A K_eff) = {Dc_an*1e3:.3f} mJ/m^2 '
        f'(D/D_c = {D_over_Dc:.3f})')
    print(
        f'  RT Eq. (18) reference R_s = '
        f'Delta / sqrt(2(1-D/D_c)) = {R_s_eq18*1e9:.3f} nm')
    print(
        f'  lattice {nx}x{ny} at a={a*1e9:.1f} nm '
        f'(box {nx*a*1e9:.0f} nm = {nx*a/Delta_an:.1f} Delta)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    p = make_single_fm_params(
        A_ex=A_ex, D=D, K_eff=K_eff, Ms=Ms,
        alpha=alpha, gamma=gamma, H_ext=H_ext,
        nx=nx, ny=ny, a=a, dt=dt, pulse=ConstantPulse(0.0))
    m0 = skyrmion_profile(
        nx=nx, ny=ny, a=a, R=R_init, dw=dw_init, polarity=+1)
    plot_ic_2d(
        m=m0, a=a, out_path=ic_png,
        title=(f'RT 2013 BH IC: R={R_init*1e9:.0f} nm, '
               f'Delta={dw_init*1e9:.0f} nm, D=3 mJ/m^2'))
    print(f'  IC figure: {ic_png}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m, tau_max, n_done = relax_single_fm(
        m_top=m0, p=p,
        alpha_relax=alpha_relax,
        max_steps=max_steps,
        tol_torque=tol_torque,
        check_every=check_every)
    print(f'  relaxed in {n_done} steps, '
          f'tau_max = {tau_max:.2e} T')
    # Collapse check.
    mz_mean = float(np.mean(m[..., 2]))
    if mz_mean > 0.95:
        print(f'  COLLAPSED to FM (mean m_z = {mz_mean:.3f}); '
              f'expected metastable isolated skyrmion.')
        return False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Radial fit; report the m_z = 0 crossing as R_s (RT's
    # definition: skyrmion core radius where m_z crosses zero).
    r, mz_r = radial_profile(m[..., 2], a=a, n_bins=n_bins)
    mask = np.isfinite(mz_r)
    r = r[mask]
    mz_r = mz_r[mask]
    # Linear interpolation to find m_z = 0 crossing.
    sign = np.sign(mz_r)
    crossings = np.where(np.diff(sign) != 0)[0]
    if crossings.size == 0:
        print('  no m_z = 0 crossing found in radial profile.')
        return False
    i0 = int(crossings[0])
    # Linear interpolation between r[i0] and r[i0+1].
    R_s_meas = float(
        r[i0] - mz_r[i0]
        * (r[i0 + 1] - r[i0]) / (mz_r[i0 + 1] - mz_r[i0]))
    # Also fit the BH ansatz for the shape residual.
    # Bogdanov-Hubert m_z(r) profile: 360-deg-arctan wall form.
    def _bh(rr, R, Delta):
        return np.cos(
            2.0 * np.arctan(np.exp(-(rr - R) / Delta)))
    popt, _ = curve_fit(
        _bh, r, mz_r, p0=(R_s_meas, Delta_an),
        bounds=([0.0, 0.1e-9], [r[-1], 50.0e-9]),
        maxfev=20000)
    R_fit, Delta_fit = float(popt[0]), float(popt[1])
    rms_resid = float(np.sqrt(np.mean(
        (mz_r - _bh(r, R_fit, Delta_fit)) ** 2)))
    rel_err = abs(R_s_meas - R_s_eq18) / R_s_eq18
    print(
        f'  measured R_s (m_z=0 crossing) = {R_s_meas*1e9:.3f} nm')
    print(
        f'  ansatz fit: R={R_fit*1e9:.3f} nm, '
        f'Delta={Delta_fit*1e9:.3f} nm '
        f'(shape RMS residual = {rms_resid:.4f})')
    print(
        f'  RT Eq. (18): R_s_ref = {R_s_eq18*1e9:.3f} nm '
        f'rel.err = {rel_err*100:.2f} %')
    passed = rel_err < rtol
    print(
        f'  status: {"PASS" if passed else "FAIL"} '
        f'(rtol = {rtol*100:.1f} %)')
    return passed


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
