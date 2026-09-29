"""Cortes-Ortuno 2D DMI standard problem (interfacial, disk).

Reproduces sub-problem 2 (Section 5, Table 2) of
Cortes-Ortuno, Beg, et al., New J. Phys. 20, 113015 (2018):
an isolated Neel skyrmion in a 50-nm-radius x 2-nm-thick
nanodisk under interfacial (C_nv) DMI, no external field, no
demag (per the MuMax3 reference script
`sims/MUMAX3/2D/skyrmion-Inter.mx3`: `NoDemagSpins = 1`, so
the supplied K_u acts as K_eff).

Canonical parameters (Table 2)
------------------------------
- Disk radius      R    = 50 nm
- Disk thickness   t    = 2 nm   (single-cell along z)
- Exchange         A    = 13 pJ/m
- Interfacial DMI  D    = 3 mJ/m^2
- Saturation Ms    M_s  = 0.86 MA/m
- Anisotropy       K_u  = 0.4 MJ/m^3  (used as K_eff: no demag)
- Cell             a    = 2 nm
- Lattice          50 x 50 cells (100 nm x 100 nm) with the
                   disk filling the box (Circle(2R)).

Reference values (CO Sec. 5 + dataset
`notebooks/data/results_2d/`)
-------------------------------------
- Theory (ODE)     r_sk = 22.03 nm
- OOMMF, Fidimag   r_sk = 21.87 nm
- MuMax3           r_sk = 22.10 nm

The reference m_z(r) profile is taken from
`result_2d_r-mz_interfacial_ODE.txt` (500-point analytic
solution of Eq. (4) in CO 2018) and compared to the simulator's
relaxed configuration along the disk diameter.

Acceptance
----------
- r_sk within `rtol_r` of the theory value (22.03 nm)
- Profile m_z(r) RMS residual to the ODE reference within
  `rtol_profile`

Functions
---------
main
    Build the disk mask, IC, relax via rk4_step + custom
    free-BC RHS, compare to CO reference profile.
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
from src.phase_diagram.relaxation import relax
from src.simulator.initial_conditions import skyrmion_profile
from src.simulator.lattice import disk_mask
from src.simulator.parameters import _precompute, default_params
from src.simulator.pulses import ConstantPulse
from src.simulator.validation._helpers import (
    plot_ic_2d,
    radial_profile,
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

# Bundled CO 2018 dataset dir holding the ODE reference profile.
CO_DATA_DIR = (
    'refs/2018 Cortes-Ortuno - Dataset Proposal for a '
    'micromagnetic standard problem for materials with '
    'Dzyaloshinskii-Moriya interaction.zip.0/notebooks/data/'
    'results_2d')


def _co_2d_params(A_ex, D, K_eff, Ms, alpha, gamma, nx, ny, a, dt):
    """Build a simulator parameter namespace for CO 2D.

    K is supplied as K_eff (the CO MuMax3 reference uses
    NoDemagSpins=1, so K_u is already the effective value).
    The bare-K convention used internally by `_precompute`
    requires K_top = K_eff + 0.5*mu_0*Ms^2 so that
    C_anis_top = 2*K_eff/Ms once the demag fold-in is
    re-applied.
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


def _measure_rsk_along_radius(m, mask, a):
    """Skyrmion radius r_sk from the m_z = 0 crossing along
    the +x diameter starting at the disc centre.

    Follows the CO definition (Sec. 5): r_sk is the radius
    where m_z(r) = 0 along the disc diameter. We extract the
    centre row, walk outward in +x, and linearly interpolate
    between the two adjacent sites that bracket the sign
    change. Returns NaN if no crossing exists inside the mask.
    """
    ny, nx = mask.shape
    # Centre row (use the row containing the lattice centre).
    iy_c = ny // 2
    ix_c = nx // 2
    mz_row = m[iy_c, ix_c:, 2]
    mask_row = mask[iy_c, ix_c:]
    # Walk outward inside the mask; locate the first sign
    # change of m_z.
    for i in range(1, mz_row.size):
        if not mask_row[i]:
            break
        if mz_row[i - 1] * mz_row[i] < 0.0:
            # Linear interpolation for the crossing.
            r_lo = (i - 1) * a
            r_hi = i * a
            mz_lo = mz_row[i - 1]
            mz_hi = mz_row[i]
            r_sk = r_lo - mz_lo * (r_hi - r_lo) / (mz_hi - mz_lo)
            return float(r_sk)
    return float('nan')


def main():
    """Run the Cortes-Ortuno 2018 DMI standard-problem validation.

    Relaxes an isolated Neel skyrmion in a 50 nm-radius disk with
    interfacial DMI (sub-problem 2) via over-damped descent, then
    compares the equilibrium radius and azimuthal profile against
    the MuMax3 reference and writes the IC figure. Passes when both
    the radius and the profile RMS residual are within tolerance.

    Returns
    -------
    passed : bool
        True if both the r_sk and profile checks pass.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cortes-Ortuno 2D problem (Table 2 + Sec. 5).
    A_ex = 13.0e-12          # J/m
    D = 3.0e-3               # J/m^2 (interfacial)
    Ms = 0.86e6              # A/m
    K_eff = 0.4e6            # J/m^3 (no demag => K_u = K_eff)
    alpha = 0.3              # CO Mumax3 alpha for relaxation
    gamma = 1.760e11
    R_disk = 50.0e-9         # disk radius
    # Cell + lattice (must match the MuMax3 ref: 50x50 cells,
    # 2x2 nm in plane). The disc Circle(2R) fills the box.
    a = 2.0e-9
    nx = 50
    ny = 50
    # dt: gamma*C_ex*dt < 1 for stability. C_ex = 2A/(Ms a^2)
    # = 2*13e-12/(8.6e5 * 4e-18) = 7.56 T; gamma_p ~ gamma/2,
    # dt = 5e-14 gives dt*gamma_p*C_ex ~ 0.33. Safe.
    dt = 5.0e-14
    # Over-damped descent.
    alpha_relax = 1.0
    max_steps = 200_000
    tol_torque = 1.0e-5
    check_every = 1_000
    # IC: polarity = +1 (core m_z = -1, background m_z = +1)
    # matches the simulator's D > 0 chirality convention
    # (left-handed Neel with outward-radial in-plane spins).
    # The CO MuMax3 reference flips both the DMI sign
    # (Dind = -3e-3) and the core polarity (core m_z = +1,
    # bg m_z = -1); the two flips together give the same
    # physics as our (positive D, polarity = +1) setup --
    # the skyrmion radius is unchanged by simultaneous DMI
    # and polarity inversion.
    R_init = 15.0e-9
    dw_init = 5.7e-9
    n_bins = 60
    # Acceptance.
    rtol_r = 0.05
    # Profile RMS residual is dominated by the wall position;
    # a 5% r_sk shift on a Delta ~ 5.7 nm wall produces an
    # RMS of order 0.1 even with otherwise perfect agreement.
    # The threshold is set so a simulator that passes the
    # r_sk check also passes the profile check.
    rtol_profile = 0.15
    ic_png = 'docs/figures/validation/dmi_sp_initial.png'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Delta_an = math.sqrt(A_ex / K_eff)
    Dc_an = (4.0 / math.pi) * math.sqrt(A_ex * K_eff)
    print('test_dmi_standard_problem (CO 2018 sub-problem 2):')
    print(
        f'  A={A_ex:.2e} J/m, K_eff={K_eff:.2e} J/m^3, '
        f'Ms={Ms:.2e} A/m, D={D*1e3:.2f} mJ/m^2')
    print(
        f'  Delta = sqrt(A/K_eff) = {Delta_an*1e9:.3f} nm, '
        f'D_c = (4/pi) sqrt(A K_eff) = {Dc_an*1e3:.3f} mJ/m^2 '
        f'(D/D_c = {D/Dc_an:.3f})')
    print(
        f'  disk R = {R_disk*1e9:.0f} nm, lattice {nx}x{ny} '
        f'at a = {a*1e9:.1f} nm '
        f'(box {nx*a*1e9:.0f} x {ny*a*1e9:.0f} nm)')
    print(
        f'  reference r_sk (CO theory ODE) = 22.03 nm; '
        f'OOMMF/Fidimag = 21.87 nm; MuMax3 = 22.10 nm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Mask + IC.
    mask = disk_mask(nx=nx, ny=ny, a=a, R=R_disk)
    print(f'  disk mask: {int(mask.sum())} / {nx*ny} cells '
          f'inside disk ({100*mask.sum()/(nx*ny):.1f}%)')
    p = _co_2d_params(
        A_ex=A_ex, D=D, K_eff=K_eff, Ms=Ms,
        alpha=alpha, gamma=gamma,
        nx=nx, ny=ny, a=a, dt=dt)
    m_top = skyrmion_profile(
        nx=nx, ny=ny, a=a, R=R_init, dw=dw_init, polarity=+1)
    # Outside the mask: dummy unit vector m_z = +1 so the
    # |m| = 1 constraint enforced by `normalize` inside
    # `rk4_step` is always satisfiable. The mask zeroes the
    # effective field outside the disc, so these dummy
    # spins do not evolve and contribute nothing physical.
    outside = ~mask
    m_top[outside, :] = np.array([0.0, 0.0, 1.0])
    plot_ic_2d(
        m=m_top, a=a, out_path=ic_png,
        title=(f'CO SP 2D IC (disk R={R_disk*1e9:.0f} nm): '
               f'core m_z=+1, polarity=-1'))
    print(f'  IC figure: {ic_png}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Relax the single FM layer through the production relax()
    # (free-BC mask, no demag, over-damped quench).
    m_top, _mb, _conv, n_done, _E, tau_max = relax(
        m_top, None, p, None, mask=mask,
        alpha_relax=alpha_relax, tol_torque=tol_torque,
        max_steps=max_steps, check_every=check_every)
    print(f'  relaxed in {n_done} steps, '
          f'tau_max = {tau_max:.2e} T')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Measure r_sk and compare to CO theory (22.03 nm).
    r_sk = _measure_rsk_along_radius(m=m_top, mask=mask, a=a)
    if not np.isfinite(r_sk):
        print('  no m_z=0 crossing along the +x diameter; '
              'skyrmion may have collapsed.')
        return False
    r_sk_theory = 22.03e-9
    r_err = abs(r_sk - r_sk_theory) / r_sk_theory
    print(
        f'  measured r_sk = {r_sk*1e9:.3f} nm '
        f'(CO theory 22.03 nm, rel.err = {r_err*100:.2f} %)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Chirality check: at the +x diameter, the in-plane
    # radial component m_r equals m_x at phi = 0. For CO's
    # interfacial (C_nv) DMI with their Dind convention, the
    # outward-Neel chirality has m_r(r_sk) = +1. Sample
    # m_x at the centre row, at the lattice cell nearest to
    # r_sk along +x.
    iy_c = m_top.shape[0] // 2
    ix_c = m_top.shape[1] // 2
    i_rsk = int(round(r_sk / a))
    ix_sample = min(ix_c + i_rsk, m_top.shape[1] - 1)
    mx_at_rsk = float(m_top[iy_c, ix_sample, 0])
    chirality = 'outward (Neel, m_r > 0)' if mx_at_rsk > 0.1 else (
        'inward (Neel, m_r < 0)' if mx_at_rsk < -0.1 else
        'Bloch or undefined (|m_r| < 0.1)')
    print(
        f'  chirality check at r = r_sk (+x diameter): '
        f'm_x = {mx_at_rsk:+.3f} -> {chirality}')
    print(
        f'  CO Cnv (interfacial) reference: outward Neel '
        f'(m_r = +1 at r_sk). Match = '
        f'{mx_at_rsk > 0.5}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load CO ODE reference m_z(r) and compare to the radial
    # profile of the relaxed configuration.
    ode_path = os.path.join(
        CO_DATA_DIR, 'result_2d_r-mz_interfacial_ODE.txt')
    if not os.path.exists(ode_path):
        raise RuntimeError(
            f'CO 2D ODE reference file not found at '
            f'{ode_path!r}.')
    ode_data = np.loadtxt(ode_path)
    r_ode_nm = ode_data[:, 0]
    # CO data: core m_z = +1, background m_z = -1.
    # This simulator (polarity=+1, D>0): core m_z = -1,
    # background m_z = +1. The two conventions are related
    # by m_z -> -m_z; flip the reference for the comparison.
    mz_ode = -ode_data[:, 1]
    # Sim radial profile (azimuthally averaged on the disk
    # interior only).
    r_sim, mz_sim = radial_profile(
        m_top[..., 2] * mask, a=a, n_bins=n_bins)
    keep = np.isfinite(mz_sim) & (r_sim <= R_disk - a)
    r_sim = r_sim[keep]
    mz_sim = mz_sim[keep]
    # Interpolate the CO ODE reference onto the sim radial
    # grid for an apples-to-apples residual. CO's ODE file
    # uses nm; we map to metres for the comparison and clip
    # any out-of-range bins to the ODE endpoint values.
    mz_ode_on_sim = np.interp(
        r_sim * 1e9, r_ode_nm, mz_ode)
    rms_profile = float(np.sqrt(np.mean(
        (mz_sim - mz_ode_on_sim) ** 2)))
    print(
        f'  profile RMS residual to CO ODE reference '
        f'(over {r_sim.size} radial bins) = '
        f'{rms_profile:.4f}')
    passed_r = r_err < rtol_r
    passed_p = rms_profile < rtol_profile
    passed = passed_r and passed_p
    print(
        f'  status: r_sk={"PASS" if passed_r else "FAIL"} '
        f'(rtol {rtol_r*100:.0f}%), '
        f'profile={"PASS" if passed_p else "FAIL"} '
        f'(rtol {rtol_profile*100:.0f}%) '
        f'-> {"PASS" if passed else "FAIL"}')
    return passed


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
