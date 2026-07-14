"""Thiele v_SOT(R/Delta) validation against Pham 2024 Fig. S49.

Reproduces the methodology of the production sweep
`scripts/sweep_S49_TSH.py`, which mirrors Figure S49 of

    Pham et al., "Fast current-induced skyrmion motion in
    synthetic antiferromagnets" (2024), supplementary sec. 1.7.

S49 is a CALCULATION: the domain-wall width is fixed at
Delta = 24.5 nm and the core radius is swept as an imposed
parameter R/Delta = 1..5 (NOT relaxed to an equilibrium, NOT a
DMI sweep). Two closed-form steady-state speeds are compared --
the SOT-driven speed (orange) and the topological-spin-Hall
(TSH) speed for lambda^2 = 3 and 50 nm^2 (blue/violet):

    v_SOT = pi H_DL R gamma / (2 alpha (R/Delta + Delta/R)),
    v_TSH = -b_j lambda^2 I / (alpha (R/Delta + Delta/R)).

Both are implemented in `src.simulator.topological_torque`.

The test has two layers:

1. Analytic gate (always, fast): evaluate the production
   `sot_thiele_speed` and `tsh_thiele_speed` over R/Delta = 1..5
   at the S49 parameters and check
     - v_SOT matches its independently-derived closed form
       v_SOT = (pi H_DL Delta gamma / 2 alpha) * x^2/(x^2+1),
       x = R/Delta  (locks the implementation),
     - v_SOT is monotone increasing and saturates,
     - v_TSH scales linearly with lambda^2 at fixed R/Delta.
2. Micromagnetic spot-check (one rigid point, R/Delta small):
   relax a SAF skyrmion seeded at the imposed R and fixed
   Delta, fit (R_fit, Delta_fit) with the Bogdanov-Hubert
   ansatz, drive with DC SOT, and compare the measured
   centroid speed to `sot_thiele_speed(R_fit, Delta_fit, p)`.
   This is where the rigid-Thiele assumption is genuinely
   exercised against the LLGS integrator; it is restricted to
   small R/Delta where the skyrmion stays rigid (S49 shows the
   rigid approximation degrading at large R/Delta, by design).

Run with:
    python -m src.simulator.validation.test_thiele_v_sot
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
from scipy.optimize import curve_fit
# Local
from src.phase_diagram.relaxation import relax
from src.simulator.analysis import (
    skyrmion_center, skyrmion_diameter)
from src.simulator.demag import precompute_demag_kernels
from src.simulator.fields import effective_field_demag_pair
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.integrator import llgs_rhs, rk4_step
from src.simulator.parameters import _precompute, default_params
from src.simulator.pulses import ConstantPulse
from src.simulator.topological_torque import (
    sot_thiele_speed,
    tsh_thiele_speed,
)
from src.simulator.validation._helpers import (
    plot_ic_2d,
    radial_profile,
)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _rhs_saf_demag(m_top, m_bot, p, t):
    """SAF RHS with FFT demag, matching phase_diagram.relaxation
    but reading the SOT pulse from `p.pulse(t)` so the test can
    integrate driven dynamics."""
    H_top, H_bot = effective_field_demag_pair(
        m_top, m_bot, p, _rhs_saf_demag.kernels)
    return (llgs_rhs(m_top, H_top, p, t),
            llgs_rhs(m_bot, H_bot, p, t))


def _bh(rr, R, Delta):
    """Bogdanov-Hubert m_z(r) ansatz."""
    return np.cos(2.0 * np.arctan(np.exp(-(rr - R) / Delta)))


def _fit_bh_delta(m_top, a, R_guess):
    """Fit the top-layer m_z radial profile to the BH ansatz
    and return (R_fit, Delta_fit, rms_residual)."""
    n_bins = 80
    r, mz_r = radial_profile(m_top[..., 2], a=a, n_bins=n_bins)
    keep = np.isfinite(mz_r)
    r = r[keep]
    mz_r = mz_r[keep]
    sgn = np.diff(np.sign(mz_r))
    cross = np.where(sgn != 0)[0]
    R0 = float(r[cross[0]]) if cross.size else float(R_guess)
    popt, _ = curve_fit(
        _bh, r, mz_r, p0=(R0, 24.5e-9),
        bounds=([0.0, 0.1e-9], [r[-1], 100.0e-9]),
        maxfev=20000)
    R_fit, Delta_fit = float(popt[0]), float(popt[1])
    rms = float(np.sqrt(np.mean(
        (mz_r - _bh(r, R_fit, Delta_fit)) ** 2)))
    return R_fit, Delta_fit, rms


# -----------------------------------------------------------------------------
def _make_set_B_params(D, alpha, gamma, J0, nx, ny, a, dt,
                       R, Delta, lambda_sq):
    """Pham Set B parameters with an imposed skyrmion size.

    Mirrors `sweep_S49_TSH._build_set_B_params`: the radius and
    wall width are set directly (S49 imposes them), and the
    constant SOT drive is attached so `sot_thiele_speed` reads
    the right current density.
    """
    p = default_params()
    p.D = float(D)
    p.alpha = float(alpha)
    p.gamma = float(gamma)
    p.nx = int(nx)
    p.ny = int(ny)
    p.a = float(a)
    p.dt = float(dt)
    p.lambda_sq = float(lambda_sq)
    p.skyrmion_R = float(R)
    p.skyrmion_dw = float(Delta)
    _precompute(p)
    p.pulse = ConstantPulse(float(J0))
    return p


def main():
    """Run the Pham 2024 S49 Thiele SOT+TSH velocity validation.

    Evaluates the closed-form SOT+TSH Thiele speed over the imposed
    R/Delta sweep of Pham Fig S49 and gates it against the analytic
    expression, then optionally runs a micromagnetic LLGS
    spot-check of the drive velocity and skyrmion rigidity, writing
    the v-vs-R and IC figures. Passes when the closed form,
    monotonicity/lambda ordering, and (if enabled) the spot-check
    velocity and rigidity all fall within tolerance.

    Returns
    -------
    all_ok : bool
        True if every enabled acceptance check passes.
    """
    # =========================== User Configuration =========================
    # S49 caption parameters (Pham 2024, Set B).
    Delta = 24.5e-9                  # m, FIXED wall width
    D = 0.62e-3                      # J/m^2
    alpha = 0.216
    gamma = 175.9e9                  # rad/(s T)
    J0 = 8.0e11                      # A/m^2 DC drive
    # Imposed R/Delta sweep (S49 figure abscissa).
    R_over_Delta = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    lambda_sq_tsh = [3.0e-18, 50.0e-18]   # m^2 (3, 50 nm^2)
    # ---- Analytic-gate tolerances ----
    # Closed-form match is exact algebra -> tight.
    closed_form_rtol = 1.0e-6
    # ---- Micromagnetic spot-check ----
    do_llgs_spot = True
    spot_R_over_Delta = 1.5          # rigid regime
    # Lower drive than the S49 analytic J0=8e11: at 8e11 the
    # skyrmion deforms (rigidity breakdown, which S49 itself
    # documents); 1e11 keeps it rigid so the LLGS motion can be
    # compared to the closed form. v_thiele is recomputed at
    # this drive (it scales linearly with J0).
    spot_J0 = 1.0e11
    # Box must be many skyrmion radii wide: the Set-B skyrmion
    # relaxes to its equilibrium (~R/Delta=2.4, R~67 nm), so a
    # small box lets PBC images suppress the motion. 160 @ 5 nm
    # = 800 nm box (~12 R) is the geometry that validated the
    # D=0.62 velocity to 1.7% at this drive.
    spot_nx = 160
    spot_ny = 160
    spot_a = 5.0e-9                  # 800 nm box
    spot_dt = 5.0e-14
    spot_demag_accuracy = 4.0
    spot_demag_tol_conv = 0.05
    spot_relax_max_steps = 120_000
    spot_relax_alpha = 1.0
    spot_relax_tol_torque = 1.0e-5
    spot_relax_tol_dE = 1.0e-8
    spot_relax_check_every = 1_000
    spot_drive_total = 1.5e-9
    spot_sample_dt = 5.0e-12
    spot_v_fit_t_lo = 0.5e-9         # skip startup transient
    spot_rtol = 0.15                 # |v_meas - v_thiele|/v
    # Gate on the DRIVE-INDUCED deformation: peak |R(t under
    # drive) - R(t at J=0)| / R_start. The J=0 control cancels
    # the relaxation creep, so this isolates the SOT-induced
    # size change (the genuine rigid-Thiele assumption).
    spot_rigid_rtol = 0.20
    plot_path = 'docs/figures/validation/thiele_v_vs_R.png'
    ic_png = 'docs/figures/validation/thiele_initial.png'
    # ======================= End User Configuration =========================
    print('test_thiele_v_sot (Pham 2024 Fig. S49, Set B: '
          f'alpha={alpha}, gamma={gamma*1e-9:.1f} GHz/T, '
          f'D={D*1e3:.2f} mJ/m^2, Delta={Delta*1e9:.1f} nm, '
          f'J0={J0:.1e} A/m^2):')
    all_ok = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (1) Analytic gate: evaluate the production closed forms and
    # check them against the independently-derived expression,
    # monotonicity/saturation, and lambda^2 scaling.
    p = _make_set_B_params(
        D=D, alpha=alpha, gamma=gamma, J0=J0,
        nx=256, ny=256, a=default_params().a, dt=5.0e-14,
        R=80.0e-9, Delta=Delta, lambda_sq=0.0)
    H_DL = p.DL_SOT * J0
    x = np.array(R_over_Delta, dtype=float)
    R_arr = x * Delta
    v_sot = np.array(
        [sot_thiele_speed(R, Delta, p) for R in R_arr])
    # Independently-derived closed form (see module docstring):
    # v_SOT = (pi H_DL Delta gamma / 2 alpha) * x^2 / (x^2 + 1).
    v_sat = math.pi * H_DL * Delta * gamma / (2.0 * alpha)
    v_closed = v_sat * x * x / (x * x + 1.0)
    max_rel = float(np.max(np.abs(v_sot - v_closed)
                           / np.abs(v_closed)))
    cf_ok = max_rel < closed_form_rtol
    all_ok = all_ok and cf_ok
    print(f'  [analytic] v_SOT vs closed form: max rel.dev='
          f'{max_rel:.2e} [{"PASS" if cf_ok else "FAIL"} < '
          f'{closed_form_rtol:.0e}]')
    print(f'             v_SOT: {v_sot[0]:.1f} m/s (R/D=1) -> '
          f'{v_sot[-1]:.1f} m/s (R/D=5), saturates at '
          f'{v_sat:.1f} m/s')
    # Monotone increasing and bounded by the saturation speed.
    mono_ok = bool(np.all(np.diff(v_sot) > 0.0)
                   and np.all(v_sot < v_sat))
    all_ok = all_ok and mono_ok
    print(f'  [analytic] v_SOT monotone increasing & < v_sat: '
          f'[{"PASS" if mono_ok else "FAIL"}]')
    # TSH curves and lambda^2 linearity at fixed R/Delta.
    v_tsh = {
        lam: np.array(
            [tsh_thiele_speed(R, Delta, lam, p) for R in R_arr])
        for lam in lambda_sq_tsh}
    lam_lo, lam_hi = lambda_sq_tsh[0], lambda_sq_tsh[1]
    ratio = v_tsh[lam_hi] / v_tsh[lam_lo]
    lam_ok = bool(np.allclose(ratio, lam_hi / lam_lo,
                              rtol=1.0e-6))
    all_ok = all_ok and lam_ok
    print(f'  [analytic] v_TSH(lam) linear in lambda^2 '
          f'(ratio={float(np.mean(ratio)):.2f} vs '
          f'{lam_hi/lam_lo:.2f}): '
          f'[{"PASS" if lam_ok else "FAIL"}]')
    for lam in lambda_sq_tsh:
        print(f'             v_TSH(lam={lam*1e18:.0f} nm^2): '
              f'{v_tsh[lam][0]:+.2f} -> {v_tsh[lam][-1]:+.2f} '
              f'm/s')
    # The figure (S49 analytic curves + the spot-check
    # annotation) is rendered at the end, once the spot-check
    # result is known.
    spot_summary = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (2) Micromagnetic spot-check at one rigid R/Delta point.
    if do_llgs_spot:
        R_imp = spot_R_over_Delta * Delta
        pm = _make_set_B_params(
            D=D, alpha=alpha, gamma=gamma, J0=spot_J0,
            nx=spot_nx, ny=spot_ny, a=spot_a, dt=spot_dt,
            R=R_imp, Delta=Delta, lambda_sq=0.0)
        box = spot_nx * spot_a
        print(f'  [llgs] spot-check R/Delta={spot_R_over_Delta}'
              f' (R_imp={R_imp*1e9:.1f} nm), {spot_nx}x{spot_ny}'
              f' @ a={spot_a*1e9:.1f} nm (box {box*1e9:.0f} nm),'
              f' J0={spot_J0:.1e} A/m^2, Newell demag...')
        kernels = precompute_demag_kernels(
            pm, kind='newell', accuracy=spot_demag_accuracy,
            tol_conv=spot_demag_tol_conv)
        # Stash kernels on the RHS function so the rk4_step
        # closure can reach them without a closure variable.
        _rhs_saf_demag.kernels = kernels
        m_top, m_bot = saf_skyrmion(
            pm.nx, pm.ny, a=pm.a, R=R_imp, dw=Delta)
        plot_ic_2d(
            m=m_top, a=spot_a, out_path=ic_png,
            title=(f'Thiele IC (R/Delta={spot_R_over_Delta}, '
                   f'R={R_imp*1e9:.0f} nm, Delta={Delta*1e9:.0f}'
                   f' nm)'))
        # Relax at J = 0 (pulse is restored afterwards).
        pm.pulse = ConstantPulse(0.0)
        m_top, m_bot, conv, n_rel, _E, _tau = relax(
            m_top, m_bot, pm, kernels,
            max_steps=spot_relax_max_steps,
            alpha_relax=spot_relax_alpha,
            tol_torque=spot_relax_tol_torque,
            tol_dE=spot_relax_tol_dE,
            check_every=spot_relax_check_every)
        R_fit, Delta_fit, rms = _fit_bh_delta(
            m_top, spot_a, R_imp)
        R_start = 0.5 * skyrmion_diameter(
            m_top, spot_a, core_polarity=+1)
        print(f'         relax {n_rel} steps conv={conv}: '
              f'R_fit={R_fit*1e9:.1f} nm, '
              f'Delta_fit={Delta_fit*1e9:.1f} nm, '
              f'R/Delta_fit={R_fit/Delta_fit:.2f}')
        # Save the relaxed state so a J = 0 control can be driven
        # from the SAME configuration; subtracting the control
        # radius cancels the (J0-independent) relaxation creep
        # and isolates the drive-INDUCED deformation.
        m_top_relaxed = m_top.copy()
        m_bot_relaxed = m_bot.copy()
        n_steps = int(round(spot_drive_total / spot_dt))
        sample_every = int(round(spot_sample_dt / spot_dt))
        # Drive with DC SOT and track the centroid + radius.
        pm.pulse = ConstantPulse(spot_J0)
        cx0, cy0 = skyrmion_center(
            m_top, spot_a, core_polarity=+1)
        t_hist = [0.0]
        ux_hist = [cx0]
        R_hist = [R_start]
        ux = cx0
        cx_prev, cy_prev = cx0, cy0
        t = 0.0
        for step in range(1, n_steps + 1):
            m_top, m_bot = rk4_step(
                _rhs_saf_demag, m_top, m_bot, t, spot_dt, pm)
            t += spot_dt
            if step % sample_every == 0:
                cx_s, cy_s = skyrmion_center(
                    m_top, spot_a, core_polarity=+1)
                ddx = cx_s - cx_prev
                # Minimum-image unwrap of the PBC centroid jump.
                ddx -= box * round(ddx / box)
                ux += ddx
                cx_prev, cy_prev = cx_s, cy_s
                t_hist.append(t)
                ux_hist.append(ux)
                R_hist.append(0.5 * skyrmion_diameter(
                    m_top, spot_a, core_polarity=+1))
        v_thiele = sot_thiele_speed(R_fit, Delta_fit, pm)
        # J = 0 control from the relaxed state: same window, no
        # SOT -> captures the relaxation creep alone.
        pm.pulse = ConstantPulse(0.0)
        mc_top = m_top_relaxed
        mc_bot = m_bot_relaxed
        Rc_hist = [R_start]
        tc = 0.0
        for step in range(1, n_steps + 1):
            mc_top, mc_bot = rk4_step(
                _rhs_saf_demag, mc_top, mc_bot, tc, spot_dt, pm)
            tc += spot_dt
            if step % sample_every == 0:
                Rc_hist.append(0.5 * skyrmion_diameter(
                    mc_top, spot_a, core_polarity=+1))
        t_arr = np.array(t_hist)
        ux_arr = np.array(ux_hist)
        R_arr_t = np.array(R_hist)
        Rc_arr_t = np.array(Rc_hist)
        # Full-window radius variation (context only): includes
        # the relaxation creep.
        R_var = (R_arr_t.max() - R_arr_t.min()) / max(
            R_start, 1e-12)
        # GATED metric: drive-induced deformation = driven radius
        # minus the J=0 control radius (creep cancels), peak over
        # the window, normalized by R_start.
        defo = np.abs(R_arr_t - Rc_arr_t) / max(R_start, 1e-12)
        drive_deform = float(defo.max())
        rigid = drive_deform < spot_rigid_rtol
        mask = t_arr > spot_v_fit_t_lo
        if mask.sum() < 5:
            print('         FAIL: too few steady samples')
            all_ok = False
        else:
            vx = float(np.polyfit(
                t_arr[mask], ux_arr[mask], 1)[0])
            rel = abs(abs(vx) - abs(v_thiele)) / max(
                abs(v_thiele), 1e-9)
            ok = (rel < spot_rtol) and rigid
            all_ok = all_ok and ok
            spot_summary = {
                'rd': R_fit / Delta_fit, 'v_meas': abs(vx),
                'v_thiele': abs(v_thiele), 'rel': rel,
                'defo': drive_deform, 'J0': spot_J0}
            print(f'         v_meas={vx:+.1f} m/s, '
                  f'v_thiele={v_thiele:+.1f} m/s, '
                  f'rel.err={rel*100:.1f}% '
                  f'[{"PASS" if rel < spot_rtol else "FAIL"} < '
                  f'{spot_rtol*100:.0f}%]')
            print(f'         drive_deform={drive_deform*100:.0f}%'
                  f' (vs J=0 control) '
                  f'[{"PASS" if rigid else "FAIL"} rigid < '
                  f'{spot_rigid_rtol*100:.0f}%]; '
                  f'R_var(full)={R_var*100:.0f}% '
                  f'(creep, context only)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure: analytic v_SOT and v_TSH vs R/Delta evaluated at
    # the SPOT-CHECK drive (J0 = spot_J0), so the LLGS point sits
    # on the same v_SOT curve. (The analytic gate above is
    # independent and uses the paper's J0 = 8e11.)
    p_plot = _make_set_B_params(
        D=D, alpha=alpha, gamma=gamma, J0=spot_J0,
        nx=256, ny=256, a=default_params().a, dt=5.0e-14,
        R=80.0e-9, Delta=Delta, lambda_sq=0.0)
    v_sot_p = np.array(
        [sot_thiele_speed(R, Delta, p_plot) for R in R_arr])
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
    ax.plot(x, v_sot_p, 'o-', color='tab:orange',
            label=f'v_SOT analytic (J0={spot_J0:.0e})')
    for lam, col in zip(lambda_sq_tsh,
                        ('tab:blue', 'tab:purple')):
        v_tsh_p = np.abs(
            [tsh_thiele_speed(R, Delta, lam, p_plot)
             for R in R_arr])
        ax.plot(x, v_tsh_p, 's--', color=col,
                label=f'|v_TSH| (lam^2={lam*1e18:.0f} nm^2)')
    if spot_summary is not None:
        ax.plot([spot_summary['rd']], [spot_summary['v_meas']],
                'D', color='black', ms=10, zorder=5,
                label='LLGS spot-check (v_meas)')
        ax.plot([spot_summary['rd']], [spot_summary['v_thiele']],
                'x', color='red', ms=11, mew=2, zorder=6,
                label='Thiele(R_fit, Delta_fit)')
        ax.text(
            0.03, 0.97,
            (f'spot-check R/Delta={spot_summary["rd"]:.2f} '
             f'(relaxed):\n'
             f'v_meas={spot_summary["v_meas"]:.1f} vs Thiele '
             f'{spot_summary["v_thiele"]:.1f} m/s '
             f'({spot_summary["rel"]*100:.1f}%)\n'
             f'drive deformation '
             f'{spot_summary["defo"]*100:.0f}% (rigid)'),
            transform=ax.transAxes, fontsize=7.5, va='top',
            ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='0.6',
                      alpha=0.9))
    ax.set_xlabel(r'$R/\Delta$')
    ax.set_ylabel('|v| (m/s)')
    ax.set_title('Pham 2024 Fig. S49 Thiele speeds '
                 f'(Set B, D={D*1e3:.2f} mJ/m$^2$)')
    ax.legend(loc='center right', fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    print(f'  figure: {plot_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f'  status: {"PASS" if all_ok else "FAIL"}')
    return all_ok


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
