"""1D domain-wall profile benchmark for the deterministic simulator.

Relaxes a 180-degree domain wall on a quasi-1D strip (a tall,
one-wall-wide 2D lattice) using the PRODUCTION field and
integrator functions (`simulator.fields.effective_field`,
`simulator.integrator.llgs_rhs` + `rk4_step_single`), then
fits the equilibrium m_z(x) profile to the analytic tanh form

    m_z(x) = -tanh((x - x_0) / Delta),
    Delta  =  sqrt(A_ex / K_eff).

The fitted wall width must agree with the analytic Delta to
within a fraction of a percent on a well-resolved lattice
(Delta >> a). This is the cleanest test of the exchange +
anisotropy ingredients in isolation and a prerequisite for
the Bogdanov-Hubert skyrmion profile fit.

Production reuse
----------------
The wall is laid out along x on a (ny, nx, 3) lattice that is
uniform along y (so the 2D `effective_field` reduces to the
1D Laplacian + anisotropy). DMI is off (D = 0) and demag is
absent (`effective_field` carries no demag term). The two end
columns (x = 0, x = nx-1) are pinned to +-z each step as the
Dirichlet boundary that holds the single wall against the PBC
seam; the box is many Delta wide so the seam is decoupled
from the physical wall. Over-damped descent uses the
production `llgs_rhs` at alpha = 1 stepped by the production
`rk4_step_single`.

Functions
---------
main
    Run the relaxation, fit the wall, report pass/fail.
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
from src.simulator.fields import effective_field
from src.simulator.integrator import llgs_rhs, rk4_step_single
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


def _make_dw_params(A_ex, K_eff, Ms, nx, ny, a, dt):
    """Parameter namespace for the 1D wall: D = 0, no demag,
    K_eff anisotropy set directly (C_anis = 2 K_eff / Ms, no
    demag-fold since this path uses no demag)."""
    p = default_params()
    p.A_ex = float(A_ex)
    p.D = 0.0
    p.Ms = float(Ms)
    p.alpha = 1.0
    p.gamma = 1.760e11
    p.H_RKKY = 0.0
    p.nx = int(nx)
    p.ny = int(ny)
    p.a = float(a)
    p.dt = float(dt)
    p.H_ext = np.array([0.0, 0.0, 0.0])
    p.pulse = ConstantPulse(0.0)
    p.J_current = 0.0
    p.lambda_sq = 0.0
    _precompute(p)
    # No demag in this path: set the anisotropy prefactor
    # directly to the bare 2 K_eff / Ms (overriding the
    # demag-folded value _precompute stores).
    p.C_anis_top = 2.0 * float(K_eff) / float(Ms)
    p.C_anis_bot = 2.0 * float(K_eff) / float(Ms)
    p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
    return p


def main():
    """Run the 1D domain-wall profile validation.

    Relaxes a tanh Neel/Bloch wall started with a deliberately wrong
    width (pinned Dirichlet ends) using the production effective
    field and single-spin RK4 integrator, then fits the wall width
    and writes the IC figure. Passes when the fitted width matches
    the analytic Delta = sqrt(A/K_eff) within tolerance.

    Returns
    -------
    passed : bool
        True if the fitted wall width is within `rtol`.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run configuration. Choose parameters so that
    # Delta = sqrt(A_ex / K_eff) is well-resolved on the lattice.
    A_ex = 16.0e-12          # J/m
    K_eff = 1.0e5            # J/m^3 (Delta ~ 12.6 nm)
    Ms = 1.43e6              # A/m
    a = 1.0e-9               # m
    nx = 401                 # sites along the wall (x)
    ny = 8                   # uniform transverse direction
    dt = 5.0e-14             # s
    max_steps = 200_000
    tol_torque = 1.0e-7
    check_every = 1_000
    rtol = 5.0e-3
    ic_png = 'docs/figures/validation/dw_1d_initial.png'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Delta_an = math.sqrt(A_ex / K_eff)
    print(f'test_dw_profile_1d (production effective_field + '
          f'rk4_step_single):')
    print(f'  A_ex={A_ex:.2e} J/m, K_eff={K_eff:.2e} J/m^3, '
          f'Ms={Ms:.2e} A/m')
    print(f'  analytic Delta = sqrt(A/K_eff) = '
          f'{Delta_an*1e9:.4f} nm; lattice {nx}x{ny} at '
          f'a = {a*1e9:.1f} nm '
          f'(box = {(nx-1)*a*1e9:.0f} nm = '
          f'{((nx-1)*a)/Delta_an:.0f} Delta).')
    p = _make_dw_params(A_ex, K_eff, Ms, nx, ny, a, dt)
    print(f'  C_ex = {p.C_ex:.3e} T, C_anis = '
          f'{p.C_anis_top:.3e} T')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # IC: tanh wall along x with a deliberately wrong width, so
    # the relaxer must refine it. Uniform along y. In-plane
    # component along x (Bloch/Neel degenerate without DMI).
    x = (np.arange(nx) - (nx - 1) / 2.0) * a
    Delta_init = 2.0 * Delta_an
    mz_line = -np.tanh(x / Delta_init)
    mx_line = np.sqrt(np.clip(1.0 - mz_line ** 2, 0.0, 1.0))
    m_top = np.zeros((ny, nx, 3))
    m_top[:, :, 0] = mx_line[np.newaxis, :]
    m_top[:, :, 2] = mz_line[np.newaxis, :]

    # Pinned end columns (Dirichlet BC holding the single wall).
    def _pin_ends(m):
        m[:, 0, :] = np.array([0.0, 0.0, +1.0])
        m[:, -1, :] = np.array([0.0, 0.0, -1.0])
        return m
    m_top = _pin_ends(m_top)
    # Decoupled bottom layer (single FM): m_other = m_top, but
    # H_RKKY = 0 so it never couples in.
    os.makedirs(os.path.dirname(ic_png), exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=160)
    ax.plot(x * 1e9, m_top[ny // 2, :, 2], 'o-', color='C0',
            markersize=2, lw=0.8, label=r'$m_z$ (IC)')
    ax.plot(x * 1e9, m_top[ny // 2, :, 0], 's-', color='C3',
            markersize=2, lw=0.8, label=r'$m_x$ (IC)')
    ax.axhline(0, color='0.7', ls=':')
    ax.set_xlabel(r'$x$ (nm)')
    ax.set_ylabel('magnetisation')
    ax.set_title(f'1D DW IC: tanh, Delta_init = '
                 f'{Delta_init*1e9:.1f} nm')
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    fig.savefig(ic_png)
    plt.close(fig)
    print(f'  IC figure: {ic_png}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Over-damped descent via the PRODUCTION single-layer
    # stepper. The RHS uses the PRODUCTION effective_field
    # (exchange + anisotropy; D = 0 -> no DMI; H_ext = 0; no
    # demag) and PRODUCTION llgs_rhs. End columns are re-pinned
    # after every step.
    def rhs(m, p_, t_):
        H = effective_field(
            m, m, p_.C_ex, p_.C_dmi, p_.C_anis_top,
            p_.H_ext, p_.H_RKKY)
        return llgs_rhs(m, H, p_, t_)
    t = 0.0
    tau_max = float('inf')
    n_done = int(max_steps)
    for step in range(int(max_steps)):
        m_top = rk4_step_single(rhs, m_top, t, p.dt, p)
        m_top = _pin_ends(m_top)
        t += p.dt
        if (step + 1) % int(check_every) == 0:
            H = effective_field(
                m_top, m_top, p.C_ex, p.C_dmi, p.C_anis_top,
                p.H_ext, p.H_RKKY)
            mxH = np.cross(m_top, H)
            mxmxH = np.cross(m_top, mxH)
            # Exclude the pinned end columns from the torque
            # norm (they are held fixed by construction).
            tau_max = float(np.max(np.linalg.norm(
                mxmxH[:, 1:-1, :], axis=-1)))
            n_done = step + 1
            if tau_max < tol_torque:
                break
    print(f'  relaxation: {n_done} steps, final tau_max = '
          f'{tau_max:.2e} T')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit m_z(x) = -tanh((x - x_0)/Delta) on the interior at a
    # central y-row (exclude the pinned ends to avoid BC bias).
    row = ny // 2
    sel = slice(2, nx - 2)
    x_fit = x[sel]
    mz_fit = m_top[row, sel, 2]

    # Analytic 180-deg wall profile; x0 and Delta are fitted.
    def _model(xx, x0, Delta):
        return -np.tanh((xx - x0) / Delta)

    popt, _ = curve_fit(
        _model, x_fit, mz_fit, p0=(0.0, Delta_init),
        maxfev=20000)
    x0_fit, Delta_fit = float(popt[0]), float(popt[1])
    rel_err = abs(Delta_fit - Delta_an) / Delta_an
    print(f'  fit: x0 = {x0_fit*1e9:+.3f} nm, Delta = '
          f'{Delta_fit*1e9:.4f} nm (analytic '
          f'{Delta_an*1e9:.4f} nm, rel.err = '
          f'{rel_err*100:.3f} %)')
    passed = rel_err < rtol
    print(f'  status: {"PASS" if passed else "FAIL"} '
          f'(rtol = {rtol*100:.2f} %)')
    return passed


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
