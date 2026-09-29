"""Skyrmion-size temperature scaling in a dot (Tomasello 2018).

Reproduces the deterministic, scaled-parameter branch of

    R. Tomasello et al., "Origin of temperature and field
    dependence of magnetic skyrmion size in ultrathin
    nanodots," Phys. Rev. B 97, 060402(R) (2018), Fig. 1(b).

Temperature enters ONLY through the reduced magnetization
m(T) = M_s(T)/M_s(0), which scales the micromagnetic
parameters by Callen-Callen-type exponents obtained from
atomistic spin dynamics:

    M_s(T) = M_s(0) m,   A(T)  = A(0)  m^1.50,
    D(T)   = D(0)  m^1.50, K_u(T) = K_u(0) m^3.585.

For each temperature the (scaled) skyrmion is relaxed in a
confined circular dot at a fixed out-of-plane field opposing
the core, and its core radius R_sk (the m_z = 0 ring) is
measured. As T rises, the reduced DMI d(m) ~ m^-0.84 rises
toward the critical line d_c(Q) = (4/pi) sqrt(Q-1) and the
skyrmion expands -- strongly at H = 0, weakly at finite field.

This is the DETERMINISTIC (athermal, scaled-parameter) branch
that the paper compares to its analytical dashed curves in
Fig. 1(b). A separate finite-temperature stochastic scan
(`scan_skyrmion_size_thermal`) reproduces the mean +/- std
symbols.

Method / reuse
--------------
The confined-dot relaxation is the production no-demag free-BC
path, reused verbatim from the RT 2013 benchmark
(`test_confined_skyrmion_radius_rt2013`): demag is folded into
the effective anisotropy K_eff = K_u - 0.5 mu0 M_s^2 (the
paper's thin-film approximation, which is exactly the
simulator's K_eff convention), with the free-BC mask + RT/
mumax3 DMI edge condition.

Reference (digitized)
---------------------
T = 0 K set: Q(0) = 2.65, d(0) = 1.41, l_ex(0) = 9.4 nm, giving
M_s = 0.60 MA/m, A = 20 pJ/m, K_u = 0.60 MJ/m^3, D = 3.0
mJ/m^2; dot radius R_d = 141 nm (= 15 l_ex, from Fig. 3
insets); thickness 0.8 nm; cell 2.5 nm. m(T) digitized from
Fig. 2 (D(T)/D(0) = m^1.5). Fig. 1(b) diameters digitized for
the gate.

Acceptance
----------
The H = 0 expansion ratio R_sk(300 K)/R_sk(0) matches the
paper's analytical value (D_sk: 28 -> 92 nm, ratio ~ 3.3)
within `ratio_rtol`. The pointwise R_sk(T) and the H = 25/50
mT curves are reported (and overlaid on the digitized data)
but the gated scalar is the H = 0 ratio.

Functions
---------
main
    Sweep (H, T) with scaled parameters, relax in the dot,
    measure R_sk(T, H), gate the H = 0 expansion ratio,
    overlay the digitized Fig. 1(b).
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
from skyrmion_simulator.simulator.relaxation import relax
from skyrmion_simulator.phase_diagram.validation \
    .test_confined_skyrmion_radius_rt2013 import _make_rt_params, \
    _measure_Rs
from skyrmion_simulator.simulator.initial_conditions import skyrmion_profile
from skyrmion_simulator.simulator.lattice import disk_mask

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
# Digitized m(T) from Fig. 2 (D(T)/D(0) = m^1.5 -> m = (D/D0)^(2/3)).
_T_REF = np.array([0., 50., 100., 150., 200., 250., 300.])
_M_REF = np.array([1.000, 0.987, 0.973, 0.964, 0.953, 0.941,
                   0.928])
# Digitized Fig. 1(b) skyrmion DIAMETER D_sk (nm), analytical
# dashed curves, per field. The H = 0 curve is the gate target.
_DSK_H0 = np.array([28., 30., 34., 42., 52., 66., 92.])
_DSK_H25 = np.array([26., 27., 30., 33., 37., 41., 44.])
_DSK_H50 = np.array([24., 25., 26., 28., 29., 31., 32.])
# Callen-Callen scaling exponents (Tomasello 2018).
_ALPHA_A = 1.50          # A(T)   = A(0)   m^alpha
_BETA_D = 1.50           # D(T)   = D(0)   m^beta
_GAMMA_K = 3.585         # K_u(T) = K_u(0) m^gamma


def _m_of_T(T):
    """Reduced magnetization m(T) by interpolation of the
    digitized Fig. 2 table (valid 0-300 K)."""
    if T < _T_REF[0] or T > _T_REF[-1]:
        raise RuntimeError(
            f'_m_of_T: T = {T} K is outside the digitized '
            f'range [0, 300] K (m(T) was digitized only to '
            f'300 K).')
    return float(np.interp(T, _T_REF, _M_REF))


def main():
    # Sweep (H, T) with m(T)-scaled parameters, relax in the dot,
    # gate the H=0 expansion ratio, overlay the digitized data.
    # =========================== User Configuration =========================
    # Tomasello 2018 T = 0 absolute set (from Q, d, l_ex).
    Ms0 = 0.60e6             # A/m
    A0 = 20.0e-12            # J/m
    Ku0 = 0.60e6             # J/m^3 (uniaxial, before shape fold)
    D0 = 3.0e-3              # J/m^2
    R_dot = 141.0e-9         # m (= 15 l_ex, from Fig. 3 insets)
    gamma = 1.760e11         # rad/(s T)
    a = 2.5e-9               # m (paper cell)
    nx = 128                 # 320 nm box (dot R = 141 nm + 19)
    ny = 128
    dt = 5.0e-14
    # Out-of-plane fields opposing the core (core m_z = -1, so
    # the destabilizing/contracting field is +z), in Tesla.
    H_list = [0.0, 25.0e-3, 50.0e-3]
    T_list = [0., 50., 100., 150., 200., 250., 300.]
    R_init = 30.0e-9         # seed; dot + field relax to R_sk(T)
    # The near-critical T=300/H=0 point relaxes slowly (flat
    # energy landscape as d -> d_c); a large step budget lets it
    # fully converge. Fast points still break early on torque.
    max_steps = 500_000
    tol_torque = 1.0e-5
    check_every = 1_000
    # Gate: H = 0 expansion ratio R_sk(300)/R_sk(0) vs the paper
    # analytical (D_sk 28 -> 92 nm => ratio 3.29).
    ratio_ref = _DSK_H0[-1] / _DSK_H0[0]
    ratio_rtol = 0.30
    plot_path = ('docs/figures/validation/'
                 'tomasello_size_vs_T.png')
    # ======================= End User Configuration =========================
    half_mu0 = 0.5 * (4.0e-7 * math.pi)
    # T=0 critical DMI D_c = (4/pi) sqrt(A K_eff); reference for
    # the d -> d_c approach driving the expansion.
    Dc0 = (4.0 / math.pi) * math.sqrt(
        A0 * (Ku0 - half_mu0 * Ms0 * Ms0))
    print('test_skyrmion_size_vs_T_tomasello2018 (Fig. 1b, '
          'deterministic scaled-parameter branch):')
    print(f'  T=0: Ms={Ms0:.2e} A/m, A={A0*1e12:.0f} pJ/m, '
          f'Ku={Ku0:.2e} J/m^3, D={D0*1e3:.2f} mJ/m^2, '
          f'dot R={R_dot*1e9:.0f} nm, a={a*1e9:.1f} nm.')
    mask = disk_mask(nx=nx, ny=ny, a=a, R=R_dot)
    ix_c, iy_c = nx // 2, ny // 2
    refs = {0.0: _DSK_H0, 25.0e-3: _DSK_H25, 50.0e-3: _DSK_H50}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    results = {}
    for H in H_list:
        Rs_T = []
        print(f'  --- H_ext = {H*1e3:.0f} mT ---')
        for T in T_list:
            m = _m_of_T(T)
            Ms = Ms0 * m
            A = A0 * m ** _ALPHA_A
            D = D0 * m ** _BETA_D
            Ku = Ku0 * m ** _GAMMA_K
            K_eff = Ku - half_mu0 * Ms * Ms
            p = _make_rt_params(
                A_ex=A, D=D, K_eff=K_eff, Ms=Ms, alpha=0.3,
                gamma=gamma, nx=nx, ny=ny, a=a, dt=dt)
            p.H_ext = np.array([0.0, 0.0, float(H)])
            m_top0 = skyrmion_profile(
                nx=nx, ny=ny, a=a, R=R_init,
                dw=math.sqrt(A / K_eff), polarity=+1)
            m_top0[~mask, :] = np.array([0.0, 0.0, 1.0])
            # Single FM layer relaxed through the production
            # relax() (free-BC mask, no demag, over-damped).
            m_top, _mb, _conv, n_done, _E, tau_max = relax(
                m_top0, None, p, None, mask=mask,
                alpha_relax=1.0, tol_torque=tol_torque,
                max_steps=max_steps, check_every=check_every)
            R_s = _measure_Rs(m_top, ix_c, iy_c, a, mask)
            R_s_nm = float('nan') if R_s is None else R_s * 1e9
            Rs_T.append(R_s_nm)
            # Reduced DMI d(m); diagnostic, printed to track its
            # approach to the critical line as T rises.
            d_m = (D * math.sqrt(2.0 * A / (
                (4.0e-7 * math.pi) * Ms * Ms))) / A
            print(f'    T={T:5.0f} K  m={m:.3f}  '
                  f'K_eff={K_eff:.3e}  d={d_m:.3f}  '
                  f'R_sk={R_s_nm:6.1f} nm  (relax {n_done}, '
                  f'tau={tau_max:.1e})')
        results[H] = np.array(Rs_T, dtype=float)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Gate: H = 0 expansion ratio.
    Rs0 = results[0.0]
    if not (np.isfinite(Rs0[0]) and np.isfinite(Rs0[-1])
            and Rs0[0] > 0.0):
        print('  status: FAIL (no R_sk at H=0 endpoints)')
        return False
    ratio = Rs0[-1] / Rs0[0]
    rel = abs(ratio - ratio_ref) / ratio_ref
    passed = rel < ratio_rtol
    print('-' * 60)
    print(f'  H=0 expansion ratio R_sk(300)/R_sk(0) = '
          f'{ratio:.2f} (paper {ratio_ref:.2f}, rel.err '
          f'{rel*100:.0f}%) '
          f'[{"PASS" if passed else "FAIL"} < '
          f'{ratio_rtol*100:.0f}%]')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Overlay: simulated R_sk (diameter 2*R_sk) vs digitized.
    fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=160)
    cols = {0.0: 'k', 25.0e-3: 'tab:red', 50.0e-3: 'tab:green'}
    for H in H_list:
        c = cols[H]
        ax.plot(T_list, 2.0 * results[H], 'o-', color=c,
                label=f'sim H={H*1e3:.0f} mT')
        ax.plot(_T_REF, refs[H], '--', color=c, alpha=0.6,
                label=f'paper H={H*1e3:.0f} mT')
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$D_{sk}$ (nm)')
    ax.set_title('Skyrmion size vs T (Tomasello 2018 Fig. 1b, '
                 'deterministic)')
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    print(f'  figure: {plot_path}')
    print(f'  status: {"PASS" if passed else "FAIL"}')
    return passed


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
