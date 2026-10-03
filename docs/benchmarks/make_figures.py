"""Regenerate the benchmark figures for docs/benchmarks.tex.

Every figure is drawn with a SQUARE plotting area (set_box_aspect)
and NO grid, per the benchmarks document spec. The physics is the
production code in src/skyrmion_simulator/simulator,
src/skyrmion_simulator/phase_diagram and
src/skyrmion_simulator/stochastic_llgs; this script only re-wires the
per-benchmark driver (initial condition, relaxation/integration, measurement)
already validated by the corresponding validation test, importing
each test's parameter and measurement helpers so no physics is
re-implemented here.

Figures backed by saved data (NPZ / trajectory text) are re-plotted
directly; the cheap deterministic benchmarks (#1, #2, #7, #11, #12)
are re-run; #3 (SP#4) is re-run only when run with the --sp4 flag
(slow free-BC Newell demag build).

Functions
---------
main
    Generate the full figure set into docs/benchmarks/figs.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import os
import pathlib
import sys
# Third-party
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root to sys.path.
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Local (production)
from skyrmion_simulator.simulator.relaxation import relax
from skyrmion_simulator.phase_diagram.sweep import _build_ic
from skyrmion_simulator.simulator.demag import precompute_demag_kernels
from skyrmion_simulator.simulator.fields import effective_field
from skyrmion_simulator.simulator.initial_conditions import (
    saf_skyrmion, skyrmion_profile, uniform_state)
from skyrmion_simulator.simulator.integrator import llgs_rhs, rk4_step, \
    rk4_step_single
from skyrmion_simulator.simulator.lattice import disk_mask, rect_mask
from skyrmion_simulator.simulator.parameters import default_params
from skyrmion_simulator.simulator.pulses import ConstantPulse
from skyrmion_simulator.simulator.topological_torque import (
    sot_thiele_speed, tsh_thiele_speed)
from skyrmion_simulator.simulator.validation._helpers import (
    integrate_single_fm, make_single_fm_params, radial_profile,
    relax_single_fm)
# Local (validation-test param/measurement helpers; no physics)
from skyrmion_simulator.phase_diagram.validation \
    .test_confined_skyrmion_radius_rt2013 import _make_rt_params, _measure_Rs
from skyrmion_simulator.simulator.validation.test_dmi_standard_problem import (
    CO_DATA_DIR, _co_2d_params, _measure_rsk_along_radius)
from skyrmion_simulator.simulator.validation.test_dw_profile_1d import \
    _make_dw_params
from skyrmion_simulator.simulator.validation.test_mumag_sp4 import (
    _make_sp4_params, _relax_single_layer, _rk4_single_layer)
from skyrmion_simulator.simulator.validation.test_mumag_sp5 import (
    _make_sp5_params, _relax_vortex, _rk4_zhang_li)
from skyrmion_simulator.simulator.validation.test_thiele_v_sot import (
    _make_set_B_params, _rhs_saf_demag)
from skyrmion_simulator.stochastic_llgs.validation.test_skyrmion_arrhenius \
    import censored_tau
from skyrmion_simulator.stochastic_llgs.validation \
    .test_skyrmion_arrhenius_dtrend import _fit_dE

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira', ]
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
FIG_DIR = os.path.join(str(ROOT), 'docs', 'benchmarks', 'figs')
OUT_VAL = os.path.join(str(ROOT), 'output')
BDATA = os.path.join(OUT_VAL, 'benchmarks')
K_B = 1.380649e-23               # J/K
MEV = 1.602176634e-22            # J per meV


# -----------------------------------------------------------------------------
def _load_or_compute(name, compute):
    """Load cached benchmark data, else compute it and cache.

    The re-run benchmarks (#1, #2, #3, #7, #11, #12) persist their
    measured arrays to output/benchmarks/<name>.npz the first time
    they run, so subsequent figure regeneration loads instantly and
    never re-runs the relaxation/integration.

    Parameters
    ----------
    name : str
        Cache file stem (without extension).
    compute : callable
        Zero-argument callable returning a dict of array-likes to
        cache and plot.

    Returns
    -------
    data : dict
        The cached or freshly-computed arrays.
    """
    path = os.path.join(BDATA, f'{name}.npz')
    if os.path.exists(path):
        print(f'  [cache] {path}')
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    data = compute()
    os.makedirs(BDATA, exist_ok=True)
    np.savez_compressed(path, **data)
    print(f'  [saved] {path}')
    return data


# -----------------------------------------------------------------------------
def _square_ax(figsize=4.4):
    """Return a (fig, ax) with a square plotting area and no grid.

    Parameters
    ----------
    figsize : float, default=4.4
        Side length of the figure in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.grid(False)
    return fig, ax


# -----------------------------------------------------------------------------
def _finish(fig, ax, name):
    """Square the data box, tighten, and save `name` into FIG_DIR."""
    ax.set_box_aspect(1)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# -----------------------------------------------------------------------------
def _tol_band(ax, ref, tol_frac, axis='x', label=None):
    """Shade a +/- tol_frac tolerance band about a reference value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ref : float
        Reference value (data units).
    tol_frac : float
        Fractional half-width of the band.
    axis : {'x', 'y'}, default='x'
        Whether the band is vertical (x) or horizontal (y).
    label : {str, None}, default=None
        Legend label for the band.
    """
    lo, hi = ref * (1.0 - tol_frac), ref * (1.0 + tol_frac)
    span = ax.axvspan if axis == 'x' else ax.axhspan
    span(lo, hi, color='0.80', alpha=0.6, zorder=0, label=label)


# -----------------------------------------------------------------------------
def _config_strip(panels, outname, scale=12.0):
    """Render a row of geometry-faithful m_z config snapshots.

    Each panel uses EQUAL data aspect (the true sample shape) with a
    coolwarm m_z map and a sparse in-plane quiver, no grid, no ticks,
    and a single shared colorbar.

    Parameters
    ----------
    panels : list[dict]
        Each dict: m (numpy.ndarray(3d), (ny,nx,3)), a (float),
        mask ({numpy.ndarray(2d), None}), title (str).
    outname : str
        Output PNG file name (written into FIG_DIR).
    scale : float, default=24.0
        Quiver scale (larger = shorter arrows).
    """
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.1))
    if n == 1:
        axes = [axes]
    im = None
    for ax, pan in zip(axes, panels):
        m = np.asarray(pan['m'], dtype=float)
        a = float(pan['a'])
        ny, nx = m.shape[:2]
        mz = m[..., 2].copy()
        mask = pan.get('mask')
        if mask is not None:
            mz = np.where(np.asarray(mask, dtype=bool), mz, np.nan)
        ext = [0.0, nx * a * 1e9, 0.0, ny * a * 1e9]
        im = ax.imshow(mz, origin='lower', cmap='coolwarm',
                       vmin=-1.0, vmax=1.0, extent=ext)
        stx = max(1, nx // 12)
        sty = max(1, ny // 12)
        ys, xs = np.mgrid[0:ny:sty, 0:nx:stx]
        u = m[ys, xs, 0].astype(float)
        v = m[ys, xs, 1].astype(float)
        if mask is not None:
            outside = ~np.asarray(mask, dtype=bool)[ys, xs]
            u[outside] = np.nan
            v[outside] = np.nan
        ax.quiver((xs + 0.5) * a * 1e9, (ys + 0.5) * a * 1e9,
                  u, v, pivot='mid', scale=scale,
                  scale_units='width', width=0.011, color='k',
                  alpha=0.85)
        ax.set_title(pan['title'], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r'$m_z$', fontsize=9)
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, outname)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


# -----------------------------------------------------------------------------
def fig_co_dmi():
    """#1 Cortes-Ortuno: relaxed m_z(r) vs the CO ODE reference."""
    print('fig_co_dmi (#1)')

    def _compute():
        A_ex, D, Ms, K_eff = 13.0e-12, 3.0e-3, 0.86e6, 0.4e6
        a, nx, ny = 2.0e-9, 50, 50
        R_disk = 50.0e-9
        p = _co_2d_params(
            A_ex=A_ex, D=D, K_eff=K_eff, Ms=Ms, alpha=0.3,
            gamma=1.760e11, nx=nx, ny=ny, a=a, dt=5.0e-14)
        mask = disk_mask(nx=nx, ny=ny, a=a, R=R_disk)
        m = skyrmion_profile(
            nx=nx, ny=ny, a=a, R=15.0e-9, dw=5.7e-9, polarity=+1)
        m[~mask, :] = np.array([0.0, 0.0, 1.0])
        m_ic = m.copy()
        m, _mb, _c, _n, _E, _tau = relax(
            m, None, p, None, mask=mask, alpha_relax=1.0,
            tol_torque=1.0e-5, max_steps=200_000, check_every=1_000)
        r_sk = _measure_rsk_along_radius(m=m, mask=mask, a=a)
        r_sim, mz_sim = radial_profile(
            m[..., 2] * mask, a=a, n_bins=60)
        keep = np.isfinite(mz_sim) & (r_sim <= R_disk - a)
        ode = np.loadtxt(os.path.join(
            CO_DATA_DIR, 'result_2d_r-mz_interfacial_ODE.txt'))
        return {'r_sim': r_sim[keep] * 1e9, 'mz_sim': mz_sim[keep],
                'r_ode': ode[:, 0], 'mz_ode': -ode[:, 1],
                'r_sk_nm': r_sk * 1e9,
                'm_ic': m_ic.astype(np.float32),
                'm_final': m.astype(np.float32),
                'mask': mask, 'a_nm': a * 1e9}

    d = _load_or_compute('co_dmi', _compute)
    r_sim, mz_sim = d['r_sim'], d['mz_sim']
    r_ode, mz_ode = d['r_ode'], d['mz_ode']
    r_sk = float(d['r_sk_nm']) * 1e-9
    fig, ax = _square_ax()
    _tol_band(ax, 22.03, 0.05, 'x', r'ref $22.03\pm5\%$ nm')
    ax.plot(r_ode, mz_ode, '-', color='0.2', lw=2.0,
            label='CO ODE reference')
    ax.plot(r_sim, mz_sim, 'o', color='C0', ms=4.5,
            label='this work (relaxed)')
    ax.axvline(r_sk * 1e9, color='C3', ls='--', lw=1.2,
               label=rf'$r_{{sk}}={r_sk*1e9:.2f}$ nm')
    ax.axhline(0.0, color='0.7', ls=':', lw=0.8)
    ax.set_xlim(0, 50)
    ax.set_xlabel(r'$r$ (nm)')
    ax.set_ylabel(r'$m_z$')
    ax.set_title(rf'CO interfacial-DMI disc: $r_{{sk}}={r_sk*1e9:.2f}$ nm')
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    _finish(fig, ax, 'fig01_co_dmi.png')
    av = float(d['a_nm']) * 1e-9
    _config_strip([
        {'m': d['m_ic'], 'a': av, 'mask': d['mask'],
         'title': 'initial (seed $R=15$ nm)'},
        {'m': d['m_final'], 'a': av, 'mask': d['mask'],
         'title': rf'relaxed ($r_{{sk}}={r_sk*1e9:.1f}$ nm)'},
    ], 'fig01_co_config.png')


# -----------------------------------------------------------------------------
def _bh_ansatz(rr, R, dd):
    """Bogdanov-Hubert m_z(r) ansatz on a radial grid."""
    return np.cos(2.0 * np.arctan(np.exp(-(rr - R) / dd)))


# -----------------------------------------------------------------------------
def fig_bh_profile():
    """#2 Bogdanov-Hubert: relaxed m_z(r) + BH ansatz + Eq.18 R_s."""
    print('fig_bh_profile (#2)')

    def _compute():
        A_ex, K_eff, Ms, D = 16.0e-12, 510.0e3, 8.755e5, 3.0e-3
        a, nx, ny = 1.0e-9, 96, 96
        Delta = math.sqrt(A_ex / K_eff)
        Dc = (4.0 / math.pi) * math.sqrt(A_ex * K_eff)
        R_s_eq18 = Delta / math.sqrt(2.0 * (1.0 - D / Dc))
        p = make_single_fm_params(
            A_ex=A_ex, D=D, K_eff=K_eff, Ms=Ms, alpha=0.14,
            gamma=1.760e11, H_ext=np.array([0.0, 0.0, 0.0]),
            nx=nx, ny=ny, a=a, dt=2.0e-14, pulse=ConstantPulse(0.0))
        m0 = skyrmion_profile(
            nx=nx, ny=ny, a=a, R=10.0e-9, dw=5.6e-9, polarity=+1)
        m, _tau, _n = relax_single_fm(
            m_top=m0, p=p, alpha_relax=1.0, max_steps=200_000,
            tol_torque=1.0e-5, check_every=1_000)
        r, mz = radial_profile(m[..., 2], a=a, n_bins=64)
        sel = np.isfinite(mz)
        r, mz = r[sel], mz[sel]
        popt, _ = curve_fit(
            _bh_ansatz, r, mz, p0=(R_s_eq18, Delta),
            bounds=([0.0, 0.1e-9], [r[-1], 50.0e-9]), maxfev=20000)
        return {'r': r * 1e9, 'mz': mz, 'R_fit_nm': popt[0] * 1e9,
                'dd_fit_nm': popt[1] * 1e9,
                'R_eq18_nm': R_s_eq18 * 1e9,
                'm_ic': m0.astype(np.float32),
                'm_final': m.astype(np.float32), 'a_nm': a * 1e9}

    d = _load_or_compute('bh_profile', _compute)
    r, mz = d['r'], d['mz']
    R_fit, dd_fit = float(d['R_fit_nm']), float(d['dd_fit_nm'])
    R_s_eq18 = float(d['R_eq18_nm'])
    rr = np.linspace(0.0, r[-1], 400)
    fig, ax = _square_ax()
    _tol_band(ax, R_s_eq18, 0.10, 'x', r'Eq.18 $\pm10\%$')
    ax.plot(rr, _bh_ansatz(rr, R_fit, dd_fit), '-', color='0.2',
            lw=2.0, label='BH ansatz fit')
    ax.plot(r, mz, 'o', color='C0', ms=4.0,
            label='this work (relaxed)')
    ax.axvline(R_s_eq18, color='C3', ls='--', lw=1.0,
               label=rf'RT Eq.18 $R_s={R_s_eq18:.2f}$ nm')
    ax.axhline(0.0, color='0.7', ls=':', lw=0.8)
    ax.set_xlabel(r'$r$ (nm)')
    ax.set_ylabel(r'$m_z$')
    ax.set_title('RT 2013 isolated skyrmion ($D/D_c=0.825$)')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    _finish(fig, ax, 'fig02_bh_profile.png')
    av = float(d['a_nm']) * 1e-9
    _config_strip([
        {'m': d['m_ic'], 'a': av, 'mask': None,
         'title': 'initial (seed $R=10$ nm)'},
        {'m': d['m_final'], 'a': av, 'mask': None,
         'title': rf'relaxed ($R_s={R_fit:.1f}$ nm)'},
    ], 'fig02_bh_config.png')


# -----------------------------------------------------------------------------
def fig_confined_rt():
    """#7 RT 2013 Fig.4d: relaxed m_z(r) in a 50 nm dot."""
    print('fig_confined_rt (#7)')

    def _compute():
        A_ex, K_eff, Ms, D = 16.0e-12, 0.50e6, 1.1e6, 4.5e-3
        a, nx, ny, R_dot = 2.0e-9, 60, 60, 50.0e-9
        Delta = math.sqrt(A_ex / K_eff)
        p = _make_rt_params(
            A_ex=A_ex, D=D, K_eff=K_eff, Ms=Ms, alpha=0.3,
            gamma=1.760e11, nx=nx, ny=ny, a=a, dt=5.0e-14)
        mask = disk_mask(nx=nx, ny=ny, a=a, R=R_dot)
        m0 = skyrmion_profile(
            nx=nx, ny=ny, a=a, R=25.0e-9, dw=Delta, polarity=+1)
        m0[~mask, :] = np.array([0.0, 0.0, 1.0])
        m_ic = m0.copy()
        m, _mb, _c, _n, _E, _tau = relax(
            m0, None, p, None, mask=mask, alpha_relax=1.0,
            tol_torque=1.0e-5, max_steps=120_000, check_every=1_000)
        R_s = _measure_Rs(m, nx // 2, ny // 2, a, mask)
        r, mz = radial_profile(m[..., 2] * mask, a=a, n_bins=50)
        keep = np.isfinite(mz) & (r <= R_dot - a)
        return {'r': r[keep] * 1e9, 'mz': mz[keep],
                'R_s_nm': R_s * 1e9,
                'm_ic': m_ic.astype(np.float32),
                'm_final': m.astype(np.float32),
                'mask': mask, 'a_nm': a * 1e9}

    d = _load_or_compute('confined_rt', _compute)
    r, mz, R_s = d['r'], d['mz'], float(d['R_s_nm'])
    fig, ax = _square_ax()
    _tol_band(ax, 25.0, 0.25, 'x', r'RT $25\pm25\%$ nm')
    ax.plot(r, mz, 'o-', color='C0', ms=4.0, lw=1.0,
            label='this work (dot relax)')
    ax.axvline(R_s, color='C3', ls='--', lw=1.2,
               label=rf'$R_s={R_s:.1f}$ nm')
    ax.axvline(25.0, color='0.4', ls=':', lw=1.2,
               label='RT Fig.4d $\\approx$25 nm')
    ax.axhline(0.0, color='0.7', ls=':', lw=0.8)
    ax.set_xlabel(r'$r$ (nm)')
    ax.set_ylabel(r'$m_z$')
    ax.set_title('Confined skyrmion ($R_{dot}=50$ nm, $D/D_c=1.25$)')
    ax.legend(loc='lower right', frameon=False, fontsize=8)
    _finish(fig, ax, 'fig07_confined_rt.png')
    av = float(d['a_nm']) * 1e-9
    _config_strip([
        {'m': d['m_ic'], 'a': av, 'mask': d['mask'],
         'title': 'initial (seed $R=25$ nm)'},
        {'m': d['m_final'], 'a': av, 'mask': d['mask'],
         'title': rf'relaxed ($R_s={R_s:.1f}$ nm)'},
    ], 'fig07_confined_config.png')


# -----------------------------------------------------------------------------
def fig_dw_profile():
    """#11 1D domain wall: relaxed m_z(x) vs -tanh fit."""
    print('fig_dw_profile (#11)')

    def _compute():
        A_ex, K_eff, Ms = 16.0e-12, 1.0e5, 1.43e6
        a, nx, ny = 1.0e-9, 401, 8
        Delta_an = math.sqrt(A_ex / K_eff)
        p = _make_dw_params(A_ex, K_eff, Ms, nx, ny, a, 5.0e-14)
        x = (np.arange(nx) - (nx - 1) / 2.0) * a
        mz_line = -np.tanh(x / (2.0 * Delta_an))
        mx_line = np.sqrt(np.clip(1.0 - mz_line ** 2, 0.0, 1.0))
        m = np.zeros((ny, nx, 3))
        m[:, :, 0] = mx_line[np.newaxis, :]
        m[:, :, 2] = mz_line[np.newaxis, :]

        def _pin(mm):
            mm[:, 0, :] = np.array([0.0, 0.0, +1.0])
            mm[:, -1, :] = np.array([0.0, 0.0, -1.0])
            return mm

        m = _pin(m)
        m_ic = m.copy()

        def rhs(mm, p_, t_):
            H = effective_field(
                mm, mm, p_.C_ex, p_.C_dmi, p_.C_anis_top,
                p_.H_ext, p_.H_RKKY)
            return llgs_rhs(mm, H, p_, t_)

        t = 0.0
        for step in range(200_000):
            m = _pin(rk4_step_single(rhs, m, t, p.dt, p))
            t += p.dt
            if (step + 1) % 1_000 == 0:
                H = effective_field(
                    m, m, p.C_ex, p.C_dmi, p.C_anis_top,
                    p.H_ext, p.H_RKKY)
                tau = float(np.max(np.linalg.norm(np.cross(
                    m, np.cross(m, H))[:, 1:-1, :], axis=-1)))
                if tau < 1.0e-7:
                    break
        row, sl = ny // 2, slice(2, nx - 2)
        popt, _ = curve_fit(
            lambda xx, x0, dd: -np.tanh((xx - x0) / dd),
            x[sl], m[row, sl, 2], p0=(0.0, 2.0 * Delta_an),
            maxfev=20000)
        return {'x': x * 1e9, 'mz': m[row, :, 2],
                'x0_nm': popt[0] * 1e9, 'Delta_fit_nm': popt[1] * 1e9,
                'Delta_an_nm': Delta_an * 1e9,
                'm_ic': m_ic.astype(np.float32),
                'm_final': m.astype(np.float32), 'a_nm': a * 1e9}

    d = _load_or_compute('dw_profile', _compute)
    x, mz = d['x'], d['mz']
    x0, Delta_fit = float(d['x0_nm']), float(d['Delta_fit_nm'])
    Delta_an = float(d['Delta_an_nm'])
    xf = np.linspace(x[0], x[-1], 400)
    fig, ax = _square_ax()
    ax.plot(xf, -np.tanh((xf - x0) / Delta_fit), '-',
            color='0.2', lw=2.0, label=r'$-\tanh$ fit')
    ax.plot(x, mz, 'o', color='C0', ms=2.5,
            label=r'this work $m_z(x)$')
    ax.axhline(0.0, color='0.7', ls=':', lw=0.8)
    ax.set_xlabel(r'$x$ (nm)')
    ax.set_ylabel(r'$m_z$')
    ax.set_title(rf'1D wall: $\Delta={Delta_fit:.3f}$ nm '
                 rf'(analytic {Delta_an:.3f}, '
                 rf'{abs(Delta_fit-Delta_an)/Delta_an*100:.2f}\% '
                 rf'$<0.5\%$)')
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    _finish(fig, ax, 'fig11_dw_profile.png')
    av = float(d['a_nm']) * 1e-9
    _config_strip([
        {'m': d['m_ic'], 'a': av, 'mask': None,
         'title': r'initial ($\Delta_{init}=2\Delta$)'},
        {'m': d['m_final'], 'a': av, 'mask': None,
         'title': rf'relaxed ($\Delta={Delta_fit:.2f}$ nm)'},
    ], 'fig11_dw_config.png', scale=12.0)


# -----------------------------------------------------------------------------
def fig_fmr():
    """#12 Uniform-mode FMR: FFT spectrum vs analytic frequency."""
    print('fig_fmr (#12)')

    def _compute():
        A_ex, K_eff, Ms = 16.0e-12, 5.0e5, 8.0e5
        gamma, a, nx, ny = 1.760e11, 2.0e-9, 32, 32
        H_K = 2.0 * K_eff / Ms
        f_an = gamma * H_K / (2.0 * math.pi)
        p = make_single_fm_params(
            A_ex=A_ex, D=0.0, K_eff=K_eff, Ms=Ms, alpha=0.005,
            gamma=gamma, H_ext=np.array([0.0, 0.0, 0.0]),
            nx=nx, ny=ny, a=a, dt=1.0e-13, pulse=ConstantPulse(0.0))
        m0 = uniform_state(nx=nx, ny=ny, direction=np.array(
            [0.05, 0.0, math.sqrt(1.0 - 0.05 ** 2)]))
        times, trace = integrate_single_fm(
            m_top=m0, p=p, n_steps=50_000, sample_every=10)
        mx = np.mean(trace[..., 0], axis=(1, 2))
        mx = mx - np.mean(mx)
        spec = np.abs(np.fft.rfft(mx))
        freqs = np.fft.rfftfreq(
            mx.size, d=float(times[1] - times[0]))
        f_peak = float(freqs[int(np.argmax(spec[1:]) + 1)])
        return {'freqs_GHz': freqs * 1e-9,
                'spec': spec / spec[1:].max(),
                'f_an_GHz': f_an * 1e-9, 'f_peak_GHz': f_peak * 1e-9}

    d = _load_or_compute('fmr_spectrum', _compute)
    freqs, spec = d['freqs_GHz'], d['spec']
    f_an, f_peak = float(d['f_an_GHz']), float(d['f_peak_GHz'])
    fig, ax = _square_ax()
    _tol_band(ax, f_an, 0.05, 'x', r'analytic $\pm5\%$')
    ax.plot(freqs, spec, '-', color='C0', lw=1.5,
            label=r'$|\mathrm{FFT}\,m_x|$')
    ax.axvline(f_an, color='C3', ls='--', lw=1.2,
               label=rf'analytic {f_an:.2f} GHz')
    ax.axvline(f_peak, color='0.4', ls=':', lw=1.2,
               label=rf'peak {f_peak:.2f} GHz')
    ax.set_xlim(0, 80)
    ax.set_xlabel(r'$f$ (GHz)')
    ax.set_ylabel('spectral amplitude (norm.)')
    ax.set_title('Uniform-mode FMR')
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    _finish(fig, ax, 'fig12_fmr.png')


# -----------------------------------------------------------------------------
def fig_sp5_trajectory():
    """#4 muMAG SP#5: vortex-core displacement trajectory."""
    print('fig_sp5_trajectory (#4)')
    txt = os.path.join(
        str(ROOT), 'docs', 'figures', 'validation',
        'sp5_trajectory.txt')
    d = np.loadtxt(txt)
    Dx, Dy = d[:, 3] * 1e9, d[:, 4] * 1e9
    n_tail = max(1, Dx.size // 2)
    sx, sy = float(Dx[-n_tail:].mean()), float(Dy[-n_tail:].mean())
    dist = math.hypot(sx - (-1.2), sy - (-14.7))
    th = np.linspace(0.0, 2.0 * math.pi, 200)
    fig, ax = _square_ax()
    ax.plot(-1.2 + 3.0 * np.cos(th), -14.7 + 3.0 * np.sin(th), '-',
            color='0.6', lw=1.0, label='tol 3 nm')
    ax.plot(Dx, Dy, '-', color='C0', lw=0.9, alpha=0.7)
    ax.plot(sx, sy, 'o', color='C0', ms=8,
            label=rf'this work ({sx:.2f}, {sy:.2f})')
    ax.plot(-1.2, -14.7, '*', color='C3', ms=15,
            label='Najafi 2009 ($-1.2$, $-14.7$)')
    ax.axhline(0.0, color='0.7', ls=':', lw=0.8)
    ax.axvline(0.0, color='0.7', ls=':', lw=0.8)
    ax.set_xlabel(r'$\Delta x$ (nm)')
    ax.set_ylabel(r'$\Delta y$ (nm)')
    ax.set_title(rf'SP#5 vortex core ($\xi=0.05$, dist '
                 rf'{dist:.2f} nm $<3$)')
    ax.legend(loc='upper left', frameon=False, fontsize=8)
    _finish(fig, ax, 'fig04_sp5_trajectory.png')


# -----------------------------------------------------------------------------
def fig_sp5_config():
    """#4 muMAG SP#5: vortex configs (IC, relaxed, driven)."""
    print('fig_sp5_config (#4) -- SP#5 re-run for configs')

    def _compute():
        Lx, Ly, a = 100.0e-9, 100.0e-9, 2.5e-9
        pad = 1.4
        nx = int(round(pad * Lx / a))
        ny = int(round(pad * Ly / a))
        dt = 5.0e-14
        p = _make_sp5_params(alpha=0.1, nx=nx, ny=ny, a=a, dt=dt)
        mask = rect_mask(nx=nx, ny=ny, a=a, Lx=Lx, Ly=Ly)
        kernels = precompute_demag_kernels(
            p, kind='newell_freebc', accuracy=4.0, tol_conv=0.05)
        xs = (np.arange(nx) - (nx - 1) / 2.0) * a
        ys = (np.arange(ny) - (ny - 1) / 2.0) * a
        Xg, Yg = np.meshgrid(xs, ys)
        m = np.zeros((ny, nx, 3))
        vz = np.full_like(Xg, 10.0e-9)
        norm = np.sqrt(Yg ** 2 + Xg ** 2 + vz ** 2)
        m[..., 0] = -Yg / norm
        m[..., 1] = Xg / norm
        m[..., 2] = vz / norm
        m[~mask, :] = np.array([0.0, 0.0, 1.0])
        m_ic = m.copy()
        m, _t, _n = _relax_vortex(
            m, p, kernels, mask, max_steps=120_000,
            tol_torque=1.0e-4, check_every=1_000)
        m_relaxed = m.copy()
        n_drive = int(round(8.0e-9 / dt))
        for step in range(1, n_drive + 1):
            m = _rk4_zhang_li(
                m, p, kernels, mask, u_T=-72.17, beta=0.05, dt=dt)
        return {'m_ic': m_ic.astype(np.float32),
                'm_relaxed': m_relaxed.astype(np.float32),
                'm_final': m.astype(np.float32),
                'mask': mask, 'a_nm': a * 1e9}

    d = _load_or_compute('sp5_config', _compute)
    av = float(d['a_nm']) * 1e-9
    _config_strip([
        {'m': d['m_ic'], 'a': av, 'mask': d['mask'],
         'title': 'initial (Landau vortex)'},
        {'m': d['m_relaxed'], 'a': av, 'mask': d['mask'],
         'title': 'relaxed (pre-current)'},
        {'m': d['m_final'], 'a': av, 'mask': d['mask'],
         'title': 'driven ($t=8$ ns, STT)'},
    ], 'fig04_sp5_config.png', scale=13.0)


# -----------------------------------------------------------------------------
def fig_thiele():
    """#10 Pham S49: analytic v_SOT(R/Delta) and v_TSH curves."""
    print('fig_thiele (#10)')
    Delta, D, alpha, gamma, J0 = 24.5e-9, 0.62e-3, 0.216, 175.9e9, 8.0e11
    p = _make_set_B_params(
        D=D, alpha=alpha, gamma=gamma, J0=J0, nx=256, ny=256,
        a=2.0e-9, dt=5.0e-14, R=80.0e-9, Delta=Delta, lambda_sq=0.0)
    x = np.linspace(1.0, 5.0, 41)
    R_arr = x * Delta
    v_sot = np.array([sot_thiele_speed(R, Delta, p) for R in R_arr])
    v_t3 = np.array(
        [tsh_thiele_speed(R, Delta, 3.0e-18, p) for R in R_arr])
    v_t50 = np.array(
        [tsh_thiele_speed(R, Delta, 50.0e-18, p) for R in R_arr])
    fig, ax = _square_ax()
    ax.plot(x, v_sot, '-', color='C0', lw=2.0, label=r'$v_{SOT}$')
    ax.plot(x, np.abs(v_t50), '--', color='C3', lw=1.6,
            label=r'$|v_{TSH}|,\ \lambda^2=50$ nm$^2$')
    ax.plot(x, np.abs(v_t3), ':', color='C2', lw=1.6,
            label=r'$|v_{TSH}|,\ \lambda^2=3$ nm$^2$')
    ax.set_xlabel(r'$R/\Delta$')
    ax.set_ylabel(r'$v$ (m/s) at $J_0=8\times10^{11}$ A/m$^2$')
    ax.set_title('Thiele SOT / TSH speeds (Pham 2024 Fig. S49)')
    ax.legend(loc='center right', frameon=False, fontsize=9)
    _finish(fig, ax, 'fig10_thiele.png')


# -----------------------------------------------------------------------------
def fig_arrhenius():
    """#8 Rohart-Arrhenius: ln tau vs 1/T with the censored fit."""
    print('fig_arrhenius (#8)')
    z = np.load(os.path.join(
        OUT_VAL, 'stochastic_llgs', 'validation',
        'skyrmion_arrhenius_agg.npz'))
    T = z['T_sub']
    tc = z['t_collapse']
    t_max = float(z['n_drive']) * float(z['dt'])
    Ts, taus = [], []
    for tv in np.unique(T):
        sel = T == tv
        tau_hat, n_ev, n_tot = censored_tau(tc[sel], t_max)
        if n_ev >= 1 and np.isfinite(tau_hat):
            Ts.append(tv)
            taus.append(tau_hat)
    Ts, taus = np.array(Ts), np.array(taus)
    inv_T = 1.0 / Ts
    ln_tau = np.log(taus)
    A = np.vstack([inv_T, np.ones_like(inv_T)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, ln_tau, rcond=None)
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((ln_tau - pred) ** 2)
    ss_tot = np.sum((ln_tau - ln_tau.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    dE_meV = slope * K_B / MEV
    tau0_ns = math.exp(intercept) * 1e9
    fig, ax = _square_ax()
    xx = np.linspace(inv_T.min(), inv_T.max(), 100)
    ax.plot(xx * 1e3, (slope * xx + intercept), '-', color='0.2',
            lw=2.0, label=(rf'fit: $\Delta E={dE_meV:.1f}$ meV, '
                           rf'$\tau_0={tau0_ns:.2f}$ ns'))
    ax.plot(inv_T * 1e3, ln_tau, 'o', color='C0', ms=6,
            label='censored MLE')
    ax.set_xlabel(r'$1/T$ ($10^{-3}$ K$^{-1}$)')
    ax.set_ylabel(r'$\ln\,\tau$ ($\tau$ in s)')
    ax.set_title(rf'Skyrmion collapse Arrhenius ($R^2={r2:.3f}$)')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    _finish(fig, ax, 'fig08_arrhenius.png')
    print(f'    dE={dE_meV:.2f} meV tau0={tau0_ns:.3f} ns R2={r2:.4f}')


# -----------------------------------------------------------------------------
def fig_arrhenius_dtrend():
    """#8b Rohart Fig.5b: fitted collapse barrier vs DMI.

    Fits Delta_E per DMI value from the DMI-sweep aggregate with the
    same censored-MLE Arrhenius fit and selection thresholds as
    `test_skyrmion_arrhenius_dtrend`.
    """
    print('fig_arrhenius_dtrend (#8b)')
    n_events_min = 10
    n_T_min = 3
    r2_min = 0.85
    z = np.load(os.path.join(
        OUT_VAL, 'stochastic_llgs', 'validation',
        'skyrmion_arrhenius_dsweep_agg.npz'))
    t_max = float(z['n_drive']) * float(z['dt'])
    by_DT = {}
    for Dv, T, tc in zip(z['D'], z['T_sub'], z['t_collapse']):
        by_DT.setdefault(float(Dv), {}).setdefault(
            float(T), []).append(float(tc))
    D_used, dE_used = [], []
    for Dv in sorted(by_DT):
        dE_meV, _tau0, r2, _n_used = _fit_dE(
            by_DT[Dv], t_max, n_events_min, n_T_min, r2_min)
        if np.isfinite(dE_meV) and r2 >= r2_min:
            D_used.append(Dv * 1e3)
            dE_used.append(dE_meV)
    D = np.array(D_used)
    dE = np.array(dE_used)
    print('    ' + ', '.join(
        f'D={d:.2f}: {e:.1f} meV' for d, e in zip(D, dE)))
    fig, ax = _square_ax()
    ax.plot(D, dE, 'o-', color='C0', ms=7, lw=1.5)
    ax.set_xlabel(r'$D$ (mJ/m$^2$)')
    ax.set_ylabel(r'$\Delta E$ (meV)')
    ax.set_title('Collapse barrier vs DMI (Rohart Fig. 5b trend)')
    _finish(fig, ax, 'fig08b_dtrend.png')


# -----------------------------------------------------------------------------
def fig_size_det():
    """#9a Tomasello deterministic: skyrmion diameter vs T."""
    print('fig_size_det (#9a)')
    T = np.array([0, 50, 100, 150, 200, 250, 300], dtype=float)
    R0 = np.array([13.3, 14.9, 17.2, 19.0, 21.7, 25.9, 33.4])
    R25 = np.array([10.9, 12.0, 13.3, 14.3, 15.6, 17.1, 19.1])
    R50 = np.array([9.3, 10.2, 11.2, 11.8, 12.7, 13.8, 15.0])
    fig, ax = _square_ax()
    ax.plot(T, 2 * R0, 'o-', color='C0', ms=6, label='$H=0$')
    ax.plot(T, 2 * R25, 's-', color='C1', ms=6, label='$H=25$ mT')
    ax.plot(T, 2 * R50, '^-', color='C2', ms=6, label='$H=50$ mT')
    ratio = R0[-1] / R0[0]
    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'skyrmion diameter $2R_{sk}$ (nm)')
    ax.set_title(rf'Deterministic size vs $T$ '
                 rf'($H{{=}}0$ ratio {ratio:.2f}; paper 3.29)')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    _finish(fig, ax, 'fig09a_size_det.png')


# -----------------------------------------------------------------------------
def fig_size_thermal():
    """#9b Tomasello thermal: <2R_sk> +/- std vs T per field."""
    print('fig_size_thermal (#9b)')
    z = np.load(os.path.join(
        OUT_VAL, 'phase_diagram', 'validation',
        'skyrmion_size_thermal_agg.npz'))
    H, T = z['H'], z['T']
    Rm, Rs = z['R_sk_mean'], z['R_sk_std']
    fig, ax = _square_ax()
    styles = {0.0: ('C0', 'o', '$H=0$'),
              0.025: ('C1', 's', '$H=25$ mT'),
              0.05: ('C2', '^', '$H=50$ mT')}
    for hv in sorted(np.unique(H)):
        col, mk, lab = styles[round(hv, 3)]
        Ts = np.unique(T[H == hv])
        dia = np.array(
            [2 * Rm[(H == hv) & (T == t)].mean() * 1e9 for t in Ts])
        err = np.array(
            [2 * Rs[(H == hv) & (T == t)].mean() * 1e9 for t in Ts])
        ax.errorbar(Ts, dia, yerr=err, fmt=mk + '-', color=col,
                    ms=6, lw=1.2, capsize=3, label=lab)
    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$\langle 2R_{sk}\rangle$ (nm)')
    ax.set_title('Thermal size vs $T$ (Brown noise)')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    _finish(fig, ax, 'fig09b_size_thermal.png')


# -----------------------------------------------------------------------------
def fig_brown():
    """Supporting: Neel-Brown reversal time, sim vs closed form."""
    print('fig_brown (support)')
    z = np.load(os.path.join(
        OUT_VAL, 'stochastic_llgs', 'validation',
        'brown_reversal.npz'))
    ts, tb, delta = z['tau_sim'], z['tau_brown'], z['delta']
    fig, ax = _square_ax()
    lo = min(tb.min(), ts.min()) * 0.5
    hi = max(tb.max(), ts.max()) * 2.0
    ax.plot([lo, hi], [lo, hi], '-', color='0.5', lw=1.0,
            label='$y=x$')
    for i in range(delta.size):
        ax.plot(tb[i], ts[i], 'o', color='C0', ms=9)
        ax.annotate(rf'$\Delta={delta[i]:.0f}$',
                    (tb[i], ts[i]), fontsize=9,
                    textcoords='offset points', xytext=(8, -3))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$\tau_{Brown}$ (s)')
    ax.set_ylabel(r'$\tau_{sim}$ (s)')
    ax.set_title('Néel-Brown macrospin reversal')
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    _finish(fig, ax, 'figS_brown.png')


# -----------------------------------------------------------------------------
def fig_langevin():
    """Supporting: <m_z> vs Langevin function."""
    print('fig_langevin (support)')
    z = np.load(os.path.join(
        OUT_VAL, 'stochastic_llgs', 'validation', 'langevin.npz'))
    x = z['x_vals']
    mz_sim, mz_se = z['mz_sim'], z['mz_se']
    xx = np.linspace(0.05, float(np.max(x)) * 1.05, 200)
    lang = 1.0 / np.tanh(xx) - 1.0 / xx
    fig, ax = _square_ax()
    ax.plot(xx, lang, '-', color='0.2', lw=2.0,
            label=r'$L(x)=\coth x-1/x$')
    for j in range(x.shape[1]):
        ax.errorbar(x[:, j], mz_sim[:, j], yerr=mz_se[:, j],
                    fmt='o', ms=5, capsize=2, color=f'C{j}',
                    label=f'sim (field {j + 1})')
    ax.set_xlabel(r'$x=\mu B/k_B T$')
    ax.set_ylabel(r'$\langle m_z\rangle$')
    ax.set_title('Langevin macrospin')
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    _finish(fig, ax, 'figS_langevin.png')


# -----------------------------------------------------------------------------
def fig_equipartition():
    """Supporting: spin-wave mode-energy equipartition ratio."""
    print('fig_equipartition (support)')
    z = np.load(os.path.join(
        OUT_VAL, 'stochastic_llgs', 'validation',
        'equipartition.npz'))
    ratio = z['ratio']
    low = z['low_k_mask'].astype(bool)
    vals = ratio[low]
    vals = vals[np.isfinite(vals)]
    fig, ax = _square_ax()
    ax.hist(vals, bins=20, color='C0', alpha=0.8,
            edgecolor='0.3')
    ax.axvline(1.0, color='C3', ls='--', lw=1.5,
               label='equipartition ($k_BT$/mode)')
    ax.axvline(float(np.median(vals)), color='0.2', ls=':',
               lw=1.5, label=rf'median {np.median(vals):.2f}')
    ax.set_xlabel(r'$E_{sim}/k_B T$ per mode')
    ax.set_ylabel('count (low-$k$ modes)')
    ax.set_title('Spin-wave equipartition')
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    _finish(fig, ax, 'figS_equipartition.png')


# -----------------------------------------------------------------------------
def fig_sp4():
    """#3 muMAG SP#4: <m_alpha>(t) under Field 1 (slow re-run)."""
    print('fig_sp4 (#3) -- slow free-BC Newell demag run')

    def _compute():
        a, pad = 2.5e-9, 1.5
        Lx_mag, Ly_mag = 500.0e-9, 125.0e-9
        nx = int(round(pad * Lx_mag / a))
        ny = int(round(pad * Ly_mag / a))
        dt = 5.0e-14
        from skyrmion_simulator.simulator.lattice import rect_mask
        p = _make_sp4_params(alpha=0.02, nx=nx, ny=ny, a=a, dt=dt)
        mask = rect_mask(nx=nx, ny=ny, a=a, Lx=Lx_mag, Ly=Ly_mag)
        kernels = precompute_demag_kernels(
            p, kind='newell_freebc', accuracy=4.0, tol_conv=0.05)
        d111 = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
        m = np.zeros((ny, nx, 3))
        m[mask, :] = d111
        m[~mask, :] = np.array([0.0, 0.0, 1.0])
        m_ic = m.copy()
        for h_mag in [1.5, 1.0, 0.5, 0.25, 0.1, 0.0]:
            steps = 40_000 if h_mag > 0.0 else 200_000
            m, _t, _n = _relax_single_layer(
                m, p, kernels, mask, H_ext=h_mag * d111,
                max_steps=steps, tol_torque=1.0e-4,
                check_every=1_000, tag=f'sat |H|={h_mag:.2f}')
        m_sstate = m.copy()
        p.H_ext = np.array([-24.6e-3, +4.3e-3, 0.0])
        p.alpha = 0.02
        p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
        drive_n = int(round(1.0e-9 / dt))
        sample_every = int(round(5.0e-12 / dt))
        w = mask[..., np.newaxis].astype(float)
        cnt = float(mask.sum())
        times_ps = [0.0]
        avg = [(m * w).sum(axis=(0, 1)) / cnt]
        t = 0.0
        for step in range(1, drive_n + 1):
            m = _rk4_single_layer(m, p, kernels, mask, t, dt)
            t += dt
            if step % sample_every == 0:
                times_ps.append(t * 1e12)
                avg.append((m * w).sum(axis=(0, 1)) / cnt)
        times_ps = np.array(times_ps)
        avg = np.array(avg)
        mx = avg[:, 0]
        cr = np.where(np.diff(np.sign(mx)) != 0)[0]
        i0 = int(cr[0])
        t_cross = times_ps[i0] - mx[i0] * (
            times_ps[i0 + 1] - times_ps[i0]) \
            / (mx[i0 + 1] - mx[i0])
        return {'times_ps': times_ps, 'avg': avg,
                't_cross_ps': t_cross,
                'm_ic': m_ic.astype(np.float32),
                'm_sstate': m_sstate.astype(np.float32),
                'm_final': m.astype(np.float32),
                'mask': mask, 'a_nm': a * 1e9}

    d = _load_or_compute('sp4_trajectory', _compute)
    times_ps, avg = d['times_ps'], d['avg']
    t_cross = float(d['t_cross_ps'])
    fig, ax = _square_ax()
    _tol_band(ax, 136.0, 0.10, 'x', r'NIST $136\pm10\%$ ps')
    ax.plot(times_ps, avg[:, 0], '-', color='C0', label=r'$\langle m_x\rangle$')
    ax.plot(times_ps, avg[:, 1], '-', color='C1', label=r'$\langle m_y\rangle$')
    ax.plot(times_ps, avg[:, 2], '-', color='C2', label=r'$\langle m_z\rangle$')
    ax.axhline(0.0, color='0.7', ls=':', lw=0.8)
    ax.axvline(136.0, color='0.2', ls='--', lw=1.2,
               label='NIST ref 136 ps')
    ax.axvline(t_cross, color='C3', ls=':', lw=1.2,
               label=rf'this work {t_cross:.1f} ps')
    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'$\langle m\rangle$')
    ax.set_title('NIST muMAG SP#4 (Field 1)')
    ax.legend(loc='upper right', frameon=False, fontsize=8)
    _finish(fig, ax, 'fig03_sp4.png')
    av = float(d['a_nm']) * 1e-9
    _config_strip([
        {'m': d['m_ic'], 'a': av, 'mask': d['mask'],
         'title': 'initial ($[1,1,1]$ uniform)'},
        {'m': d['m_sstate'], 'a': av, 'mask': d['mask'],
         'title': 'S-state (relaxed)'},
        {'m': d['m_final'], 'a': av, 'mask': d['mask'],
         'title': rf'$t=1$ ns (Field 1)'},
    ], 'fig03_sp4_config.png', scale=13.0)


# -----------------------------------------------------------------------------
def fig_arrhenius_config():
    """#8 collapse configs: metastable sk -> shrinking -> FM."""
    print('fig_arrhenius_config (#8)')

    def _compute():
        A, K, Ms, D = 16.0e-12, 0.50e6, 1.1e6, 3.0e-3
        a, nx, ny = 2.0e-9, 128, 128
        Delta = math.sqrt(A / K)
        p = make_single_fm_params(
            A_ex=A, D=D, K_eff=K, Ms=Ms, alpha=1.0,
            gamma=1.760e11, H_ext=np.array([0.0, 0.0, 0.0]),
            nx=nx, ny=ny, a=a, dt=2.0e-14, pulse=ConstantPulse(0.0))
        m = skyrmion_profile(
            nx=nx, ny=ny, a=a, R=10.0e-9, dw=Delta, polarity=+1)
        m, _t, _n = relax_single_fm(
            m_top=m, p=p, alpha_relax=1.0, max_steps=120_000,
            tol_torque=1.0e-5, check_every=1_000)
        m_ic = m.copy()
        # Supercritical destabilising field -> deterministic
        # athermal collapse (illustrates the path to FM).
        p.H_ext = np.array([0.0, 0.0, 0.30])
        p.alpha = 1.0
        p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)

        def rhs(mm, p_, t_):
            H = effective_field(
                mm, mm, p_.C_ex, p_.C_dmi, p_.C_anis_top,
                p_.H_ext, p_.H_RKKY)
            return llgs_rhs(mm, H, p_, t_)

        m_mid = None
        t = 0.0
        for step in range(300_000):
            m = rk4_step_single(rhs, m, t, p.dt, p)
            t += p.dt
            if step % 200 == 0:
                mzbar = float(m[..., 2].mean())
                if m_mid is None and mzbar > 0.4:
                    m_mid = m.copy()
                if mzbar > 0.985:
                    break
        if m_mid is None:
            m_mid = m.copy()
        return {'m_ic': m_ic.astype(np.float32),
                'm_mid': m_mid.astype(np.float32),
                'm_final': m.astype(np.float32), 'a_nm': a * 1e9}

    d = _load_or_compute('arrhenius_config', _compute)
    av = float(d['a_nm']) * 1e-9
    _config_strip([
        {'m': d['m_ic'], 'a': av, 'mask': None,
         'title': 'metastable skyrmion'},
        {'m': d['m_mid'], 'a': av, 'mask': None,
         'title': 'collapsing'},
        {'m': d['m_final'], 'a': av, 'mask': None,
         'title': 'collapsed (FM)'},
    ], 'fig08_collapse_config.png', scale=14.0)


# -----------------------------------------------------------------------------
def fig_tomasello_config():
    """#9 configs: deterministic relaxed skyrmion at T=0 vs 300 K."""
    print('fig_tomasello_config (#9)')

    def _compute():
        mu0 = 4.0e-7 * math.pi
        T_ref = np.array([0, 50, 100, 150, 200, 250, 300], float)
        m_ref = np.array(
            [1.000, 0.987, 0.973, 0.964, 0.953, 0.941, 0.928])
        a, nx, ny, R_d = 2.5e-9, 128, 128, 141.0e-9
        mask = disk_mask(nx=nx, ny=ny, a=a, R=R_d)
        out = {}
        for T in (0.0, 300.0):
            mm = float(np.interp(T, T_ref, m_ref))
            Ms = 0.60e6 * mm
            A = 20.0e-12 * mm ** 1.5
            D = 3.0e-3 * mm ** 1.5
            Ku = 0.60e6 * mm ** 3.585
            K_eff = Ku - 0.5 * mu0 * Ms * Ms
            p = make_single_fm_params(
                A_ex=A, D=D, K_eff=K_eff, Ms=Ms, alpha=1.0,
                gamma=1.760e11, H_ext=np.array([0.0, 0.0, 0.0]),
                nx=nx, ny=ny, a=a, dt=5.0e-14,
                pulse=ConstantPulse(0.0))
            m = skyrmion_profile(
                nx=nx, ny=ny, a=a, R=20.0e-9,
                dw=math.sqrt(A / max(K_eff, 1.0)), polarity=+1)
            m[~mask, :] = np.array([0.0, 0.0, 1.0])
            m, _mb, _c, _n, _E, _tau = relax(
                m, None, p, None, mask=mask, alpha_relax=1.0,
                tol_torque=1.0e-5, max_steps=500_000,
                check_every=1_000)
            out[f'm_T{int(T)}'] = m.astype(np.float32)
        out['mask'] = mask
        out['a_nm'] = a * 1e9
        return out

    d = _load_or_compute('tomasello_config', _compute)
    av = float(d['a_nm']) * 1e-9
    _config_strip([
        {'m': d['m_T0'], 'a': av, 'mask': d['mask'],
         'title': '$T=0$ K ($H=0$)'},
        {'m': d['m_T300'], 'a': av, 'mask': d['mask'],
         'title': '$T=300$ K ($H=0$)'},
    ], 'fig09_tomasello_config.png', scale=13.0)


# -----------------------------------------------------------------------------
def fig_thiele_config():
    """#10 configs: relaxed SAF skyrmion -> DC-SOT driven."""
    print('fig_thiele_config (#10) -- SAF relax + drive')

    def _compute():
        Delta, D, alpha, gamma = 24.5e-9, 0.62e-3, 0.216, 175.9e9
        nx, ny, a, dt = 160, 160, 5.0e-9, 5.0e-14
        J0 = 1.0e11
        R_imp = 1.5 * Delta
        pm = _make_set_B_params(
            D=D, alpha=alpha, gamma=gamma, J0=J0, nx=nx, ny=ny,
            a=a, dt=dt, R=R_imp, Delta=Delta, lambda_sq=0.0)
        kernels = precompute_demag_kernels(
            pm, kind='newell', accuracy=4.0, tol_conv=0.05)
        _rhs_saf_demag.kernels = kernels
        m_top, m_bot = saf_skyrmion(nx, ny, a, R=R_imp, dw=Delta)
        m_ic = m_top.copy()
        pm.pulse = ConstantPulse(0.0)
        m_top, m_bot, _c, _n, _E, _tau = relax(
            m_top, m_bot, pm, kernels, max_steps=120_000,
            alpha_relax=1.0, tol_torque=1.0e-5, tol_dE=1.0e-8,
            check_every=1_000)
        m_relaxed = m_top.copy()
        pm.pulse = ConstantPulse(J0)
        n_steps = int(round(1.5e-9 / dt))
        t = 0.0
        for step in range(1, n_steps + 1):
            m_top, m_bot = rk4_step(
                _rhs_saf_demag, m_top, m_bot, t, dt, pm)
            t += dt
        return {'m_ic': m_ic.astype(np.float32),
                'm_relaxed': m_relaxed.astype(np.float32),
                'm_final': m_top.astype(np.float32), 'a_nm': a * 1e9}

    d = _load_or_compute('thiele_config', _compute)
    av = float(d['a_nm']) * 1e-9
    _config_strip([
        {'m': d['m_ic'], 'a': av, 'mask': None,
         'title': r'initial ($R/\Delta=1.5$ seed)'},
        {'m': d['m_relaxed'], 'a': av, 'mask': None,
         'title': 'relaxed (top layer)'},
        {'m': d['m_final'], 'a': av, 'mask': None,
         'title': 'DC-SOT driven ($t=1.5$ ns)'},
    ], 'fig10_thiele_config.png', scale=13.0)


# -----------------------------------------------------------------------------
def fig_gungordu_config():
    """#5 candidate initial textures (the four sweep ansaetze)."""
    print('fig_gungordu_config (#5)')

    def _compute():
        p = default_params()
        p.nx, p.ny, p.a = 64, 64, 5.0e-9
        p.A_ex, p.D, p.Ms = 16.0e-12, 1.5e-3, 875500.0
        p.H_ext = np.array([0.0, 0.0, 0.05])
        out = {}
        for name in ('fm_par', 'stripe', 'sk_lattice', 'sq_lattice'):
            m_top, _mb = _build_ic(name, None, p)
            out[f'm_{name}'] = m_top.astype(np.float32)
        out['a_nm'] = p.a * 1e9
        return out

    d = _load_or_compute('gungordu_config', _compute)
    av = float(d['a_nm']) * 1e-9
    _config_strip([
        {'m': d['m_fm_par'], 'a': av, 'mask': None,
         'title': 'FM (fm_par)'},
        {'m': d['m_stripe'], 'a': av, 'mask': None,
         'title': 'spiral (stripe)'},
        {'m': d['m_sk_lattice'], 'a': av, 'mask': None,
         'title': 'SkX (sk_lattice)'},
        {'m': d['m_sq_lattice'], 'a': av, 'mask': None,
         'title': 'SC (sq_lattice)'},
    ], 'fig05_gungordu_config.png', scale=14.0)


# =============================================================================
def main():
    """Generate the benchmark figure set.

    With no arguments, regenerates every figure (re-running and
    caching the deterministic benchmarks). Pass space-separated
    keys to regenerate a subset.
    """
    only = [a for a in sys.argv[1:] if not a.startswith('--')]
    figs = {
        'co': fig_co_dmi, 'bh': fig_bh_profile,
        'confined': fig_confined_rt, 'dw': fig_dw_profile,
        'fmr': fig_fmr, 'sp5': fig_sp5_trajectory,
        'thiele': fig_thiele, 'arrhenius': fig_arrhenius,
        'dtrend': fig_arrhenius_dtrend, 'size_det': fig_size_det,
        'size_thermal': fig_size_thermal, 'brown': fig_brown,
        'langevin': fig_langevin, 'equip': fig_equipartition,
        'sp4': fig_sp4, 'sp5cfg': fig_sp5_config,
        'collapse': fig_arrhenius_config,
        'tomcfg': fig_tomasello_config,
        'thielecfg': fig_thiele_config,
        'gungcfg': fig_gungordu_config,
    }
    keys = only if only else list(figs.keys())
    for key in keys:
        figs[key]()
    print('done.')


# =============================================================================
if __name__ == '__main__':
    main()
