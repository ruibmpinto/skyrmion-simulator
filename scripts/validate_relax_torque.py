"""Relaxation-torque diagnostic / validator (Python reference + plots).

Tests whether the elevated relax `tau_max` floor on a larger box is a
max-norm artifact of a well-relaxed skyrmion, or a genuinely
under-relaxed skyrmion. The four checks:

1. Torque map + histogram: where do the highest tangential-torque sites
   sit -- delocalized in the ferromagnetic background (artifact) or in a
   ring at the domain wall (real under-relaxation)?
2. Size / energy convergence vs step: do D1/D2 (LCC) and the total
   energy plateau while `tau_max` floors? If so the skyrmion is
   physically relaxed regardless of the torque gate.
3. Box-size scaling: the plateau `tau_max` should rise with the number
   of sites N while the bulk statistic (rms/median) and the size stay
   ~constant if the floor is a max-of-N artifact.
4. 256x256 positive control: `tau_max` floors ~3.4e-5 (matches the
   stored S47 metadata) and D ~= 186.5 nm.

This mirrors the C++ `validate_relax_torque` (same NPZ schema) and adds
the plots. Both write `box_*.npz` to the same directory; the plotter
reads every NPZ there (C++ and Python). Configure via the variables at
the top of `main()`; no argparse.

Run with:
    python -m scripts.validate_relax_torque
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
# Local
from src.phase_diagram.params_helper import make_params
from src.phase_diagram.relaxation import relax
from src.simulator.demag import precompute_demag_kernels
from src.simulator.energy import total_energy
from src.simulator.fields import effective_field_demag_pair
from src.simulator.initial_conditions import saf_skyrmion
from src.stochastic_llgs.diagnostics import (
    skyrmion_center_lcc_pbc,
    skyrmion_ellipse_lcc,
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


def _torque_field(m, h):
    """Per-site tangential torque |m x (m x H)| (shape (ny, nx))."""
    tau = np.cross(m, np.cross(m, h))
    return np.sqrt(np.sum(tau * tau, axis=-1))


# -----------------------------------------------------------------------------
def _diagnose_box(cfg, nx, ny):
    """Chunk-relax one box, sampling torque/size/energy vs step.

    Returns
    -------
    out : dict
        Convergence series and the final torque field / m_z, matching
        the C++ validator NPZ schema.
    """
    p = make_params(nx=int(nx), ny=int(ny))
    p.dt = float(cfg['dt'])
    kernels = precompute_demag_kernels(
        p, kind=str(cfg['demag_kind']),
        accuracy=float(cfg['demag_accuracy']),
        tol_conv=float(cfg['demag_tol_conv']))
    m_top, m_bot = saf_skyrmion(
        p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw)
    chunk = int(cfg['chunk'])
    n_chunks = int(cfg['n_chunks'])
    steps, t_max, t_p99, t_med, t_rms = [], [], [], [], []
    d1s, d2s, energies, rmax = [], [], [], []
    step = 0
    tfield = None
    for _ in range(n_chunks):
        # Relax `chunk` more steps; tol=0 so it never early-stops.
        m_top, m_bot, _c, _n, _E, _tau = relax(
            m_top, m_bot, p, kernels, max_steps=chunk,
            alpha_relax=float(cfg['alpha_relax']),
            tol_torque=0.0, tol_dE=0.0, check_every=chunk,
            print_every=0, mask=None)
        step += chunk
        h_top, _h_bot = effective_field_demag_pair(
            m_top, m_bot, p, kernels, mask=None)
        tfield = _torque_field(m_top, h_top)
        flat = tfield.ravel()
        t_max.append(float(flat.max()))
        t_p99.append(float(np.percentile(flat, 99.0)))
        t_med.append(float(np.median(flat)))
        t_rms.append(float(np.sqrt(np.mean(flat * flat))))
        d1, d2, _th = skyrmion_ellipse_lcc(m_top, p.a, core_polarity=1)
        d1s.append(float(d1))
        d2s.append(float(d2))
        energies.append(float(total_energy(m_top, m_bot, p, kernels)))
        # Distance of the max-torque site from the skyrmion core.
        k = int(np.argmax(flat))
        cx, cy = skyrmion_center_lcc_pbc(m_top, p.a, core_polarity=1)
        rmax.append(float(np.hypot(
            (k % p.nx) * p.a - cx, (k // p.nx) * p.a - cy)))
        steps.append(step)
        print(f'  {nx}x{ny} step {step}: tau_max={t_max[-1]:.2e} '
              f'rms={t_rms[-1]:.2e} D1={d1*1e9:.1f} D2={d2*1e9:.1f}',
              flush=True)
    return {
        'nx': p.nx, 'ny': p.ny, 'a': p.a, 'dt': p.dt,
        'steps': np.array(steps, dtype=float),
        'tau_max': np.array(t_max), 'tau_p99': np.array(t_p99),
        'tau_median': np.array(t_med), 'tau_rms': np.array(t_rms),
        'D1': np.array(d1s), 'D2': np.array(d2s),
        'E': np.array(energies),
        'rmax_from_core': np.array(rmax),
        'torque_field': tfield.astype(np.float32),
        'mz': m_top[..., 2].astype(np.float32),
    }


# -----------------------------------------------------------------------------
def _plot_map_hist(d, out_path):
    """Torque heatmap (log) with the skyrmion outline + histogram."""
    a = float(d['a'])
    tf = np.asarray(d['torque_field'], dtype=float)
    mz = np.asarray(d['mz'], dtype=float)
    ny, nx = tf.shape
    x_nm = np.arange(nx) * a * 1e9
    y_nm = np.arange(ny) * a * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    floor = max(float(np.median(tf)) * 1e-2, 1e-9)
    im = axes[0].pcolormesh(
        x_nm, y_nm, np.maximum(tf, floor),
        norm=mcolors.LogNorm(), cmap='inferno', shading='nearest')
    # mz=0 contour marks the skyrmion domain wall.
    axes[0].contour(x_nm, y_nm, mz, levels=[0.0], colors='cyan',
                    linewidths=1.0)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlabel(r'$x$ (nm)')
    axes[0].set_ylabel(r'$y$ (nm)')
    axes[0].set_title(r'$|m\times(m\times H)|$ (T); cyan = DW')
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[1].hist(tf.ravel(), bins=np.logspace(
        np.log10(floor), np.log10(tf.max() + 1e-30), 60))
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].axvline(float(np.median(tf)), color='g', ls='--',
                    label='median')
    axes[1].axvline(tf.max(), color='r', ls='--', label='max')
    axes[1].set_xlabel(r'tangential torque (T)')
    axes[1].set_ylabel('site count')
    axes[1].set_title('torque distribution')
    axes[1].legend()
    fig.suptitle(f'box {d["nx"]}x{d["ny"]}')
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_convergence(d, out_path):
    """D1/D2, energy, and torque percentiles vs relax step."""
    s = np.asarray(d['steps'], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.5))
    axes[0].plot(s, np.asarray(d['D1']) * 1e9, 'o-', label=r'$D_1$')
    axes[0].plot(s, np.asarray(d['D2']) * 1e9, 's-', label=r'$D_2$')
    axes[0].set_ylabel('diameter (nm)')
    axes[0].legend()
    axes[0].set_title('size vs step')
    axes[1].plot(s, np.asarray(d['E']), 'o-')
    axes[1].set_ylabel('energy (J)')
    axes[1].set_title('energy vs step')
    axes[2].semilogy(s, np.asarray(d['tau_max']), 'o-', label='max')
    axes[2].semilogy(s, np.asarray(d['tau_p99']), '^-', label='p99')
    axes[2].semilogy(s, np.asarray(d['tau_median']), 's-', label='median')
    axes[2].semilogy(s, np.asarray(d['tau_rms']), 'd-', label='rms')
    axes[2].set_ylabel('tangential torque (T)')
    axes[2].set_title('torque vs step')
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel('relax step')
        ax.grid(True, alpha=0.3)
    fig.suptitle(f'box {d["nx"]}x{d["ny"]} convergence')
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_scaling(dicts, out_path):
    """Plateau torque + size vs lattice site count N across boxes."""
    dd = sorted(dicts, key=lambda d: int(d['nx']) * int(d['ny']))
    N = np.array([int(d['nx']) * int(d['ny']) for d in dd], dtype=float)
    tmax = np.array([float(np.asarray(d['tau_max'])[-1]) for d in dd])
    trms = np.array([float(np.asarray(d['tau_rms'])[-1]) for d in dd])
    tmed = np.array([float(np.asarray(d['tau_median'])[-1]) for d in dd])
    d1 = np.array([float(np.asarray(d['D1'])[-1]) * 1e9 for d in dd])
    d2 = np.array([float(np.asarray(d['D2'])[-1]) * 1e9 for d in dd])
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    axes[0].loglog(N, tmax, 'o-', label='max')
    axes[0].loglog(N, tmed, 's-', label='median')
    axes[0].loglog(N, trms, 'd-', label='rms')
    axes[0].set_xlabel('N sites')
    axes[0].set_ylabel('plateau tangential torque (T)')
    axes[0].set_title('torque-floor scaling')
    axes[0].legend()
    axes[1].semilogx(N, d1, 'o-', label=r'$D_1$')
    axes[1].semilogx(N, d2, 's-', label=r'$D_2$')
    axes[1].axhline(186.5, color='k', ls='--', label='256$^2$ ref')
    axes[1].set_xlabel('N sites')
    axes[1].set_ylabel('relaxed diameter (nm)')
    axes[1].set_title('size vs box')
    axes[1].legend()
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def main():
    """Run the Python diagnostics (optional) and render the plots."""
    # =========================== User Configuration =========================
    # Boundary condition: 'newell' (periodic) or 'newell_freebc' (isolated
    # zero-padded). Each writes/reads its own subdirectory so the two never
    # overwrite each other; matches the C++ validate_relax_torque <bc> flag.
    demag_kind = 'newell'
    # Boxes to relax in PYTHON (slow with newell; keep modest). Set to []
    # to skip running and only plot existing NPZ (e.g. from the C++ tool).
    boxes_to_run = [(160, 140), (200, 200)]
    chunk = 2000
    n_chunks = 20
    alpha_relax = 1.0
    dt = 5.0e-14
    demag_accuracy = 4.0
    demag_tol_conv = 0.02
    # Box to use for the map + convergence panels (prefix match on
    # filename, e.g. '350x500'); falls back to the largest available.
    map_box = '350x500'
    out_dir = os.path.join(
        'output/stochastic_llgs/validation/relax_torque', demag_kind)
    fig_dir = os.path.join('output/figures_sllg/relax_torque', demag_kind)
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    cfg = {
        'dt': dt, 'chunk': chunk, 'n_chunks': n_chunks,
        'alpha_relax': alpha_relax, 'demag_kind': demag_kind,
        'demag_accuracy': demag_accuracy, 'demag_tol_conv': demag_tol_conv,
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run the Python reference diagnostics for the configured boxes.
    for nx, ny in boxes_to_run:
        d = _diagnose_box(cfg, nx, ny)
        path = os.path.join(out_dir, f'box_{nx}x{ny}.npz')
        np.savez_compressed(path, **d)
        print(f'saved {path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot from every NPZ in the directory (C++ and Python).
    files = sorted(glob.glob(os.path.join(out_dir, 'box_*.npz')))
    if not files:
        raise RuntimeError(
            f'validate_relax_torque: no box_*.npz in {out_dir!r}; run '
            f'this script or the C++ validate_relax_torque first.')
    dicts = [dict(np.load(f, allow_pickle=True)) for f in files]
    # Map/convergence box: prefer map_box, else the largest.
    pick = None
    for f, d in zip(files, dicts):
        if map_box in os.path.basename(f):
            pick = d
    if pick is None:
        pick = max(dicts, key=lambda d: int(d['nx']) * int(d['ny']))
    _plot_map_hist(pick, os.path.join(fig_dir, 'torque_map_hist.png'))
    _plot_convergence(pick, os.path.join(fig_dir, 'convergence.png'))
    _plot_scaling(dicts, os.path.join(fig_dir, 'box_scaling.png'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 256x256 positive control (if present).
    for d in dicts:
        if int(d['nx']) == 256 and int(d['ny']) == 256:
            print(
                f'control 256x256: tau_max={float(d["tau_max"][-1]):.2e} '
                f'(S47 ref 3.4e-5), D1={float(d["D1"][-1])*1e9:.1f} '
                f'D2={float(d["D2"][-1])*1e9:.1f} nm (ref 186.5)')


# =============================================================================
if __name__ == '__main__':
    main()
