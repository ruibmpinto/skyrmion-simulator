"""Render the IC ensemble at a chosen parameter point.

For each IC in `sweep._ic_specs()`, build the initial spin
configuration and classify it. The output is a single
figure with one panel per IC showing the top-layer
m_z(x, y); the panel title shows the IC name and the
classifier's verdict on the *unrelaxed* state, color-coded
by the predicted phase.

Reading the figure
------------------
- Each panel's image is the initial condition.
- Each panel's title is the phase the IC would carry if
  the relaxation never changed it. After a real sweep,
  compare with `gs_label_idx[i, j]` from the NPZ to see
  which ICs the relaxation moved out of their starting
  basins.

Usage
-----
Edit the variables in `main()`'s User Configuration block,
then run:

    python -m src.phase_diagram.show_ics

Output: `output/phase_diagram/ic_gallery.png`.

Functions
---------
render_gallery
    Build, classify, and plot every IC at one (D, H_z, K)
    point. Pure-Python entry point.
main
    Read the User Configuration block and call
    `render_gallery`.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import os
import sys
# Third-party
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.phase_diagram.classifier import classify
from src.phase_diagram.params_helper import make_params
from src.phase_diagram.plot_phase_diagram import _PHASE_COLORS
from src.phase_diagram.sweep import _build_ic, _ic_specs

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def render_gallery(D, H_z, K_top, K_bot, nx, ny, a, out_path):
    """Render every IC at one (D, H_z, K) parameter point.

    Parameters
    ----------
    D : float
        DMI strength in J/m^2.
    H_z : float
        External field along z in T.
    K_top, K_bot : float
        Per-layer raw anisotropy in J/m^3.
    nx, ny : int
        Lattice size in cells.
    a : float
        Lattice constant in m.
    out_path : str
        Output PNG path; parent directory is created if
        absent.

    Returns
    -------
    out_path : str
        The written PNG path.
    """
    p = make_params(
        nx=nx, ny=ny, a=a,
        D=float(D),
        H_ext=np.array([0.0, 0.0, float(H_z)]),
        K_top=float(K_top), K_bot=float(K_bot),
    )
    ic_list = _ic_specs()
    n = len(ic_list)
    # Pre-classify every IC so the table prints first.
    records = []
    for name, seed in ic_list:
        m_top, m_bot = _build_ic(name, seed, p)
        label, obs = classify(m_top, m_bot, p)
        if label not in _PHASE_COLORS:
            raise RuntimeError(
                f'Classifier returned unknown label '
                f'{label!r} for IC {name!r}; '
                f'cannot color the gallery panel.'
            )
        records.append({
            'name': name, 'seed': seed,
            'label': label, 'obs': obs,
            'm_top': m_top,
        })
    _print_table(records, D, H_z, K_top)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Gallery
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(3.8 * cols, 4.6 * rows),
        gridspec_kw={'hspace': 0.55, 'wspace': 0.2},
    )
    axes = np.atleast_1d(axes).ravel()
    for ax, rec in zip(axes, records):
        color = _PHASE_COLORS[rec['label']]
        seed_str = (
            '' if rec['seed'] is None
            else f' (seed={rec["seed"]})'
        )
        ax.imshow(
            rec['m_top'][..., 2], cmap='RdBu_r',
            vmin=-1.0, vmax=1.0, origin='lower',
        )
        ax.set_title(
            f'{rec["name"]}{seed_str}\n'
            f'→ {rec["label"]}\n'
            f'Q={rec["obs"]["Q"]:+.2f}, '
            f'<m_z>={rec["obs"]["mz_top"]:+.2f}, '
            f'm_dot={rec["obs"]["m_dot"]:+.2f}',
            color=color, fontsize=9,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.0)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(
        f'IC ensemble (unrelaxed), '
        f'D = {D * 1e3:.3f} mJ/m$^2$, '
        f'$H_z$ = {H_z:+.3f} T, '
        f'K = {K_top * 1e-6:.2f} MJ/m$^3$,  '
        f'{nx}×{ny} lattice, a = {a * 1e9:.2f} nm',
        fontsize=13, y=0.995,
    )
    fig.subplots_adjust(
        top=0.90, bottom=0.04, left=0.04, right=0.98,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
def _print_table(records, D, H_z, K_top):
    """Print a readable name → predicted-phase mapping."""
    print(
        f'\nIC ensemble (unrelaxed) at '
        f'D = {D * 1e3:.3f} mJ/m^2, '
        f'H_z = {H_z:+.3f} T, '
        f'K = {K_top * 1e-6:.2f} MJ/m^3'
    )
    print('-' * 72)
    print(
        f'{"IC name":<16} {"seed":>5} {"→ phase":<10}  '
        f'{"Q":>7}  {"<m_z>":>7}  {"m_dot":>7}'
    )
    print('-' * 72)
    for rec in records:
        seed = '' if rec['seed'] is None else str(rec['seed'])
        print(
            f'{rec["name"]:<16} {seed:>5} '
            f'→ {rec["label"]:<8}  '
            f'{rec["obs"]["Q"]:+7.3f}  '
            f'{rec["obs"]["mz_top"]:+7.3f}  '
            f'{rec["obs"]["m_dot"]:+7.3f}'
        )
    print('-' * 72)


# ---------------------------------------------------------------------
def main():
    """Read the User Configuration block and render."""
    # ================ User Configuration ================
    # Material / lattice constants — should match the
    # sweep run you want to compare against.
    D       = 2.86e-3         # J/m^2 (D_c at K = 1.60 MJ/m^3)
    H_z     = 0.0             # T
    K_top   = 1.60e6          # J/m^3
    K_bot   = 1.60e6          # J/m^3
    nx      = 256
    ny      = 256
    a       = 1.0e-9          # m
    out_path = 'output/phase_diagram/ic_gallery.png'
    # ============ End User Configuration =================
    written = render_gallery(
        D=D, H_z=H_z, K_top=K_top, K_bot=K_bot,
        nx=nx, ny=ny, a=a, out_path=out_path,
    )
    print(f'Wrote {written}')


# =====================================================================
if __name__ == '__main__':
    sys.exit(main())
