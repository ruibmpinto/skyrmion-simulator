"""Time profiles of the six compared drive pulses.

Plots J(t) for each shape at matched peak amplitude and duration
alongside its RUNNING charge and action integrals, so the rate at which
each shape spends its budget is visible, not only the endpoint. Those
two integrals are the denominators of the efficiency measures. The
three triangular asymmetries share both integrals exactly, so comparing
them isolates the rise/fall asymmetry alone.

No argparse; configure the run via the variables at the top of
`main()`.

Run with:
    python -m scripts.plot_pulse_profiles
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.orchestrator.pulsed_run import make_pulse, shape_names

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
    """Render J(t) with its running charge and action per shape."""
    # =========================== User Configuration =========================
    out_dir = 'output/figures_driving_T'
    peak_j = 1.0e11            # A/m^2, common peak for the comparison
    t_pulse = 500.0e-12        # s, common duration
    t_settle = 500.0e-12       # s, unforced interval after the pulse
    gauss_fwhm = 250.0e-12     # s, Gaussian width
    n_grid = 4001              # samples for the running integrals
    # ======================= End User Configuration =========================
    plt.rcParams.update({
        'font.size': 15,
        'axes.titlesize': 19,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })
    os.makedirs(out_dir, exist_ok=True)
    t_end = t_pulse + t_settle
    grid = np.linspace(0.0, t_end, n_grid)
    names = shape_names()
    # Single colour for every shape: each row is its own panel, so
    # colour carries no information and a shared one keeps the reader
    # comparing curves rather than hues.
    colour = '#2c7bb6'
    fig, axes = plt.subplots(len(names), 3, sharex=True,
                             figsize=(14.5, 2.7*len(names)))
    print('shape            charge (A s/m^2)   action (A^2 s/m^4)  '
          'rel. charge')
    totals = {}
    for row, shape in enumerate(names):
        pulse = make_pulse(shape, peak_j, t_pulse, gauss_fwhm)
        j_t = np.array([pulse(t) for t in grid])
        # Running integrals by cumulative trapezoid, so each panel shows
        # how the cost accumulates rather than only its endpoint.
        dt = np.diff(grid)
        mid_j = 0.5*(j_t[1:] + j_t[:-1])
        mid_j2 = 0.5*(j_t[1:]**2 + j_t[:-1]**2)
        charge_t = np.concatenate(([0.0], np.cumsum(mid_j*dt)))
        action_t = np.concatenate(([0.0], np.cumsum(mid_j2*dt)))
        totals[shape] = (charge_t[-1], action_t[-1])
        for col, (values, scale) in enumerate((
                (j_t, 1e-11), (charge_t, 1.0), (action_t, 1e-12))):
            ax = axes[row][col]
            ax.plot(grid*1e12, values*scale, color=colour, lw=3.0)
            ax.axvline(t_pulse*1e12, color='0.6', ls=':', lw=1.5)
            if col == 0:
                ax.set_ylabel('%s\n$J$ ($10^{11}$)' % shape,
                              fontsize=14)
            if row == len(names) - 1:
                ax.set_xlabel(r'$t$ (ps)')
    axes[0][0].set_title(r'$J(t)$')
    axes[0][1].set_title(r"charge $\int_0^t J\,dt'$ (A s/m$^2$)")
    axes[0][2].set_title(
        r"action $\int_0^t J^2\,dt'$ ($10^{12}$ A$^2$ s/m$^4$)")
    # Common y limits per column so the shapes are directly comparable.
    for col in range(3):
        hi = max(ax.get_ylim()[1] for ax in axes[:, col])
        for ax in axes[:, col]:
            ax.set_ylim(0.0, hi)
    ref = totals['square'][0]
    for shape in names:
        charge, action = totals[shape]
        print('%-15s %14.4g %20.4g %12.3f'
              % (shape, charge, action, charge/ref))
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'pulse_profiles.png')
    fig.savefig(out_path, dpi=160)
    print('Saved: %s' % out_path)
    plt.close(fig)


# =============================================================================
if __name__ == '__main__':
    main()
