"""(D, K_top) sweep at fixed H_z = 0 -- chiral window vs anisotropy.

Maps the K-dependence of the chiral phase window at zero
applied field, from just above the perpendicular-magnetic-
anisotropy threshold K = (1/2) mu0 Ms^2 = 1.285 MJ/m^3 to
just above the K = 1.60 MJ/m^3 baseline already on disk.

The sweep machinery is shared with `skyrmion_simulator.phase_diagram.sweep`:
this module is just a thin entry point that fixes the axis
choices, axis grids and output paths, then dispatches to
`sweep` (local mode) or `sweep_array_partial` (SLURM-array
mode).

Run as:

    python -m studies.saf_racetrack.phase_diagram.sweep_DK_H0

SLURM-array mode activates automatically when the
environment variable `SLURM_ARRAY_TASK_ID` is set.

Functions
---------
main
    Entry point -- edit the User Configuration block to
    change axis grids, lattice, tolerances, or output paths.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import os
import sys
# Third-party
import numpy as np
# Local
from skyrmion_simulator.phase_diagram.sweep import sweep, sweep_array_partial

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def main():
    """Entry point for the (D, K_top) sweep at H_z = 0."""
    # ================ User Configuration ================
    axis_x_name = 'D'
    # D_c spans ~0.63 mJ/m^2 (K = 1.30) to ~3.08 mJ/m^2
    # (K = 1.65). A uniform 30-point grid covers the union
    # of chiral windows across K; clustering around a single
    # D_c would skew the (D, K) map at the K extremes.
    axis_x_values = np.linspace(0.0, 4.0e-3, 30)
    axis_y_name = 'K_top'
    # K bounded below by (1/2) mu0 Ms^2 = 1.285 MJ/m^3
    # (K_eff = 0 -- PMA threshold). 1.30 MJ/m^3 keeps
    # K_eff > 0 with the widest chiral window; 1.65 MJ/m^3
    # sits a hair above the K = 1.60 MJ/m^3 sweep already
    # on disk.
    axis_y_values = np.linspace(1.30e6, 1.65e6, 30)
    # H_z = 0 fixes the antiparallel SAF manifold. The
    # 'H_z' axis builder writes H_ext = (0, 0, H_z); we pin
    # that vector explicitly so make_params reads zero.
    fixed_overrides = {
        'H_ext': np.array([0.0, 0.0, 0.0]),
    }
    # Lattice size in cells. At a = 1.0 nm the physical
    # box is L = nx * a = 256 nm.
    nx = 256
    ny = 256
    # Lattice constant in metres.
    a = 1.0e-9
    # Time step in seconds.
    dt = 2.0e-14
    # Relaxation safety cutoff and convergence tolerances.
    max_steps = 100000
    tol_torque = 1e-3
    tol_dE = 1e-7
    # Gilbert damping override during relaxation.
    alpha_relax = 1.0
    # Process-pool worker count for LOCAL mode.
    workers = None
    # Number of tasks to run sequentially per SLURM array
    # element.
    sims_per_task = 1
    # Output paths -- distinct namespace from the (D, H_z)
    # sweep results.
    out_path = 'output/phase_diagram/D_K_H0.npz'
    partial_dir = 'output/phase_diagram/D_K_H0_partials'
    # ============ End User Configuration =================
    array_id_env = os.environ.get('SLURM_ARRAY_TASK_ID')
    if array_id_env is None:
        sweep(
            axis_x_name=axis_x_name,
            axis_x_values=axis_x_values,
            axis_y_name=axis_y_name,
            axis_y_values=axis_y_values,
            fixed_overrides=fixed_overrides,
            nx=nx, ny=ny,
            max_steps=max_steps,
            tol_torque=tol_torque, tol_dE=tol_dE,
            alpha_relax=alpha_relax,
            a=a, dt=dt,
            workers=workers, out_path=out_path,
        )
        return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SLURM-array mode: process this element's slice of
    # tasks and write a partial NPZ.
    array_task_id = int(array_id_env)
    sweep_array_partial(
        axis_x_name=axis_x_name,
        axis_x_values=axis_x_values,
        axis_y_name=axis_y_name,
        axis_y_values=axis_y_values,
        fixed_overrides=fixed_overrides,
        nx=nx, ny=ny,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax,
        a=a, dt=dt,
        array_task_id=array_task_id,
        sims_per_task=sims_per_task,
        partial_dir=partial_dir,
    )


# =====================================================================
if __name__ == '__main__':
    sys.exit(main())
