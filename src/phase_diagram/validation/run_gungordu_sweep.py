"""Phase-diagram sweep over (K_top, H_z) at fixed (D, A_ex, M_s)
for the Güngördü 2016 / Banerjee 2014 benchmark.

Runs the generic phase-diagram pipeline on a `nx_grid x ny_grid`
(K_top, H_z) grid, with each cell relaxed on a `nx_lat x ny_lat`
spin lattice. The total task count is 2500 (50 x 50 grid) by
default. With seven initial conditions per cell and a 64 x 64
lattice, this is intended for cluster execution as a SLURM array
(set `SLURM_ARRAY_TASK_ID` to dispatch one cell per task) but
will also run locally in serial or process-pool mode.

Coordinates (match Güngördü 2016 Fig. 3 exactly)
------------------------------------------------
Güngördü's phase diagram is in the dimensionless plane
    a_s = A_s J / D^2  (anisotropy, x-axis),
    h_g = H   J / D^2  (field,      y-axis),
with J <-> A_ex, A_s <-> the uniaxial anisotropy, D <-> D. The
sweep is built directly on a rectangular (a_s, h_g) grid:
    K_top  = - a_s D^2 / A_ex     (sign flip; see below),
    H_z    = h_g D^2 / (M_s A_ex).
Güngördü's model has NO demag (anisotropy only), so the sweep
runs with `demag_kind='none'`. With demag off, BOTH the field
(`effective_field_demag_pair`) and the energy (`total_energy`)
use the BARE K via `bare_anis_prefactors` plus an explicit
(here zero) demag term -- so the simulator's effective
anisotropy IS the bare K, with no fold. The simulator K > 0 is
easy-AXIS (energy -K m_z^2) whereas Güngördü A_s > 0 is
easy-PLANE (energy +A_s n_z^2), hence K = -A_s_phys. a_s and
h_g are recovered for plotting via a_s = -K A_ex / D^2.

The reference D = 1.5 mJ/m^2 with A_ex = 16 pJ/m, M_s =
8.755e5 A/m (mu_0 M_s = 1.1 T) places the swept window over
a_s in [-1.5, 1.5] and h_g in [0, 2.0], matching Fig. 3.

Output
------
output/phase_diagram/validation/gungordu_K_H.npz

Functions
---------
main
    Build the (a_s, h_g) -> (K_top, H_z) grid and invoke the
    production `sweep` / `sweep_array_partial` orchestrators
    with `demag_kind='none'`.
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
from src.phase_diagram.sweep import sweep, sweep_array_partial

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
    # Build the (a_s, h_g) -> (K_top, H_z) grid and dispatch it
    # through the production sweep (serial or SLURM array).
    # =============================== User Configuration ======================
    # Fixed material: D and A_ex anchor the dimensionless map;
    # M_s is fixed for the h_RT axis conversion. RKKY off (the
    # benchmark is for an isolated FM layer-pair; without RKKY
    # the top and bottom layers in the SAF integrator evolve
    # independently and the top layer reproduces the single-FM
    # phase diagram). Demag is left in its default state in the
    # underlying relaxer; for a strict no-demag run, edit the
    # sweep harness to bypass `precompute_demag_kernels` (not
    # changed here so the sweep API stays untouched).
    A_ex = 16.0e-12          # J/m
    D = 1.5e-3               # J/m^2
    Ms = 8.755e5             # A/m
    mu0 = 4.0 * math.pi * 1.0e-7
    # Gungordu Fig. 3 dimensionless axes (J <-> A_ex, A_s <-> the
    # uniaxial anisotropy, D <-> D):
    #     a_s = A_s J / D^2   (= -K A_ex / D^2 here, see below)
    #     h_g = H   J / D^2 = (M_s B_z) A_ex / D^2
    # Demag is OFF; effective_field_demag_pair and total_energy
    # both use the BARE K (bare_anis_prefactors) + explicit
    # (zero) demag, so the effective anisotropy IS the bare K.
    # Sim K>0 = easy-axis (-K m_z^2); Gungordu A_s>0 = easy-plane
    # (+A_s n_z^2) -> K = -A_s_phys. Field is K-independent:
    # B_z = h_g D^2 / (M_s A_ex).
    a_s_lo, a_s_hi = -1.5, 1.5      # Fig. 3 x-axis
    h_g_lo, h_g_hi = 0.0, 2.0       # Fig. 3 y-axis
    n_x = 50                 # a_s grid points
    n_y = 50                 # h_g grid points  -> 2500 cells
    a_s_grid = np.linspace(a_s_lo, a_s_hi, n_x)
    h_g_grid = np.linspace(h_g_lo, h_g_hi, n_y)
    D2_over_A = D * D / A_ex
    A_s_phys = a_s_grid * D2_over_A
    # Demag is OFF, and BOTH the field (effective_field_demag_pair
    # via bare_anis_prefactors) and the energy (total_energy via
    # bare_anis_prefactors) use the BARE K with no demag fold.
    # So the simulator's effective anisotropy IS the bare K. The
    # simulator K > 0 is easy-AXIS (energy -K m_z^2), while
    # Güngördü A_s > 0 is easy-PLANE (energy +A_s n_z^2), hence
    # the sign flip: K = -A_s_phys. (Earlier code added a
    # 0.5 mu0 Ms^2 fold to cancel a -mu0 Ms field term that this
    # demag-explicit path never applies -- that fold made K > 0
    # easy-axis and collapsed every cell to FM.)
    K_values = -A_s_phys
    # h_g is independent of K, so the H grid is a clean 1-D
    # array (unlike the old h_RT which scaled with K).
    # B_z = h_g D^2 / (M_s A_ex)  [Tesla]. (NOTE: not
    # D2_over_A / (Ms A_ex), which would carry a spurious
    # extra 1/A_ex.)
    H_values = h_g_grid * (D * D) / (Ms * A_ex)
    # Axes used by the sweep harness.
    axis_x_name = 'K_top'
    axis_x_values = K_values
    axis_y_name = 'H_z'
    axis_y_values = H_values
    # Natural DMI helix wavelength lambda = 4 pi A / D. D and A
    # are fixed across the sweep, so lambda (and the IC sizes
    # derived from it) are constant. The single-skyrmion IC
    # otherwise defaults to a 93 nm radius unrelated to the
    # cell; pin it to lambda/4 (matching the lattice ICs) so it
    # is a sane seed at this geometry.
    helix_lambda = 4.0 * math.pi * A_ex / D
    sk_R = helix_lambda / 4.0
    sk_dw = sk_R / 2.5
    # Fixed overrides applied to every task.
    fixed_overrides = {
        'D': float(D),
        'A_ex': float(A_ex),
        'Ms': float(Ms),
        # Decouple the two SAF layers so the top layer
        # reproduces single-FM physics.
        'H_RKKY': 0.0,
        # Cell-derived single-skyrmion IC size (not the 93 nm
        # default, which does not fit the box).
        'skyrmion_R': float(sk_R),
        'skyrmion_dw': float(sk_dw),
    }
    # Demag OFF: Gungordu is a local model (anisotropy only, no
    # magnetostatics). Runs through the production sweep / relax
    # / effective_field_demag_pair path with a zero demag kernel.
    demag_kind = 'none'
    # IC ensemble = the 4 Gungordu candidate phases (FM, SP,
    # SkX, SC), matching the paper's 4-ansatz energy comparison
    # (FM, SkX, SC, SP). The generic 7-IC default over-seeds FM
    # and includes an isolated skyrmion that is not a Gungordu
    # phase, while lacking a square (SC) seed; this 4-IC set is
    # both targeted and ~1.75x cheaper.
    #   fm_par     -> FM (uniform)
    #   stripe     -> SP (spin spiral)
    #   sk_lattice -> SkX (hex skyrmion lattice)
    #   sq_lattice -> SC (square skyrmion lattice)
    ic_list = (
        ('fm_par', None),
        ('stripe', None),
        ('sk_lattice', None),
        ('sq_lattice', None),
    )
    # Lattice: the box must hold >= ~2 helix periods AND resolve
    # the texture (cell < A/D = 10.7 nm). lambda = 134 nm here,
    # so 64 x 64 at a = 5 nm gives a 320 nm box = 2.4 lambda
    # with ~2 cells per A/D -- good Gungordu-quality resolution
    # and extent. (A 30 x 30 box could not satisfy both.)
    nx_lat = 64
    ny_lat = 64
    a = 5.0e-9
    dt = 2.0e-14
    # Relaxation tolerances (Option A: near-equilibrium energy
    # comparison, matching Gungordu's compare-optimized-ansatz
    # method). The textured states (SP, SkX, SC) settle their
    # ENERGY quickly but their residual torque relaxes slowly
    # (incommensurate textures in a finite box never reach
    # tol < 1e-4 even at 150k steps). The ground-state gate
    # only counts converged records, so a too-tight tol_torque
    # excludes the textures and every cell defaults to FM. We
    # therefore use a near-equilibrium tol_torque = 5e-3 (the
    # energy is well-settled there) with more headroom in
    # max_steps so the textures qualify as converged and enter
    # the min-energy comparison.
    max_steps = 100_000
    tol_torque = 5.0e-3
    tol_dE = 1.0e-8
    alpha_relax = 1.0
    # Output paths.
    out_dir = 'output/phase_diagram/validation'
    out_path = os.path.join(out_dir, 'gungordu_K_H.npz')
    partial_dir = os.path.join(out_dir, 'gungordu_partials')
    os.makedirs(out_dir, exist_ok=True)
    # ============================ End User Configuration ====================
    print(
        f'run_gungordu_sweep (Gungordu Fig.3 coords, demag '
        f'{demag_kind!r}): D={D*1e3:.3f} mJ/m^2, '
        f'A={A_ex*1e12:.1f} pJ/m, mu_0 M_s={mu0*Ms:.3f} T')
    print(
        f'  grid {n_x}x{n_y} = {n_x*n_y} cells; '
        f'a_s=A_s J/D^2 in [{a_s_lo:.2f}, {a_s_hi:.2f}] '
        f'(K in [{K_values.min():.2e}, {K_values.max():.2e}] '
        f'J/m^3), h_g=H J/D^2 in [{h_g_lo:.2f}, {h_g_hi:.2f}] '
        f'(H_z in [{H_values.min():.3f}, {H_values.max():.3f}] T)')
    print(
        f'  lattice {nx_lat}x{ny_lat} at a={a*1e9:.2f} nm '
        f'(box {nx_lat*a*1e9:.0f} nm); '
        f'max_steps={max_steps}, tol_torque={tol_torque:.0e}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # No array index: run the whole grid in one process.
    array_id_env = os.environ.get('SLURM_ARRAY_TASK_ID')
    if array_id_env is None:
        sweep(
            axis_x_name=axis_x_name,
            axis_x_values=axis_x_values,
            axis_y_name=axis_y_name,
            axis_y_values=axis_y_values,
            fixed_overrides=fixed_overrides,
            nx=nx_lat, ny=ny_lat,
            max_steps=max_steps,
            tol_torque=tol_torque, tol_dE=tol_dE,
            alpha_relax=alpha_relax,
            a=a, dt=dt,
            workers=None, out_path=out_path,
            demag_kind=demag_kind, ic_list=ic_list,
        )
        return
    # SLURM-array mode.
    array_task_id = int(array_id_env)
    sims_per_task = int(os.environ.get('SIMS_PER_TASK', '1'))
    sweep_array_partial(
        axis_x_name=axis_x_name,
        axis_x_values=axis_x_values,
        axis_y_name=axis_y_name,
        axis_y_values=axis_y_values,
        fixed_overrides=fixed_overrides,
        nx=nx_lat, ny=ny_lat,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax,
        a=a, dt=dt,
        array_task_id=array_task_id,
        sims_per_task=sims_per_task,
        partial_dir=partial_dir,
        demag_kind=demag_kind, ic_list=ic_list,
    )


# =============================================================================
if __name__ == '__main__':
    main()
