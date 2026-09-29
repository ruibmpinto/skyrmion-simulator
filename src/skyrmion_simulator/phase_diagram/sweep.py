"""Generic 2D parameter sweep for the phase-diagram pipeline.

User picks two sweep axes in `main()` from the registry in
`skyrmion_simulator.phase_diagram.axis_specs` (e.g. `('D', 'H_z')`,
`('D', 'K_top')`, `('D', 'H_RKKY')`). For each (axis_x,
axis_y) point on the configured grid, seven initial
conditions are relaxed independently and the lowest-energy
converged state is taken as the ground state.

Two execution modes share a single entry point:

* Local mode (default): one process runs the full sweep
  with a `ProcessPoolExecutor` and writes
  `output/phase_diagram/<run_tag>.npz`.
* SLURM array mode: each array element runs the slice of
  tasks it owns and writes a partial NPZ. Combined later
  by an aggregator (kept legacy until updated).
  Triggered automatically when `SLURM_ARRAY_TASK_ID` is set.

Configuration lives at the top of `main()`. No legacy
fallbacks: unknown axis names, empty axis arrays, and
schema mismatches all raise `RuntimeError`.

Functions
---------
build_tasks
    Build the list of tasks for the configured sweep.
run_one
    Worker: relax + classify a single task and return
    observables and final spin pair.
sweep
    Local-mode orchestrator: dispatch tasks, aggregate
    ground states, write NPZ.
sweep_array_partial
    SLURM-array mode: run the assigned slice of tasks and
    write a partial NPZ.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
# Third-party
import numpy as np
# Local
from skyrmion_simulator.phase_diagram.axis_specs import overrides_for
from skyrmion_simulator.phase_diagram.classifier import classify, PHASE_LABELS
from skyrmion_simulator.phase_diagram.initial_conditions_ext import (
    hex_lattice_bubbles,
    hex_lattice_skyrmions,
    random_state,
    square_lattice_skyrmions,
    stripe_state,
)
from skyrmion_simulator.simulator.params_helper import make_params
from skyrmion_simulator.simulator.relaxation import relax
from skyrmion_simulator.simulator.demag import precompute_demag_kernels
from skyrmion_simulator.simulator.initial_conditions import (
    saf_skyrmion,
    uniform_state,
)

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================
_PHASE_LABELS = PHASE_LABELS


# Suggested grid sizes per dimension (cells per axis). These
# are just hints for `np.linspace(..., n)` in `main()`; the
# user is free to use any positive integer.
grid_size = {
    'test': 4,
    'quick': 8,
    'coarse': 20,
    'medium': 40,
    'fine': 50,
}


def _ic_specs():
    """Return the canonical 7-member prepared IC ensemble.

    Designed to seed every phase region the classifier can
    label:

    * 5 random spheres for statistical exploration (commented out).
    * `fm_anti`, `fm_par` for the two FM branches across
      the spin-flop.
    * `skyrmion` for a single isolated Neel skyrmion (iSk).
    * `stripe` (x) and `stripe_y` for spin-spiral states
      with two orthogonal wavevectors.
    * `sk_lattice` for the SkX basin (Q != 0 per cell).
    * `bubble_lattice` for the BX basin (Q = 0 per cell,
      same FFT signature as SkX).
    """
    return (
        # ('random', 11),  # commented out (see docstring)
        # ('random', 22),  # commented out (see docstring)
        # ('random', 33),  # commented out (see docstring)
        # ('random', 44),  # commented out (see docstring)
        # ('random', 55),  # commented out (see docstring)
        ('fm_anti', None),
        ('fm_par', None),
        ('skyrmion', None),
        ('stripe', None),
        ('stripe_y', None),
        ('sk_lattice', None),
        ('bubble_lattice', None),
    )


# ---------------------------------------------------------------------
def build_tasks(axis_x_name, axis_x_values,
                axis_y_name, axis_y_values,
                fixed_overrides,
                nx, ny, max_steps, tol_torque, tol_dE,
                alpha_relax, a, dt, demag_kind='slab',
                ic_list=None):
    """Build the flat task list for a generic 2D sweep.

    Each (axis_x, axis_y) point is multiplied by the IC
    ensemble. The per-task `overrides` dict is the union of
    `fixed_overrides`, the x-axis override, and the y-axis
    override. Keyed overrides from different axes must not
    collide; this function raises if they do.

    Parameters
    ----------
    axis_x_name, axis_y_name : str
        Keys into `axis_specs.axes`.
    axis_x_values, axis_y_values : numpy.ndarray(1d)
        Non-empty arrays of axis values in SI units.
    fixed_overrides : dict
        Constants applied to every task before the axis
        overrides. Must not collide with axis overrides.
    nx, ny : int
        Lattice size in cells.
    max_steps : int
        Relaxation cutoff.
    tol_torque, tol_dE : float
        Convergence tolerances.
    alpha_relax : {float, None}
        Override Gilbert damping during relaxation.
    a : float
        Lattice constant (m).
    dt : float
        Time step (s).

    Returns
    -------
    tasks : list of dict
        Each carries `i, j, k, axis_x_value, axis_y_value,
        ic_name, ic_seed, overrides, nx, ny, max_steps,
        tol_torque, tol_dE, alpha_relax, a, dt`.
    axis_x_values, axis_y_values : numpy.ndarray(1d)
        The validated SI arrays.
    """
    if axis_x_name == axis_y_name:
        raise RuntimeError(
            f'axis_x_name and axis_y_name must differ; '
            f'both are {axis_x_name!r}.'
        )
    axis_x_values = np.asarray(axis_x_values, dtype=float)
    axis_y_values = np.asarray(axis_y_values, dtype=float)
    if axis_x_values.size == 0 or axis_y_values.size == 0:
        raise RuntimeError(
            'axis_x_values and axis_y_values must be non-empty.'
        )
    if axis_x_values.ndim != 1 or axis_y_values.ndim != 1:
        raise RuntimeError(
            'axis_*_values must be 1-D arrays.'
        )
    # IC ensemble: default 7-member generic set, or a caller-
    # supplied (ic_name, ic_seed) list (e.g. the 4-IC Gungordu
    # set). All cells use the same list so the per-cell IC
    # count is consistent for aggregation.
    if ic_list is None:
        ic_list = _ic_specs()
    tasks = []
    for i, x_val in enumerate(axis_x_values):
        x_overrides = overrides_for(axis_x_name, x_val)
        for j, y_val in enumerate(axis_y_values):
            y_overrides = overrides_for(axis_y_name, y_val)
            collision = (set(x_overrides) & set(y_overrides))
            if collision:
                raise RuntimeError(
                    f'Axes {axis_x_name!r} and {axis_y_name!r}'
                    f' both override {sorted(collision)}.')
            for k, (ic_name, ic_seed) in enumerate(ic_list):
                overrides = dict(fixed_overrides)
                fix_collision = (
                    (set(overrides) & set(x_overrides))
                    | (set(overrides) & set(y_overrides))
                )
                if fix_collision:
                    raise RuntimeError(
                        f'fixed_overrides collides with the '
                        f'swept axes on {sorted(fix_collision)}.')
                overrides.update(x_overrides)
                overrides.update(y_overrides)
                tasks.append({
                    'i': i, 'j': j, 'k': k,
                    'axis_x_value': float(x_val),
                    'axis_y_value': float(y_val),
                    'ic_name': ic_name,
                    'ic_seed': ic_seed,
                    'overrides': overrides,
                    'nx': int(nx), 'ny': int(ny),
                    'max_steps': int(max_steps),
                    'tol_torque': float(tol_torque),
                    'tol_dE': float(tol_dE),
                    'alpha_relax': (
                        float(alpha_relax)
                        if alpha_relax is not None else None
                    ),
                    'a': float(a),
                    'dt': float(dt),
                    'demag_kind': str(demag_kind),
                })
    return tasks, axis_x_values, axis_y_values


# ---------------------------------------------------------------------
def _build_ic(ic_name, ic_seed, p):
    """Materialize an initial-condition pair for parameters p."""
    if ic_name == 'random':
        return random_state(p.nx, p.ny, ic_seed)
    if ic_name == 'fm_anti':
        # Antiparallel SAF; top aligned with sign(H_z) so
        # the layer-swap symmetric polarization is favored
        # by the field as much as possible (the antiparallel
        # state has zero net Zeeman, but starting from the
        # correct top-polarity avoids a metastable mirror).
        sign = -1.0 if p.H_ext[2] < 0.0 else 1.0
        m_top = uniform_state(p.nx, p.ny, np.array([0.0, 0.0, sign]),)
        m_bot = uniform_state(p.nx, p.ny, np.array([0.0, 0.0, -sign]),)

        return m_top, m_bot
    if ic_name == 'fm_par':
        # Parallel FM aligned with the field. Lets the
        # relaxation find the spin-flop transition above
        # |H_z| > H_RKKY where parallel-FM beats
        # antiparallel-FM (Zeeman gain > RKKY cost).
        sign = -1.0 if p.H_ext[2] < 0.0 else 1.0
        m_top = uniform_state(p.nx, p.ny, np.array([0.0, 0.0, sign]),)
        m_bot = uniform_state(p.nx, p.ny, np.array([0.0, 0.0, sign]),)
        return m_top, m_bot
    if ic_name == 'skyrmion':
        return saf_skyrmion(
            p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
    if ic_name == 'stripe':
        return stripe_state(p.nx, p.ny, _helix_period(p), p.a, axis='x',)
    if ic_name == 'stripe_y':
        return stripe_state(p.nx, p.ny, _helix_period(p), p.a, axis='y',)
    if ic_name == 'sk_lattice':
        period = _helix_period(p)
        R, dw = _lattice_R_dw(period)
        return hex_lattice_skyrmions(
            p.nx, p.ny, p.a, R=R, period=period, dw=dw,)
    if ic_name == 'bubble_lattice':
        period = _helix_period(p)
        R, dw = _lattice_R_dw(period)
        return hex_lattice_bubbles(p.nx, p.ny, p.a, R=R, period=period, dw=dw,)
    if ic_name == 'sq_lattice':
        # Square skyrmion lattice -> seeds the four-fold
        # square-cell (SC) basin of Gungordu 2016.
        period = _helix_period(p)
        R, dw = _lattice_R_dw(period)
        return square_lattice_skyrmions(
            p.nx, p.ny, p.a, R=R, period=period, dw=dw,)
    raise RuntimeError(f'Unknown IC {ic_name!r}.')


def _helix_period(p):
    """Natural DMI helix wavelength lambda = 4 pi A / D.

    Floors at 4 * a to remain resolvable, and at D = 0
    falls back to 4 pi A / 1e-5 J/m^2 (essentially the
    whole box) so the helical seed is still defined.
    """
    if p.D > 0.0:
        period = 4.0 * np.pi * p.A_ex / p.D
    else:
        period = 4.0 * np.pi * p.A_ex / 1.0e-5
    return max(period, 4.0 * p.a)


def _lattice_R_dw(period):
    """Pick (R, dw) for the hex lattice ICs from `period`.

    Skyrmion radius scales as period / 4 so neighbouring
    cores do not overlap heavily. The wall width is
    R / 2.5, slightly thinner than the standard 27 nm
    Co/Pt value to match the strong-PMA regime where the
    helix wavelength is short.
    """
    R = period / 4.0
    dw = R / 2.5
    return R, dw


# ---------------------------------------------------------------------
def run_one(task):
    """Worker: relax + classify a single task."""
    overrides = dict(task['overrides'])
    overrides['J_current'] = 0.0
    overrides['nx'] = int(task['nx'])
    overrides['ny'] = int(task['ny'])
    overrides['a'] = float(task['a'])
    overrides['dt'] = float(task['dt'])
    p = make_params(**overrides)
    # Demag model for this sweep (default 'slab'; 'none' for
    # local-model benchmarks such as the Gungordu phase
    # diagram). 'none'/'slab' take no accuracy/tol_conv.
    demag_kind = task.get('demag_kind', 'slab')
    kernels = precompute_demag_kernels(
        p, kind=demag_kind, accuracy=None, tol_conv=None)
    m_top, m_bot = _build_ic(task['ic_name'], task['ic_seed'], p,)
    relax_kwargs = {
        'max_steps': int(task['max_steps']),
        'tol_torque': float(task['tol_torque']),
        'tol_dE': float(task['tol_dE']),
    }
    if task['alpha_relax'] is not None:
        relax_kwargs['alpha_relax'] = float(task['alpha_relax'])
    m_top, m_bot, converged, n_steps, E, tau = relax(
        m_top, m_bot, p, kernels, **relax_kwargs,
    )
    label, obs = classify(m_top, m_bot, p)
    return {
        'i': task['i'], 'j': task['j'], 'k': task['k'],
        'axis_x_value': task['axis_x_value'],
        'axis_y_value': task['axis_y_value'],
        'ic_name': task['ic_name'],
        'ic_seed': (
            -1 if task['ic_seed'] is None
            else int(task['ic_seed'])
        ),
        'converged': bool(converged),
        'n_steps': int(n_steps),
        'E': float(E),
        'tau_max': float(tau),
        'label': label,
        'mz_top': obs['mz_top'],
        'mz_bot': obs['mz_bot'],
        'm_dot': obs['m_dot'],
        'Q': obs['Q'],
        'k_star': obs['k_star'],
        'P_2': obs['P_2'],
        'P_6': obs['P_6'],
        'P_iso': obs['P_iso'],
        'peak_over_bg': obs['peak_over_bg'],
        'n_periods': obs['n_periods'],
        'q_per_period': obs['q_per_period'],
        'm_top': m_top.astype(np.float32),
        'm_bot': m_bot.astype(np.float32),
    }


# ---------------------------------------------------------------------
def _label_to_int(label):
    # Map a phase label string to its index in _PHASE_LABELS.
    return _PHASE_LABELS.index(label)


def _allocate_arrays(n_D, n_H, n_IC, ny, nx):
    """Allocate aggregation arrays."""
    return {
        'E': np.full((n_D, n_H, n_IC), np.nan),
        'Q': np.zeros((n_D, n_H, n_IC)),
        'mz_top': np.zeros((n_D, n_H, n_IC)),
        'mz_bot': np.zeros((n_D, n_H, n_IC)),
        'm_dot': np.zeros((n_D, n_H, n_IC)),
        'k_star': np.zeros((n_D, n_H, n_IC)),
        'P_2': np.zeros((n_D, n_H, n_IC)),
        'P_6': np.zeros((n_D, n_H, n_IC)),
        'P_iso': np.zeros((n_D, n_H, n_IC)),
        'peak_over_bg': np.zeros((n_D, n_H, n_IC)),
        'n_periods': np.zeros((n_D, n_H, n_IC)),
        'q_per_period': np.zeros((n_D, n_H, n_IC)),
        'tau_max': np.full((n_D, n_H, n_IC), np.nan),
        'n_steps': np.zeros((n_D, n_H, n_IC), dtype=np.int32),
        'converged': np.zeros((n_D, n_H, n_IC), dtype=bool),
        'label_idx_per_ic': np.full((n_D, n_H, n_IC), -1, dtype=np.int8,),
        'gs_idx': np.full((n_D, n_H), -1, dtype=np.int8),
        'gs_label_idx': np.full((n_D, n_H), -1, dtype=np.int8,),
        'gs_m_top': np.zeros((n_D, n_H, ny, nx, 3), dtype=np.float32,),
        'gs_m_bot': np.zeros((n_D, n_H, ny, nx, 3), dtype=np.float32,),
    }


def sweep(axis_x_name, axis_x_values,
          axis_y_name, axis_y_values,
          fixed_overrides,
          nx, ny, max_steps, tol_torque, tol_dE,
          alpha_relax, a, dt,
          workers=None, out_path=None, verbose=True,
          demag_kind='slab', ic_list=None):
    """Run the configured 2D sweep and write the final NPZ.

    The two swept axes are looked up in
    `skyrmion_simulator.phase_diagram.axis_specs.axes`. Output path
    defaults to
    `output/phase_diagram/{axis_x_name}_{axis_y_name}.npz`.
    `demag_kind` selects the demag model ('slab' default;
    'none' for demag-free local-model benchmarks).
    """
    tasks, axis_x_values, axis_y_values = build_tasks(
        axis_x_name=axis_x_name,
        axis_x_values=axis_x_values,
        axis_y_name=axis_y_name,
        axis_y_values=axis_y_values,
        fixed_overrides=fixed_overrides,
        nx=nx, ny=ny,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax, a=a, dt=dt,
        demag_kind=demag_kind, ic_list=ic_list,
    )
    # Resolve the IC list actually used (for the NPZ metadata).
    ic_list_used = _ic_specs() if ic_list is None else ic_list
    if not tasks:
        raise RuntimeError('Empty task list.')
    n_x = len(axis_x_values)
    n_y = len(axis_y_values)
    n_ic = len(_ic_specs())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Probe namespace built from `fixed_overrides` + lattice
    # + a + dt. Used to record axis-invariant material
    # scalars in the NPZ.
    p_probe = _build_probe(fixed_overrides, nx, ny, a, dt)
    arr = _allocate_arrays(n_x, n_y, n_ic, p_probe.ny, p_probe.nx)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if workers is None:
        workers = os.cpu_count() or 1
    t0 = time.time()
    n_done = 0
    n_total = len(tasks)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Running per-(i, j) best converged energy; texture for
    # that winner is written into the gs_m_* arrays in place,
    # so we never store more than one texture per (i, j) at
    # any time.
    best_E = np.full((n_x, n_y), np.inf)
    if workers <= 1:
        iterator = (run_one(t) for t in tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(run_one, t) for t in tasks]
        iterator = (f.result() for f in as_completed(futures))
    for r in iterator:
        # Scatter each per-task result into the (i, j, k) cell
        # of every aggregation array.
        i, j, k = r['i'], r['j'], r['k']
        arr['E'][i, j, k] = r['E']
        arr['Q'][i, j, k] = r['Q']
        arr['mz_top'][i, j, k] = r['mz_top']
        arr['mz_bot'][i, j, k] = r['mz_bot']
        arr['m_dot'][i, j, k] = r['m_dot']
        arr['k_star'][i, j, k] = r['k_star']
        arr['P_2'][i, j, k] = r['P_2']
        arr['P_6'][i, j, k] = r['P_6']
        arr['P_iso'][i, j, k] = r['P_iso']
        arr['peak_over_bg'][i, j, k] = r['peak_over_bg']
        arr['n_periods'][i, j, k] = r['n_periods']
        arr['q_per_period'][i, j, k] = r['q_per_period']
        arr['tau_max'][i, j, k] = r['tau_max']
        arr['n_steps'][i, j, k] = r['n_steps']
        arr['converged'][i, j, k] = r['converged']
        arr['label_idx_per_ic'][i, j, k] = _label_to_int(r['label'])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update running ground state if this IC converged
        # to a lower energy than any seen so far at (i, j).
        if r['converged'] and r['E'] < best_E[i, j]:
            best_E[i, j] = r['E']
            arr['gs_idx'][i, j] = k
            arr['gs_label_idx'][i, j] = _label_to_int(r['label'])
            arr['gs_m_top'][i, j] = r['m_top']
            arr['gs_m_bot'][i, j] = r['m_bot']
        n_done += 1
        if verbose and (n_done % max(1, n_total // 50) == 0):
            elapsed = time.time() - t0
            print(
                f'  {n_done}/{n_total} '
                f'({n_done / n_total * 100:5.1f}%) '
                f'elapsed {elapsed:6.1f}s',
                flush=True,
            )
    if workers > 1:
        pool.shutdown(wait=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write NPZ
    if out_path is None:
        out_path = os.path.join(
            'output', 'phase_diagram',
            f'{axis_x_name}_{axis_y_name}.npz',)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    half_mu0_Ms2 = 0.5 * p_probe.mu0 * p_probe.Ms ** 2
    np.savez_compressed(
        out_path,
        axis_x_name=np.array(axis_x_name, dtype=object),
        axis_x_values=axis_x_values,
        axis_y_name=np.array(axis_y_name, dtype=object),
        axis_y_values=axis_y_values,
        ic_names=np.array(
            [s[0] for s in ic_list_used],
            dtype=object,
        ),
        labels=np.array(_PHASE_LABELS, dtype=object),
        # Material scalars from the probe namespace. These
        # are axis-invariant if (axis_x_name, axis_y_name)
        # is not in {Ms, A_ex, K_top, t_Co, d_Ru} -- the
        # axis_x/y per-task overrides may further modify
        # them for individual tasks.
        Ms=np.float64(p_probe.Ms),
        A_ex=np.float64(p_probe.A_ex),
        mu0=np.float64(p_probe.mu0),
        half_mu0_Ms2=np.float64(half_mu0_Ms2),
        H_RKKY=np.float64(p_probe.H_RKKY),
        K_top_probe=np.float64(p_probe.K_top),
        K_bot_probe=np.float64(p_probe.K_bot),
        t_Co=np.float64(p_probe.t_Co),
        d_Ru=np.float64(p_probe.d_Ru),
        a=np.float64(p_probe.a),
        **{k: v for k, v in arr.items()},
    )
    if verbose:
        elapsed = time.time() - t0
        print('Wrote {out_path} ({elapsed:.1f}s wall, {n_total} tasks).')
    return out_path


# ---------------------------------------------------------------------
def _build_probe(fixed_overrides, nx, ny, a, dt):
    """Probe namespace used to record material constants
    that are axis-invariant for the sweep."""
    probe_kwargs = dict(fixed_overrides)
    probe_kwargs['nx'] = int(nx)
    probe_kwargs['ny'] = int(ny)
    probe_kwargs['a'] = float(a)
    probe_kwargs['dt'] = float(dt)
    return make_params(**probe_kwargs)


def sweep_array_partial(axis_x_name, axis_x_values,
                        axis_y_name, axis_y_values,
                        fixed_overrides,
                        nx, ny, max_steps,
                        tol_torque, tol_dE, alpha_relax,
                        a, dt,
                        array_task_id, sims_per_task,
                        partial_dir, demag_kind='slab',
                        ic_list=None):
    """Run one SLURM array element's slice of tasks.

    Each array element processes `sims_per_task` consecutive
    tasks starting at `array_task_id * sims_per_task` and
    writes a partial NPZ named `part_<id>.npz` in
    `partial_dir`. The partial NPZ stores one record per
    task. Final aggregation into the canonical
    `(n_x, n_y, n_ic)` arrays is performed by an aggregator
    after every array element has finished.

    Parameters
    ----------
    axis_x_name, axis_y_name : str
        Keys into `axis_specs.axes`.
    axis_x_values, axis_y_values : numpy.ndarray(1d)
        SI-valued axis arrays.
    fixed_overrides : dict
        Constants applied to every task.
    nx, ny, max_steps : int
    tol_torque, tol_dE, a, dt : float
    alpha_relax : {float, None}
    array_task_id : int
        SLURM_ARRAY_TASK_ID for this element.
    sims_per_task : int
        Number of tasks to run sequentially per array element.
    partial_dir : str
        Directory in which to write the partial NPZ.
    """
    tasks, axis_x_values, axis_y_values = build_tasks(
        axis_x_name=axis_x_name,
        axis_x_values=axis_x_values,
        axis_y_name=axis_y_name,
        axis_y_values=axis_y_values,
        fixed_overrides=fixed_overrides,
        nx=nx, ny=ny,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax, a=a, dt=dt,
        demag_kind=demag_kind, ic_list=ic_list,
    )
    n_total = len(tasks)
    start = array_task_id * sims_per_task
    end = min(start + sims_per_task, n_total)
    if start >= n_total:
        print(
            f'[array {array_task_id}] start={start} >= '
            f'n_total={n_total}; nothing to do.',
            flush=True,
        )
        return
    my_tasks = tasks[start:end]
    print(
        f'[array {array_task_id}] processing tasks '
        f'{start}..{end - 1} ({len(my_tasks)} of {n_total}).',
        flush=True,
    )
    t0 = time.time()
    results = []
    for n, task in enumerate(my_tasks, start=1):
        r = run_one(task)
        results.append(r)
        elapsed = time.time() - t0
        print(
            f'[array {array_task_id}] '
            f'{n}/{len(my_tasks)} done '
            f'(i={task["i"]} j={task["j"]} k={task["k"]} '
            f'ic={task["ic_name"]}), elapsed {elapsed:.1f}s.',
            flush=True,
        )
    os.makedirs(partial_dir, exist_ok=True)
    out_path = os.path.join(
        partial_dir, f'part_{array_task_id:05d}.npz',
    )
    p_probe = _build_probe(fixed_overrides, nx, ny, a, dt)
    _write_partial_npz(
        out_path, results,
        axis_x_name, axis_x_values,
        axis_y_name, axis_y_values,
        p_probe,
        ic_list=(_ic_specs() if ic_list is None else ic_list),
    )
    elapsed = time.time() - t0
    print(
        f'[array {array_task_id}] wrote {out_path} '
        f'({len(results)} records, {elapsed:.1f}s total).',
        flush=True,
    )


# ---------------------------------------------------------------------
def _write_partial_npz(path, results,
                       axis_x_name, axis_x_values,
                       axis_y_name, axis_y_values,
                       p_probe, ic_list=None):
    """Write per-task records to a partial NPZ.

    Parameters
    ----------
    path : str
        Output NPZ path.
    results : list[dict]
        Per-task result dicts returned by `run_one`.
    axis_x_name, axis_y_name : str
        Axis names (saved for aggregate-time checks).
    axis_x_values, axis_y_values : numpy.ndarray(1d)
        SI axis arrays (saved for aggregate-time checks).
    p_probe : SimpleNamespace
        Probe parameters namespace built from
        `fixed_overrides`. Used to record axis-invariant
        material scalars (Ms, A_ex, mu0, K_top, K_bot,
        H_RKKY, t_Co, d_Ru, a). The aggregator cross-checks
        these across partials.
    """
    if len(results) == 0:
        raise RuntimeError(
            f'Refusing to write empty partial NPZ at {path}.'
        )
    half_mu0_Ms2 = 0.5 * p_probe.mu0 * p_probe.Ms ** 2
    # Record the ACTUAL IC ensemble used for this sweep (the
    # caller's ic_list, or the 7-IC default). Must match the
    # per-record k indices, or the aggregator over-counts the
    # expected record total.
    ic_full = _ic_specs() if ic_list is None else list(ic_list)
    ic_names_full = np.array(
        [name for (name, _seed) in ic_full],
        dtype=object,
    )
    ic_seeds_full = np.array(
        [-1 if seed is None else int(seed)
         for (_name, seed) in ic_full],
        dtype=np.int32,
    )
    np.savez_compressed(
        path,
        axis_x_name=np.array(axis_x_name, dtype=object),
        axis_x_grid=axis_x_values,
        axis_y_name=np.array(axis_y_name, dtype=object),
        axis_y_grid=axis_y_values,
        # Full IC ensemble (every partial carries the
        # complete list so the aggregator can allocate
        # the (n_x, n_y, n_ic) arrays before scanning all
        # records).
        ic_names_full=ic_names_full,
        ic_seeds_full=ic_seeds_full,
        # Material scalars (axis-invariant for this sweep).
        Ms=np.float64(p_probe.Ms),
        A_ex=np.float64(p_probe.A_ex),
        mu0=np.float64(p_probe.mu0),
        half_mu0_Ms2=np.float64(half_mu0_Ms2),
        H_RKKY=np.float64(p_probe.H_RKKY),
        K_top_probe=np.float64(p_probe.K_top),
        K_bot_probe=np.float64(p_probe.K_bot),
        t_Co=np.float64(p_probe.t_Co),
        d_Ru=np.float64(p_probe.d_Ru),
        a=np.float64(p_probe.a),
        i=np.array([r['i'] for r in results], dtype=np.int32),
        j=np.array([r['j'] for r in results], dtype=np.int32),
        k=np.array([r['k'] for r in results], dtype=np.int32),
        axis_x_value=np.array(
            [r['axis_x_value'] for r in results],
        ),
        axis_y_value=np.array(
            [r['axis_y_value'] for r in results],
        ),
        ic_name=np.array(
            [r['ic_name'] for r in results], dtype=object,
        ),
        ic_seed=np.array(
            [r['ic_seed'] for r in results], dtype=np.int32,
        ),
        converged=np.array(
            [r['converged'] for r in results], dtype=bool,
        ),
        n_steps=np.array(
            [r['n_steps'] for r in results], dtype=np.int32,
        ),
        E=np.array([r['E'] for r in results]),
        tau_max=np.array([r['tau_max'] for r in results]),
        label=np.array(
            [r['label'] for r in results], dtype=object,
        ),
        mz_top=np.array([r['mz_top'] for r in results]),
        mz_bot=np.array([r['mz_bot'] for r in results]),
        m_dot=np.array([r['m_dot'] for r in results]),
        Q=np.array([r['Q'] for r in results]),
        k_star=np.array([r['k_star'] for r in results]),
        P_2=np.array([r['P_2'] for r in results]),
        P_6=np.array([r['P_6'] for r in results]),
        P_iso=np.array([r['P_iso'] for r in results]),
        peak_over_bg=np.array(
            [r['peak_over_bg'] for r in results],
        ),
        n_periods=np.array(
            [r['n_periods'] for r in results],
        ),
        q_per_period=np.array(
            [r['q_per_period'] for r in results],
        ),
        m_top=np.stack(
            [r['m_top'] for r in results], axis=0,
        ),
        m_bot=np.stack(
            [r['m_bot'] for r in results], axis=0,
        ),
    )


# ---------------------------------------------------------------------
def main():
    """Entry point for both local and SLURM-array modes.

    The block below ("User Configuration") is the only place
    users should edit. SLURM-array mode activates when the
    environment variable `SLURM_ARRAY_TASK_ID` is set;
    otherwise a single local process runs the full sweep.
    """
    # ================ User Configuration ================
    # Sweep axes. Both names must be keys of
    # `skyrmion_simulator.phase_diagram.axis_specs.axes`. Currently
    # registered: 'D', 'H_z', 'K_top', 'H_RKKY', 'Ms',
    # 'A_ex', 't_Co', 'd_Ru', 'alpha'.
    axis_x_name = 'D'
    # Cluster D points around the Bogdanov-Hubert
    # threshold D_c ~= 1.73 mJ/m^2 at K = 1.40 MJ/m^3
    # (K_eff ~= 0.115 MJ/m^3). 5 + 5 + 10 + 5 + 5 = 30
    # monotonic points; the inner 10 fall in
    # D_c +- 0.2 mJ/m^2 where the chiral phase boundaries
    # actually live.
    _D_c = 1.73e-3
    axis_x_values = np.concatenate([
        np.linspace(0.0,         _D_c - 0.6e-3, 5,
                    endpoint=False),
        np.linspace(_D_c - 0.6e-3, _D_c - 0.2e-3, 5,
                    endpoint=False),
        np.linspace(_D_c - 0.2e-3, _D_c + 0.2e-3, 10,
                    endpoint=False),
        np.linspace(_D_c + 0.2e-3, _D_c + 0.6e-3, 5,
                    endpoint=False),
        np.linspace(_D_c + 0.6e-3, 4.0e-3, 5),
    ])
    axis_y_name = 'H_z'
    # Cluster H_z points around the spin-flop boundaries
    # |H_z| = H_RKKY = 0.205 T. 4 + 9 + 4 + 9 + 4 = 30
    # monotonic points; the 9-point windows around each
    # transition resolve it to ~0.013 T, ~2x finer than a
    # uniform 30-point grid would.
    _H_sf = 0.205
    axis_y_values = np.concatenate([
        np.linspace(-0.5,         -_H_sf - 0.05, 4,
                    endpoint=False),
        np.linspace(-_H_sf - 0.05, -_H_sf + 0.05, 9,
                    endpoint=False),
        np.linspace(-_H_sf + 0.05, +_H_sf - 0.05, 4,
                    endpoint=False),
        np.linspace(+_H_sf - 0.05, +_H_sf + 0.05, 9,
                    endpoint=False),
        np.linspace(+_H_sf + 0.05, +0.5, 4),
    ])
    # Constants applied to every task. Must not collide
    # with the keys produced by either axis's override
    # builder (e.g. if axis_y_name == 'K_top', do not set
    # K_top / K_bot here).
    fixed_overrides = {
        'K_top': 1.40e6,
        'K_bot': 1.40e6,
    }
    # Lattice size in cells. At a = 1.0 nm the physical
    # box is L = nx * a = 256 nm.
    nx = 256
    ny = 256
    # Lattice constant in metres. a = 1.0 nm resolves
    # Delta_DW ~ 7.13 nm with 7 cells for K = 1.6 MJ/m^3.
    a = 1.0e-9
    # Time step in seconds. RK4 stability requires
    # dt < ~0.1 / (gamma * H_max). At a = 1 nm,
    # C_ex = 2 A / (Ms a^2) is 4x the a = 2 nm value, so
    # dt must shrink correspondingly.
    dt = 2.0e-14
    # Relaxation safety cutoff and convergence tolerances.
    max_steps = 100000
    tol_torque = 1e-3
    tol_dE = 1e-7
    # Gilbert damping override during relaxation. Use 1.0
    # for fast over-damped quench.
    alpha_relax = 1.0
    # Process-pool worker count for LOCAL mode. Ignored in
    # SLURM-array mode. Set to None to use os.cpu_count().
    workers = None
    # Number of tasks to run sequentially per SLURM array
    # element.
    sims_per_task = 1
    # Output paths. Suffixed with the material's K so the
    # K = 1.40 MJ/m^3 results live alongside the existing
    # K = 1.60 MJ/m^3 sweep at output/phase_diagram/D_H_z.npz
    # rather than overwriting it.
    out_path = 'output/phase_diagram/D_H_z_K1p4e6Jm3.npz'
    partial_dir = (
        'output/phase_diagram/D_H_z_K1p4e6Jm3_partials'
    )
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
    if partial_dir is None:
        partial_dir = os.path.join(
            'output', 'phase_diagram',
            f'{axis_x_name}_{axis_y_name}_partials',
        )
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
