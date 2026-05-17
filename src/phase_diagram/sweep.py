"""Parallel parameter sweep over (D, H_z).

For each (D, H_z) point on the configured grid, six initial
conditions are relaxed independently and the lowest-energy
converged state is taken as the ground state. Per-IC
observables and the ground-state spin textures are written
to `output/phase_diagram/<grid>.npz`.

Two execution modes share a single entry point:

* Local mode (default): one process aggregates everything
  into `<grid>.npz` via a `ProcessPoolExecutor`.
* SLURM array mode: each array element runs the slice of
  tasks it owns and writes a partial NPZ. The partials are
  later combined into the same final NPZ schema by
  `src.phase_diagram.aggregate`. Triggered automatically
  when `SLURM_ARRAY_TASK_ID` is set in the environment.

Configuration lives at the top of `main()`. Edit the
variables in that block to change grid, lattice size,
tolerances, etc., then submit:

    python -m src.phase_diagram.sweep                   # local
    sbatch scripts/submit_sweep_array.sh                # cluster

Functions
---------
build_tasks
    Build the list of (D, H_z, IC) tasks for a given grid.
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
from src.phase_diagram.classifier import classify, PHASE_LABELS
from src.phase_diagram.fields_demag import (
    effective_field_demag_pair,  # noqa: F401  (used in workers)
)
from src.phase_diagram.initial_conditions_ext import (
    random_state,
    stripe_state,
)
from src.phase_diagram.params_helper import make_params
from src.phase_diagram.relaxation import relax
from src.simulator.demag import precompute_demag_kernels
from src.simulator.energy import (
    critical_dmi,
    effective_anisotropy,
    pma_anisotropy_field,
)
from src.simulator.initial_conditions import (
    saf_skyrmion,
    uniform_state,
)

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================
_PHASE_LABELS = PHASE_LABELS


def _grid(name):
    """Return (D_array, H_array) for a named grid.

    D ranges to 6 mJ/m^2 so D/D_c reaches ~2.1 when
    K = 1.6 MJ/m^3 (D_c ~ 2.86 mJ/m^2) — past the
    Bogdanov-Hubert chiral threshold D/D_c = 1. With the
    much weaker default K (D_c ~ 0.67 mJ/m^2) this range
    is well into the spiral/SkX regime already.
    """
    if name == 'coarse':
        D = np.linspace(0.0, 6.0e-3, 20)
        H = np.linspace(-0.5, 0.5, 20)
    elif name == 'fine':
        D = np.linspace(0.0, 6.0e-3, 50)
        H = np.linspace(-0.5, 0.5, 50)
    elif name == 'medium':
        D = np.linspace(0.0, 6.0e-3, 8)
        H = np.linspace(-0.5, 0.5, 8)
    elif name == 'test':
        D = np.linspace(0.0, 6.0e-3, 4)
        H = np.linspace(-0.5, 0.5, 4)
    else:
        raise RuntimeError(
            f"Unknown grid {name!r}. "
            f'Use coarse, medium, fine, or test.'
        )
    return D, H


def _ic_specs():
    """Return the canonical 7-member IC ensemble.

    `fm_anti` seeds the antiparallel SAF (RKKY ground
    state) so the relaxation can find the antiparallel-FM
    branch; `fm_par` seeds both layers aligned with the
    field so the relaxation can find the parallel-FM branch
    above the spin-flop (|H_z| > H_RKKY).
    """
    return (
        ('random', 11),
        ('random', 22),
        ('random', 33),
        ('fm_anti', None),
        ('fm_par', None),
        ('skyrmion', None),
        ('stripe', None),
    )


# ---------------------------------------------------------------------
def build_tasks(grid_name, nx=None, ny=None,
                max_steps=None, tol_torque=None, tol_dE=None,
                alpha_relax=None,
                K_top=None, K_bot=None):
    """Build the flat task list for a grid sweep.

    Returns
    -------
    tasks : list of dict
        Each dict carries everything a worker needs.
    D_arr, H_arr : numpy.ndarray(1d)
        The grid axes.
    """
    D_arr, H_arr = _grid(grid_name)
    ic_list = _ic_specs()
    tasks = []
    for i, D_val in enumerate(D_arr):
        for j, H_z in enumerate(H_arr):
            for k, (ic_name, ic_seed) in enumerate(ic_list):
                tasks.append({
                    'i': i, 'j': j, 'k': k,
                    'D': float(D_val),
                    'H_z': float(H_z),
                    'ic_name': ic_name,
                    'ic_seed': ic_seed,
                    'nx': nx, 'ny': ny,
                    'max_steps': max_steps,
                    'tol_torque': tol_torque,
                    'tol_dE': tol_dE,
                    'alpha_relax': alpha_relax,
                    'K_top': K_top,
                    'K_bot': K_bot,
                })
    return tasks, D_arr, H_arr


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
        m_top = uniform_state(
            p.nx, p.ny, np.array([0.0, 0.0, sign]),
        )
        m_bot = uniform_state(
            p.nx, p.ny, np.array([0.0, 0.0, -sign]),
        )
        return m_top, m_bot
    if ic_name == 'fm_par':
        # Parallel FM aligned with the field. Lets the
        # relaxation find the spin-flop transition above
        # |H_z| > H_RKKY where parallel-FM beats
        # antiparallel-FM (Zeeman gain > RKKY cost).
        sign = -1.0 if p.H_ext[2] < 0.0 else 1.0
        m_top = uniform_state(
            p.nx, p.ny, np.array([0.0, 0.0, sign]),
        )
        m_bot = uniform_state(
            p.nx, p.ny, np.array([0.0, 0.0, sign]),
        )
        return m_top, m_bot
    if ic_name == 'skyrmion':
        return saf_skyrmion(
            p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw,
        )
    if ic_name == 'stripe':
        # lambda = 4*pi*A_ex / D ; floor at half lattice
        if p.D > 0.0:
            period = 4.0 * np.pi * p.A_ex / p.D
        else:
            period = 4.0 * np.pi * p.A_ex / 1e-5
        period = max(period, 4.0 * p.a)
        return stripe_state(p.nx, p.ny, period, p.a)
    raise RuntimeError(f'Unknown IC {ic_name!r}.')


# ---------------------------------------------------------------------
def run_one(task):
    """Worker: relax + classify a single task."""
    overrides = {
        'D': task['D'],
        'H_ext': np.array([0.0, 0.0, task['H_z']]),
        'J_current': 0.0,
    }
    if task['nx'] is not None:
        overrides['nx'] = int(task['nx'])
    if task['ny'] is not None:
        overrides['ny'] = int(task['ny'])
    if task.get('K_top') is not None:
        overrides['K_top'] = float(task['K_top'])
    if task.get('K_bot') is not None:
        overrides['K_bot'] = float(task['K_bot'])
    p = make_params(**overrides)
    kernels = precompute_demag_kernels(p)
    m_top, m_bot = _build_ic(
        task['ic_name'], task['ic_seed'], p,
    )
    relax_kwargs = {}
    if task['max_steps'] is not None:
        relax_kwargs['max_steps'] = int(task['max_steps'])
    if task['tol_torque'] is not None:
        relax_kwargs['tol_torque'] = float(task['tol_torque'])
    if task['tol_dE'] is not None:
        relax_kwargs['tol_dE'] = float(task['tol_dE'])
    if task['alpha_relax'] is not None:
        relax_kwargs['alpha_relax'] = float(task['alpha_relax'])
    m_top, m_bot, converged, n_steps, E, tau = relax(
        m_top, m_bot, p, kernels, **relax_kwargs,
    )
    label, obs = classify(m_top, m_bot, p)
    return {
        'i': task['i'], 'j': task['j'], 'k': task['k'],
        'D': task['D'], 'H_z': task['H_z'],
        'ic_name': task['ic_name'],
        'ic_seed': (
            -1 if task['ic_seed'] is None else int(task['ic_seed'])
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
        'label_idx_per_ic': np.full(
            (n_D, n_H, n_IC), -1, dtype=np.int8,
        ),
        'gs_idx': np.full((n_D, n_H), -1, dtype=np.int8),
        'gs_label_idx': np.full(
            (n_D, n_H), -1, dtype=np.int8,
        ),
        'gs_m_top': np.zeros(
            (n_D, n_H, ny, nx, 3), dtype=np.float32,
        ),
        'gs_m_bot': np.zeros(
            (n_D, n_H, ny, nx, 3), dtype=np.float32,
        ),
    }


def sweep(grid_name='coarse', nx=None, ny=None,
          workers=None, out_path=None,
          max_steps=None, tol_torque=None, tol_dE=None,
          alpha_relax=None, K_top=None, K_bot=None,
          verbose=True):
    """Run the (D, H_z) sweep and write `<grid>.npz`."""
    tasks, D_arr, H_arr = build_tasks(
        grid_name, nx=nx, ny=ny,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax,
        K_top=K_top, K_bot=K_bot,
    )
    if not tasks:
        raise RuntimeError('Empty task list.')
    n_D, n_H = len(D_arr), len(H_arr)
    n_IC = len(_ic_specs())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Probe lattice size and material constants from the
    # first task to allocate aggregation arrays and compute
    # reduced-unit normalizations.
    probe_kwargs = {}
    if nx is not None:
        probe_kwargs['nx'] = int(nx)
    if ny is not None:
        probe_kwargs['ny'] = int(ny)
    if K_top is not None:
        probe_kwargs['K_top'] = float(K_top)
    if K_bot is not None:
        probe_kwargs['K_bot'] = float(K_bot)
    p_probe = make_params(**probe_kwargs)
    arr = _allocate_arrays(n_D, n_H, n_IC, p_probe.ny, p_probe.nx)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reduced-unit normalizations. These are material
    # (not field) dependent, so D_c and H_K are scalars for
    # the whole sweep at fixed K, Ms, A_ex.
    K_eff_dict = effective_anisotropy(p_probe)
    D_c = critical_dmi(p_probe)
    H_K = pma_anisotropy_field(p_probe)
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
    best_E = np.full((n_D, n_H), np.inf)
    if workers <= 1:
        iterator = (run_one(t) for t in tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        futures = [pool.submit(run_one, t) for t in tasks]
        iterator = (f.result() for f in as_completed(futures))
    for r in iterator:
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
        arr['label_idx_per_ic'][i, j, k] = _label_to_int(
            r['label']
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update running ground state if this IC converged
        # to a lower energy than any seen so far at (i, j).
        if r['converged'] and r['E'] < best_E[i, j]:
            best_E[i, j] = r['E']
            arr['gs_idx'][i, j] = k
            arr['gs_label_idx'][i, j] = _label_to_int(
                r['label']
            )
            arr['gs_m_top'][i, j] = r['m_top']
            arr['gs_m_bot'][i, j] = r['m_bot']
        n_done += 1
        if verbose and (n_done % max(1, n_total // 50) == 0):
            dt = time.time() - t0
            print(
                f'  {n_done}/{n_total} '
                f'({n_done / n_total * 100:5.1f}%) '
                f'elapsed {dt:6.1f}s',
                flush=True,
            )
    if workers > 1:
        pool.shutdown(wait=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write NPZ
    if out_path is None:
        out_path = os.path.join(
            'output', 'phase_diagram', f'{grid_name}.npz',
        )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        D=D_arr, H_z=H_arr,
        ic_names=np.array(
            [s[0] for s in _ic_specs()],
            dtype=object,
        ),
        labels=np.array(_PHASE_LABELS, dtype=object),
        # Material-derived reduced-unit normalizations
        D_c=np.float64(D_c),
        H_K=np.float64(H_K),
        K_top_raw=np.float64(p_probe.K_top),
        K_bot_raw=np.float64(p_probe.K_bot),
        K_eff_top=np.float64(K_eff_dict['K_eff_top']),
        K_eff_bot=np.float64(K_eff_dict['K_eff_bot']),
        K_eff_avg=np.float64(K_eff_dict['K_eff_avg']),
        **{k: v for k, v in arr.items()},
    )
    if verbose:
        dt = time.time() - t0
        print(
            f'Wrote {out_path} ({dt:.1f}s wall, '
            f'{n_total} tasks).'
        )
    return out_path


# ---------------------------------------------------------------------
def sweep_array_partial(grid_name, nx, ny, max_steps,
                        tol_torque, tol_dE, alpha_relax,
                        array_task_id, sims_per_task,
                        partial_dir,
                        K_top=None, K_bot=None):
    """Run one SLURM array element's slice of tasks.

    Each array element processes `sims_per_task` consecutive
    tasks starting at `array_task_id * sims_per_task` and
    writes a partial NPZ named `part_<id>.npz` in
    `partial_dir`. The partial NPZ stores one record per
    task (no aggregation); the final NPZ is built by
    `src.phase_diagram.aggregate` after all array elements
    finish.

    Parameters
    ----------
    grid_name : str
        Grid resolution: 'coarse', 'medium', 'fine', 'test'.
    nx : {int, None}
        Lattice size along x; None inherits the default.
    ny : {int, None}
        Lattice size along y; None inherits the default.
    max_steps : {int, None}
        Relaxation safety cutoff; None inherits the default.
    tol_torque : {float, None}
        Torque convergence threshold; None for default.
    tol_dE : {float, None}
        Energy-drift convergence threshold; None for default.
    alpha_relax : {float, None}
        Gilbert damping override during relaxation; None for
        default.
    array_task_id : int
        SLURM_ARRAY_TASK_ID for this element.
    sims_per_task : int
        Number of tasks to run sequentially per array element.
    partial_dir : str
        Directory in which to write the partial NPZ.
    K_top, K_bot : {float, None}, default=None
        Per-layer raw anisotropy overrides in J/m^3.
        `None` keeps the namespace defaults.
    """
    tasks, D_arr, H_arr = build_tasks(
        grid_name, nx=nx, ny=ny,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax,
        K_top=K_top, K_bot=K_bot,
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
        dt = time.time() - t0
        print(
            f'[array {array_task_id}] '
            f'{n}/{len(my_tasks)} done '
            f'(i={task["i"]} j={task["j"]} k={task["k"]} '
            f'ic={task["ic_name"]}), elapsed {dt:.1f}s.',
            flush=True,
        )
    os.makedirs(partial_dir, exist_ok=True)
    out_path = os.path.join(
        partial_dir, f'part_{array_task_id:05d}.npz',
    )
    _write_partial_npz(out_path, results, D_arr, H_arr)
    dt = time.time() - t0
    print(
        f'[array {array_task_id}] wrote {out_path} '
        f'({len(results)} records, {dt:.1f}s total).',
        flush=True,
    )


# ---------------------------------------------------------------------
def _write_partial_npz(path, results, D_arr, H_arr):
    """Write per-task records to a partial NPZ.

    Parameters
    ----------
    path : str
        Output NPZ path.
    results : list[dict]
        Per-task result dicts returned by `run_one`.
    D_arr : numpy.ndarray(1d)
        Grid axis along D (saved for aggregate-time checks).
    H_arr : numpy.ndarray(1d)
        Grid axis along H_z (saved for aggregate-time
        checks).

    Notes
    -----
    Stores one record per task. Final aggregation into the
    canonical `(n_D, n_H, n_IC)` arrays is performed by
    `src.phase_diagram.aggregate` once every array element
    has finished.
    """
    if len(results) == 0:
        raise RuntimeError(
            f'Refusing to write empty partial NPZ at {path}.'
        )
    np.savez_compressed(
        path,
        D_grid=D_arr, H_z_grid=H_arr,
        i=np.array([r['i'] for r in results], dtype=np.int32),
        j=np.array([r['j'] for r in results], dtype=np.int32),
        k=np.array([r['k'] for r in results], dtype=np.int32),
        D=np.array([r['D'] for r in results]),
        H_z=np.array([r['H_z'] for r in results]),
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
    # Grid resolution along (D, H_z): one of 'coarse',
    # 'medium', 'fine', 'test'.
    grid_name = 'medium'
    # Lattice size. Set nx = ny = None to inherit the
    # namespace default (256x256). Use 128x128 for fast
    # medium runs, 256x256 or 512x512 for production.
    nx = 128
    ny = 128
    # Relaxation safety cutoff and convergence tolerances.
    max_steps = 20000
    tol_torque = 1e-4
    tol_dE = 1e-7
    # Gilbert damping override during relaxation. Use 1.0
    # for fast over-damped quench to the (meta)stable state.
    alpha_relax = 1.0
    # Process-pool worker count for LOCAL mode. Ignored in
    # SLURM-array mode (each array element is a single
    # worker on a single CPU). Set to None to use
    # os.cpu_count().
    workers = 10
    # Number of tasks to run sequentially per SLURM array
    # element. Use 1 unless the cluster's MaxArraySize is
    # smaller than the total task count; in that case set
    # sims_per_task = ceil(n_tasks / max_array_size).
    sims_per_task = 1
    # Output paths. Set to None to use the standard layout.
    # Local mode writes to out_path (default
    # output/phase_diagram/<grid>.npz). SLURM-array mode
    # writes one partial NPZ to partial_dir (default
    # output/phase_diagram/<grid>_partials/).
    out_path = None
    partial_dir = None
    # Material anisotropy override (J/m^3) per layer.
    # `None` keeps the value from `default_params()`.
    # Use `Q_PMA` below for a physics-targeted override
    # (mutually exclusive with explicit K_top / K_bot).
    # K = 1.60e6 J/m^3 -> K_eff ~ 0.315 MJ/m^3, Q_PMA ~ 0.25,
    # D_c ~ 2.86 mJ/m^2, H_K ~ 0.45 T (Pt/Co/Ru/Co regime).
    K_top = 1.60e6
    K_bot = 1.60e6
    # Target PMA quality factor Q_PMA = K_eff / (mu0 Ms^2/2).
    # If set (not None), both K_top and K_bot are computed
    # as K = Q_PMA * (mu0 Ms^2/2) + (mu0 Ms^2/2) and the
    # explicit K_top / K_bot above must be None.
    # Typical values: 0.007 (default material), 0.25 for
    # Pt/Co/Ru/Co with strong interface PMA.
    Q_PMA = None
    # ============ End User Configuration =================
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Resolve K from Q_PMA if requested. The half_mu0_Ms2
    # term is read from a probe namespace so the convention
    # stays in sync with `effective_anisotropy(p)`.
    if Q_PMA is not None:
        if K_top is not None or K_bot is not None:
            raise RuntimeError(
                'Set either Q_PMA or explicit K_top/K_bot, '
                'not both.'
            )
        p_probe = make_params()
        half_mu0_Ms2 = effective_anisotropy(p_probe)[
            'mu0_Ms2_over_2'
        ]
        K_resolved = float(Q_PMA) * half_mu0_Ms2 + half_mu0_Ms2
        K_top = K_resolved
        K_bot = K_resolved
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    array_id_env = os.environ.get('SLURM_ARRAY_TASK_ID')
    if array_id_env is None:
        sweep(
            grid_name=grid_name, nx=nx, ny=ny,
            workers=workers, out_path=out_path,
            max_steps=max_steps,
            tol_torque=tol_torque, tol_dE=tol_dE,
            alpha_relax=alpha_relax,
            K_top=K_top, K_bot=K_bot,
        )
        return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SLURM-array mode: process this element's slice of
    # tasks and write a partial NPZ.
    array_task_id = int(array_id_env)
    if partial_dir is None:
        partial_dir = os.path.join(
            'output', 'phase_diagram',
            f'{grid_name}_partials',
        )
    sweep_array_partial(
        grid_name=grid_name, nx=nx, ny=ny,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax,
        array_task_id=array_task_id,
        sims_per_task=sims_per_task,
        partial_dir=partial_dir,
        K_top=K_top, K_bot=K_bot,
    )


# =====================================================================
if __name__ == '__main__':
    sys.exit(main())
