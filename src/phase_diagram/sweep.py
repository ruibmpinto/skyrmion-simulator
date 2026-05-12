"""Parallel parameter sweep over (D, H_z).

For each (D, H_z) point on the configured grid, six initial
conditions are relaxed independently and the lowest-energy
converged state is taken as the ground state. Per-IC
observables and the ground-state spin textures are written
to `output/phase_diagram/<grid>.npz`.

Usage
-----
Run from the project root:

    python -m src.phase_diagram.sweep --grid coarse
    python -m src.phase_diagram.sweep --grid fine
    python -m src.phase_diagram.sweep --grid test --nx 32 --ny 32

CLI flags
---------
--grid {coarse,fine,test}
    Selects the (D, H_z) grid resolution.
--nx, --ny INT
    Override lattice size.
--workers INT
    Process-pool worker count. Defaults to os.cpu_count().
--max-steps INT
    Override the relaxation safety cutoff.
--tol-torque FLOAT, --tol-de FLOAT
    Override the relaxation tolerances.
--out PATH
    Output NPZ path. Default
    `output/phase_diagram/<grid>.npz`.

Functions
---------
build_tasks
    Build the list of (D, H_z, IC) tasks for a given grid.
run_one
    Worker: relax + classify a single task and return
    observables and final spin pair.
sweep
    Orchestrator: dispatch tasks, aggregate ground states,
    write NPZ.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import argparse
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
    """Return (D_array, H_array) for a named grid."""
    if name == 'coarse':
        D = np.linspace(0.0, 2.0e-3, 20)
        H = np.linspace(-0.5, 0.5, 20)
    elif name == 'fine':
        D = np.linspace(0.0, 2.0e-3, 50)
        H = np.linspace(-0.5, 0.5, 50)
    elif name == 'medium':
        D = np.linspace(0.0, 2.0e-3, 8)
        H = np.linspace(-0.5, 0.5, 8)
    elif name == 'test':
        D = np.linspace(0.0, 2.0e-3, 4)
        H = np.linspace(-0.5, 0.5, 4)
    else:
        raise RuntimeError(
            f"Unknown grid {name!r}. "
            f'Use coarse, medium, fine, or test.'
        )
    return D, H


def _ic_specs():
    """Return the canonical 6-member IC ensemble."""
    return (
        ('random', 11),
        ('random', 22),
        ('random', 33),
        ('fm', None),
        ('skyrmion', None),
        ('stripe', None),
    )


# ---------------------------------------------------------------------
def build_tasks(grid_name, nx=None, ny=None,
                max_steps=None, tol_torque=None, tol_dE=None,
                alpha_relax=None):
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
                })
    return tasks, D_arr, H_arr


# ---------------------------------------------------------------------
def _build_ic(ic_name, ic_seed, p):
    """Materialize an initial-condition pair for parameters p."""
    if ic_name == 'random':
        return random_state(p.nx, p.ny, ic_seed)
    if ic_name == 'fm':
        # Align the SAF FM polarity with the applied field
        # so the IC explores the field-favored FM basin
        # rather than being trapped in the wrong-sign one.
        sign = -1.0 if p.H_ext[2] < 0.0 else 1.0
        m_top = uniform_state(
            p.nx, p.ny, np.array([0.0, 0.0, sign]),
        )
        m_bot = uniform_state(
            p.nx, p.ny, np.array([0.0, 0.0, -sign]),
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
          alpha_relax=None, verbose=True):
    """Run the (D, H_z) sweep and write `<grid>.npz`."""
    tasks, D_arr, H_arr = build_tasks(
        grid_name, nx=nx, ny=ny,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax,
    )
    if not tasks:
        raise RuntimeError('Empty task list.')
    n_D, n_H = len(D_arr), len(H_arr)
    n_IC = len(_ic_specs())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Probe lattice size from the first task to allocate
    p_probe = make_params(
        nx=int(nx) if nx else None,
        ny=int(ny) if ny else None,
    ) if (nx is not None or ny is not None) else make_params()
    arr = _allocate_arrays(n_D, n_H, n_IC, p_probe.ny, p_probe.nx)
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
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Sweep the (D, H_z) phase diagram of the SAF '
            'skyrmion simulator.'
        ),
    )
    parser.add_argument(
        '--grid',
        choices=('coarse', 'medium', 'fine', 'test'),
        default='coarse',
    )
    parser.add_argument('--nx', type=int, default=None)
    parser.add_argument('--ny', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument(
        '--max-steps', type=int, default=None,
        dest='max_steps',
    )
    parser.add_argument(
        '--tol-torque', type=float, default=None,
        dest='tol_torque',
    )
    parser.add_argument(
        '--tol-de', type=float, default=None, dest='tol_dE',
    )
    parser.add_argument(
        '--alpha-relax', type=float, default=None,
        dest='alpha_relax',
        help='Override Gilbert damping during relaxation.',
    )
    parser.add_argument('--out', type=str, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    sweep(
        grid_name=args.grid, nx=args.nx, ny=args.ny,
        workers=args.workers, out_path=args.out,
        max_steps=args.max_steps,
        tol_torque=args.tol_torque, tol_dE=args.tol_dE,
        alpha_relax=args.alpha_relax,
    )


# =====================================================================
if __name__ == '__main__':
    sys.exit(main())
