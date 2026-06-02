"""Ensemble runner over independent stochastic trajectories.

Wraps `concurrent.futures.ProcessPoolExecutor` to dispatch one
trajectory per worker. Each worker is fully isolated and
constructs its own RNG from a per-trajectory seed
(`seed_base + traj_idx`) so single trajectories can be
re-run byte-for-byte without re-running the rest of the
ensemble.

Functions
---------
run_ensemble
    Dispatch `n_ens` independent calls of `worker_fn` over a
    worker pool and return the list of results in submission
    order.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def run_ensemble(worker_fn, base_config, n_ens, seed_base,
                 n_workers):
    """Dispatch `n_ens` worker calls in parallel.

    Parameters
    ----------
    worker_fn : callable
        Worker callable with signature `worker_fn(config)`.
        `config` is a dict (picklable). The worker must
        return a picklable result. The function must be
        importable by the worker process (no closures, no
        lambdas).
    base_config : dict
        Shared configuration. The runner attaches
        `'seed': seed_base + traj_idx` and
        `'traj_idx': traj_idx` to a deep copy of this dict
        for each worker.
    n_ens : int
        Number of trajectories. Strictly positive.
    seed_base : int
        Integer base seed. Trajectory `i` uses
        `seed_base + i`.
    n_workers : int
        Process-pool size. Strictly positive.

    Returns
    -------
    results : list
        `n_ens` worker return values, ordered by trajectory
        index.
    """
    if not callable(worker_fn):
        raise RuntimeError(
            f'run_ensemble: worker_fn must be callable, got '
            f'{type(worker_fn).__name__}.'
        )
    if not isinstance(base_config, dict):
        raise RuntimeError(
            f'run_ensemble: base_config must be a dict, '
            f'got {type(base_config).__name__}.'
        )
    if not isinstance(n_ens, (int, np.integer)) \
            or n_ens <= 0:
        raise RuntimeError(
            f'run_ensemble: n_ens must be a positive int, '
            f'got {n_ens!r}.'
        )
    if not isinstance(n_workers, (int, np.integer)) \
            or n_workers <= 0:
        raise RuntimeError(
            f'run_ensemble: n_workers must be a positive '
            f'int, got {n_workers!r}.'
        )
    if not isinstance(seed_base, (int, np.integer)):
        raise RuntimeError(
            f'run_ensemble: seed_base must be int, got '
            f'{type(seed_base).__name__}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # One deep-copied config per trajectory, each tagged with its
    # own seed (seed_base + idx) and trajectory index.
    configs = []
    for idx in range(int(n_ens)):
        cfg = copy.deepcopy(base_config)
        cfg['seed'] = int(seed_base) + idx
        cfg['traj_idx'] = idx
        configs.append(cfg)
    results = [None] * int(n_ens)
    # Single-worker fast path: run inline, no pool overhead.
    if int(n_workers) == 1:
        for idx, cfg in enumerate(configs):
            results[idx] = worker_fn(cfg)
        return results
    # Parallel path: scatter over the pool, gather back into
    # submission order via the future-to-index map.
    with ProcessPoolExecutor(max_workers=int(n_workers)) as pool:
        futures = {
            pool.submit(worker_fn, cfg): idx
            for idx, cfg in enumerate(configs)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
    return results
