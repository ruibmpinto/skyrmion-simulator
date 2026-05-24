"""NPZ I/O helpers for stochastic-LLGS production runs.

Lightweight wrappers around `numpy.savez_compressed` that
attach a uniform header (version, timestamp, configuration
echo) so downstream analysis can dispatch on the file's
provenance without re-deriving it from filename conventions.

Functions
---------
save_trajectory
    Persist a single-trajectory result dict to NPZ.
save_grid_scan
    Persist an aggregated grid-scan result to NPZ.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import time
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
_SCHEMA_VERSION = 'stochastic_llgs/0.1'


def save_trajectory(out_path, payload, config):
    """Persist a trajectory NPZ.

    Parameters
    ----------
    out_path : str
        Output file path. Parent directories must exist.
    payload : dict
        Arrays and scalars to save. Keys with reserved
        prefix `meta_` are forbidden (used for metadata).
    config : dict
        Run-configuration dict; echoed under `meta_config`.
    """
    if not isinstance(out_path, str) or not out_path:
        raise RuntimeError(
            f'save_trajectory: out_path must be a non-empty '
            f'string, got {out_path!r}.'
        )
    if not isinstance(payload, dict):
        raise RuntimeError(
            f'save_trajectory: payload must be a dict, got '
            f'{type(payload).__name__}.'
        )
    if not isinstance(config, dict):
        raise RuntimeError(
            f'save_trajectory: config must be a dict, got '
            f'{type(config).__name__}.'
        )
    for k in payload:
        if k.startswith('meta_'):
            raise RuntimeError(
                f'save_trajectory: payload key {k!r} uses '
                f'reserved prefix `meta_`.'
            )
    parent = os.path.dirname(out_path)
    if parent and not os.path.isdir(parent):
        raise RuntimeError(
            f'save_trajectory: parent directory '
            f'{parent!r} does not exist.'
        )
    record = dict(payload)
    record['meta_schema_version'] = _SCHEMA_VERSION
    record['meta_timestamp'] = float(time.time())
    record['meta_config_repr'] = repr(config)
    np.savez_compressed(out_path, **record)


# -----------------------------------------------------------------------------
def save_grid_scan(out_path, payload, config):
    """Persist an aggregated grid-scan NPZ.

    Identical interface to `save_trajectory`; separated for
    clarity at the call site (and so a future bump to the
    grid-scan schema does not touch the trajectory schema).
    """
    save_trajectory(out_path, payload, config)
