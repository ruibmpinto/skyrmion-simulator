"""NPZ persistence for sweep traces with explicit metadata.

`save_trace` writes one NPZ file containing every key in the
trace dict produced by `driver.run_one` plus a `_metadata`
key holding the JSON-serialised metadata dict. `load_trace`
recovers `(trace, metadata)`.

Functions
---------
save_trace
    Write a trace dict and a metadata dict to NPZ.
load_trace
    Read an NPZ written by `save_trace` and return the trace
    plus the metadata.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import json
import os
# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

# Reserved key inside the NPZ used to embed the metadata dict.
# A leading underscore avoids collision with any observable key.
_METADATA_KEY = '_metadata'
# Sentinel string written for keys whose value is None (NPZ
# cannot store None directly).
_NONE_SENTINEL = '__NONE__'


def save_trace(path, trace, metadata):
    """Write a trace and its metadata to NPZ.

    Parameters
    ----------
    path : str
        Destination path. Parent directories must already
        exist; `save_trace` does not create them.
    trace : dict
        Trace returned by `driver.run_one`. Values that are
        `None` are stored as a sentinel string and decoded back
        to `None` by `load_trace`.
    metadata : dict
        Sweep-grid metadata (e.g. `{'J0': 1e11, 'FWHM': 5e-10}`).
        Required; pass `{}` to record nothing. Stored as
        JSON inside the NPZ under `_METADATA_KEY`.

    Raises
    ------
    RuntimeError
        If `trace` already contains the reserved metadata key.
    """
    # Loud rejection of a missing metadata argument; the caller
    # must opt into "no metadata" explicitly by passing {}.
    if metadata is None:
        raise RuntimeError(
            'save_trace: `metadata` is required (pass {} for none).')
    # Reject collision with the metadata key so the user does
    # not silently lose metadata to a name clash.
    if _METADATA_KEY in trace:
        raise RuntimeError(
            f'save_trace: trace already contains reserved key '
            f'{_METADATA_KEY!r}.')
    # Confirm parent directory exists; do not create silently.
    parent = os.path.dirname(path) or '.'
    if not os.path.isdir(parent):
        raise RuntimeError(
            f'save_trace: parent directory {parent!r} does not '
            f'exist; create it before calling save_trace.')
    # Encode the trace: replace None values with the sentinel
    # so np.savez does not crash on None.
    encoded = {}
    for key, value in trace.items():
        if value is None:
            encoded[key] = np.asarray(_NONE_SENTINEL)
        else:
            encoded[key] = np.asarray(value)
    # JSON-serialise the metadata and store it as a 0-d string array.
    encoded[_METADATA_KEY] = np.asarray(json.dumps(metadata))
    # np.savez_compressed keeps file sizes small without
    # changing the load semantics.
    np.savez_compressed(path, **encoded)


# -----------------------------------------------------------------------------
def load_trace(path):
    """Read an NPZ written by `save_trace`.

    Parameters
    ----------
    path : str
        Source path. Must exist.

    Returns
    -------
    trace : dict
        Same shape as the input to `save_trace`. None-valued
        keys are restored to Python `None`.
    metadata : dict
        Sweep metadata previously passed to `save_trace`.

    Raises
    ------
    RuntimeError
        If the NPZ does not contain the metadata key (i.e. it
        was not written by `save_trace`).
    """
    # `np.load` raises on missing path already; rely on that.
    npz = np.load(path, allow_pickle=False)
    # Detect files not written by `save_trace`.
    if _METADATA_KEY not in npz.files:
        raise RuntimeError(
            f'load_trace: {path!r} does not contain the reserved '
            f'metadata key {_METADATA_KEY!r}; was it written by '
            f'save_trace?')
    # Decode metadata first so trace decoding can run unaffected.
    # Two writers exist: the Python save_trace stores a 0-d unicode
    # array; the C++ sweep binaries store a 1-d uint8 array of UTF-8
    # JSON bytes (libnpy cannot emit 0-d unicode). Handle both.
    meta_arr = npz[_METADATA_KEY]
    if meta_arr.dtype.kind in ('U', 'S'):
        metadata = json.loads(str(meta_arr))
    elif meta_arr.dtype.kind in ('u', 'i'):
        metadata = json.loads(bytes(meta_arr.tolist()).decode('utf-8'))
    else:
        raise RuntimeError(
            f'load_trace: unsupported {_METADATA_KEY!r} dtype '
            f'{meta_arr.dtype!r}.')
    # Walk every other key, restoring the None sentinel.
    trace = {}
    for key in npz.files:
        if key == _METADATA_KEY:
            continue
        value = npz[key]
        # Sentinel comparison: a 0-d string array equal to the
        # sentinel means the original value was None.
        if value.shape == () and value.dtype.kind in ('U', 'S'):
            text = str(value)
            if text == _NONE_SENTINEL:
                trace[key] = None
                continue
            # Fall-through: any other 0-d string is preserved
            # as a Python str (not currently used by the driver
            # but keeps the loader robust).
            trace[key] = text
            continue
        trace[key] = value
    # C++ save_trace omits snapshot keys when no snapshot was recorded
    # (the Python writer stores a sentinel instead). Default any missing
    # snapshot key to None so downstream `is None` checks stay valid.
    for snap_key in ('snapshot_m_top', 'snapshot_m_bot', 'snapshot_t'):
        if snap_key not in trace:
            trace[snap_key] = None
    return trace, metadata
