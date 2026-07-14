"""Single-trajectory orchestrator for sweep scripts.

`run_one(...)` advances one full trajectory: a relaxation
phase with `J = 0` followed by a drive phase with a
user-supplied pulse. During the drive phase observables are
sampled at fixed intervals and optionally one full-spin
snapshot is captured. The integrator step is supplied by the
caller, so deterministic, demag-aware, or stochastic runs all
flow through the same orchestrator.

Functions
---------
run_one
    Run one trajectory and return a trace dict ready for
    `sweeps.io.save_trace`.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
# Local
from src.simulator.pulses import ConstantPulse
from src.orchestrator.observers import observe_state

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def run_one(p,
            pulse,
            n_relax,
            n_drive,
            sample_every,
            step_drive,
            step_relax,
            ic_factory,
            record_snapshot_at,
            print_every):
    """Run one full trajectory.

    Phase 1 swaps `p.pulse` for `ConstantPulse(0.0)` and zeroes
    `p.H_DL` / `p.H_FL` so the relaxation runs with no current
    regardless of which integrator is used. Phase 2 installs
    the supplied pulse and samples observables every
    `sample_every` steps.

    Parameters
    ----------
    p : SimpleNamespace
        Simulation parameters; mutated in place during the
        phase switches but restored on exit so the caller can
        reuse the namespace.
    pulse : callable
        Current-density profile for the drive phase. The
        integrator reads `pulse(t)` at each RK4 / Heun substage.
    n_relax : int
        Number of RK4 / Heun steps in the relaxation phase.
        Must be `>= 0`.
    n_drive : int
        Number of steps in the drive phase. Must be `> 0`.
    sample_every : int
        Observables are recorded every `sample_every` steps
        during the drive phase. Must be `> 0`.
    step_drive : callable
        Integrator step used during the drive phase. Signature
        `(m_top, m_bot, t, dt, p) -> (m_top_new, m_bot_new)`.
    step_relax : callable
        Integrator step used during the relaxation phase. Pass
        the same callable as `step_drive` if no separation is
        wanted. Required, no default.
    ic_factory : callable
        Builds the initial condition. Signature `(p) ->
        (m_top, m_bot)`. Required, no default.
    record_snapshot_at : {None, float}
        If a float, record a copy of `(m_top, m_bot)` at the
        first sampled time at or after this value (seconds
        relative to the drive-phase origin). Pass `None` to
        skip the snapshot.
    print_every : int
        Stride of progress prints during the relax and drive
        phases. Pass 0 to disable. Required, no default.

    Returns
    -------
    trace : dict
        Keys:
            t : numpy.ndarray(1d)
                Sample times in seconds (drive-phase origin).
            <observable> : numpy.ndarray(1d)
                One array per key returned by `observe_state`,
                same length as `t`.
            snapshot_m_top, snapshot_m_bot : numpy.ndarray(3d)
                Spin configurations at the recorded snapshot
                time, or `None` if no snapshot was requested.
            snapshot_t : float
                Time at which the snapshot was taken (s), or
                `None`.

    Notes
    -----
    Raises `RuntimeError` on invalid arguments (negative step
    counts, zero stride, `record_snapshot_at` outside the
    drive window). No silent fallbacks; every requirement is
    enforced up front.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Validate inputs loudly.
    if not isinstance(n_relax, int) or n_relax < 0:
        raise RuntimeError(
            f'run_one: `n_relax` must be a non-negative int, '
            f'got {n_relax!r}.')
    if not isinstance(n_drive, int) or n_drive <= 0:
        raise RuntimeError(
            f'run_one: `n_drive` must be a positive int, '
            f'got {n_drive!r}.')
    if not isinstance(sample_every, int) or sample_every <= 0:
        raise RuntimeError(
            f'run_one: `sample_every` must be a positive int, '
            f'got {sample_every!r}.')
    if step_drive is None or step_relax is None:
        raise RuntimeError(
            'run_one: both `step_drive` and `step_relax` are '
            'required (no defaults).')
    if ic_factory is None:
        raise RuntimeError('run_one: `ic_factory` is required.')
    if not isinstance(print_every, int) or print_every < 0:
        raise RuntimeError(
            f'run_one: `print_every` must be a non-negative '
            f'int, got {print_every!r}.')
    # Drive window in seconds; snapshot time must fall inside it.
    # The final state at n_drive * dt is sampled unconditionally,
    # so the full drive window is reachable by the snapshot.
    drive_window = n_drive * p.dt
    if record_snapshot_at is not None:
        if not np.isfinite(record_snapshot_at):
            raise RuntimeError(
                f'run_one: `record_snapshot_at` must be finite, '
                f'got {record_snapshot_at!r}.')
        if record_snapshot_at < 0.0 or record_snapshot_at > drive_window:
            raise RuntimeError(
                f'run_one: `record_snapshot_at` = '
                f'{record_snapshot_at:.3e} s falls outside the '
                f'drive window [0, {drive_window:.3e}] s.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build the initial condition.
    m_top, m_bot = ic_factory(p)
    # Shape sanity check; the caller's ic_factory should match
    # (p.ny, p.nx, 3) but verify so a malformed IC fails loudly.
    expected_shape = (p.ny, p.nx, 3)
    if m_top.shape != expected_shape:
        raise RuntimeError(
            f'run_one: ic_factory returned m_top.shape '
            f'{m_top.shape} != expected {expected_shape}.')
    if m_bot.shape != expected_shape:
        raise RuntimeError(
            f'run_one: ic_factory returned m_bot.shape '
            f'{m_bot.shape} != expected {expected_shape}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Phase 1: relaxation with J = 0.
    pulse_save = p.pulse
    H_DL_save = p.H_DL
    H_FL_save = p.H_FL
    p.pulse = ConstantPulse(0.0)
    p.H_DL = 0.0
    p.H_FL = 0.0
    t = 0.0
    for step in range(1, n_relax + 1):
        m_top, m_bot = step_relax(m_top, m_bot, t, p.dt, p)
        t += p.dt
        if print_every > 0 and step % print_every == 0:
            print(
                f'    relax step {step}/{n_relax} '
                f'({step*p.dt*1e12:.1f} ps)',
                flush=True,)
    # Restore everything before installing the drive pulse so a
    # later exception cannot leave `p` in a half-mutated state.
    p.pulse = pulse_save
    p.H_DL = H_DL_save
    p.H_FL = H_FL_save
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Phase 2: drive with the user-supplied pulse.
    p.pulse = pulse
    # Reset the integration clock so the pulse's t = 0 aligns
    # with the start of the drive phase (matches the convention
    # documented in simulator.main.run).
    t = 0.0
    # Buffers for the trace.
    sample_times = []
    sample_observations = []
    snapshot = None
    snapshot_t = None
    try:
        for step in range(n_drive):
            # Sampling check: record observables at every
            # sample_every steps including step == 0.
            if step % sample_every == 0:
                sample_times.append(t)
                sample_observations.append(observe_state(m_top, m_bot, p))
                # Snapshot once, at the first sample reaching the
                # requested time.
                if (record_snapshot_at is not None
                        and snapshot is None
                        and t >= record_snapshot_at):
                    snapshot = (m_top.copy(), m_bot.copy())
                    snapshot_t = t
            # Advance one step.
            m_top, m_bot = step_drive(m_top, m_bot, t, p.dt, p)
            t += p.dt
            if (print_every > 0
                    and (step + 1) % print_every == 0):
                print(
                    f'    drive step {step+1}/{n_drive} '
                    f'({(step+1)*p.dt*1e12:.1f} ps)',
                    flush=True,)
        # Final state at t = n_drive * dt: sampled unconditionally
        # so the trace endpoint is the state the drive ends on.
        sample_times.append(t)
        sample_observations.append(observe_state(m_top, m_bot, p))
        if (record_snapshot_at is not None
                and snapshot is None
                and t >= record_snapshot_at):
            snapshot = (m_top.copy(), m_bot.copy())
            snapshot_t = t
    finally:
        # Always restore p.pulse on exit so the caller's
        # namespace is left in a clean state, even if the
        # integrator raised.
        p.pulse = pulse_save
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pack the trace into a dict of 1D arrays plus the optional
    # snapshot. Numpy stack keeps the per-key column order
    # consistent across rows.
    trace = {'t': np.asarray(sample_times, dtype=float)}
    # Derive the observation key set from the first sample.
    if sample_observations:
        keys = list(sample_observations[0].keys())
        for key in keys:
            trace[key] = np.asarray(
                [obs[key] for obs in sample_observations],
                dtype=float)
    # Snapshot fields are stored even when unused so the trace
    # schema is fixed; `save_trace` handles None encoding.
    if snapshot is not None:
        trace['snapshot_m_top'] = snapshot[0]
        trace['snapshot_m_bot'] = snapshot[1]
        trace['snapshot_t'] = float(snapshot_t)
    else:
        trace['snapshot_m_top'] = None
        trace['snapshot_m_bot'] = None
        trace['snapshot_t'] = None
    return trace
