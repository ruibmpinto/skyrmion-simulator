"""Parameter-namespace builder for the phase-diagram sweep.

Wraps `skyrmion_simulator.simulator.parameters.default_params` so the
caller can override individual material constants and have
all dependent prefactors recomputed consistently. Mirrors
the formulas used by `parameters._precompute` without
importing it.

Functions
---------
make_params
    Return a SimpleNamespace built from `default_params()`
    with `**overrides` applied and all dependent
    prefactors re-derived.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import copy
# Third-party
import numpy as np
# Local
from skyrmion_simulator.simulator.parameters import default_params

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def make_params(**overrides):
    """Build a parameters namespace with overrides applied.

    Parameters
    ----------
    **overrides
        Keyword overrides of attributes on the default
        parameters namespace. Both raw material constants
        (e.g. `D`, `K_top`, `Ms`, `a`, `alpha`,
        `J_current`, `H_ext`) and structural constants
        (e.g. `nx`, `ny`, `t_Co`, `d_Ru`) are accepted.
        Unknown keys raise `RuntimeError` so typos are
        caught immediately instead of silently doing
        nothing.

    Returns
    -------
    p : SimpleNamespace
        New parameters namespace; the caller is free to
        mutate it without affecting subsequent calls.

    Notes
    -----
    Dependent prefactors are always recomputed:

        C_ex        = 2 * A_ex / (Ms * a^2)
        C_dmi       = D / (Ms * a)
        C_anis_top  = 2 * K_top / Ms - mu0 * Ms
        C_anis_bot  = 2 * K_bot / Ms - mu0 * Ms
        gamma_p     = gamma / (1 + alpha^2)
        H_DL, H_FL  = (DL_SOT, FL_SOT) * J_current

    The `C_anis_*` prefactors retain the K_eff convention
    (so existing code paths continue to work). New
    phase-diagram code that uses explicit demag should
    obtain bare K via
    `skyrmion_simulator.simulator.fields.bare_anis_prefactors`.
    """
    p = copy.deepcopy(default_params())
    known_keys = set(vars(p).keys())
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reject unknown override keys to catch typos.
    unknown = set(overrides) - known_keys
    if unknown:
        raise RuntimeError(
            f'Unknown make_params override(s): {sorted(unknown)}. '
            f'Valid keys: {sorted(known_keys)}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Apply user overrides.
    for key, value in overrides.items():
        if key == 'H_ext':
            value = np.asarray(value, dtype=float)
            if value.shape != (3,):
                raise RuntimeError(
                    f'H_ext override must have shape (3,), '
                    f'got {value.shape}.'
                )
        setattr(p, key, value)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Re-derive prefactors (mirrors parameters._precompute)
    a2 = p.a * p.a
    p.C_ex = 2.0 * p.A_ex / (p.Ms * a2)
    p.C_dmi = p.D / (p.Ms * p.a)
    mu0_Ms = p.mu0 * p.Ms
    p.C_anis_top = 2.0 * p.K_top / p.Ms - mu0_Ms
    p.C_anis_bot = 2.0 * p.K_bot / p.Ms - mu0_Ms
    if p.J_current != 0.0:
        p.H_DL = p.DL_SOT * p.J_current
        p.H_FL = p.FL_SOT * p.J_current
    else:
        p.H_DL = 0.0
        p.H_FL = 0.0
    p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
    return p
