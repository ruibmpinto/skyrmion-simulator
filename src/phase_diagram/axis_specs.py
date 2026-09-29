"""Registry of sweep axes for the generic phase-diagram pipeline.

Each entry maps an axis name (a string used in `sweep.main()`
configuration and in NPZ output) to:

- a display label for matplotlib axes,
- a multiplicative scale that converts the SI value to the
  natural plotting unit (e.g. J/m^2 -> mJ/m^2),
- a builder that turns a single axis value into a dict of
  `make_params(**overrides)` kwargs.

Adding a new sweep axis means appending one entry here; the
sweep and plot modules consume the registry without further
changes. Unknown axis names raise `RuntimeError`; the
registry never silently falls back.

Functions
---------
overrides_for
    Build the make_params override dict for one (name, value).
display_value
    Convert an SI axis value to its plotting-natural number.
label_for
    Look up the matplotlib display label for an axis name.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================
# Each entry: {
#     'label':           matplotlib display label,
#     'display_scale':   factor to convert SI -> plot units,
#     'make_overrides':  value -> dict of make_params kwargs,
# }
axes = {
    'D': {
        'label': r'DMI $D$ (mJ/m$^2$)',
        'display_scale': 1.0e3,
        'make_overrides': lambda v: {'D': float(v)},
    },
    'H_z': {
        'label': r'External field $H_z$ (T)',
        'display_scale': 1.0,
        'make_overrides': lambda v: {
            'H_ext': np.array([0.0, 0.0, float(v)]),
        },
    },
    'K_top': {
        'label': r'Anisotropy $K$ (MJ/m$^3$)',
        'display_scale': 1.0e-6,
        'make_overrides': lambda v: {
            'K_top': float(v),
            'K_bot': float(v),
        },
    },
    'H_RKKY': {
        'label': r'RKKY field $H_{\mathrm{RKKY}}$ (T)',
        'display_scale': 1.0,
        'make_overrides': lambda v: {'H_RKKY': float(v)},
    },
    'Ms': {
        'label': r'$M_s$ (MA/m)',
        'display_scale': 1.0e-6,
        'make_overrides': lambda v: {'Ms': float(v)},
    },
    'A_ex': {
        'label': r'Exchange $A_{\mathrm{ex}}$ (pJ/m)',
        'display_scale': 1.0e12,
        'make_overrides': lambda v: {'A_ex': float(v)},
    },
    't_Co': {
        'label': r'Co thickness $t_{\mathrm{Co}}$ (nm)',
        'display_scale': 1.0e9,
        'make_overrides': lambda v: {'t_Co': float(v)},
    },
    'd_Ru': {
        'label': r'Ru spacer $d_{\mathrm{Ru}}$ (nm)',
        'display_scale': 1.0e9,
        'make_overrides': lambda v: {'d_Ru': float(v)},
    },
    'alpha': {
        'label': r'Gilbert damping $\alpha$',
        'display_scale': 1.0,
        'make_overrides': lambda v: {'alpha': float(v)},
    },
}


def _require(axis_name):
    """Return the axes entry or raise."""
    if axis_name not in axes:
        raise RuntimeError(
            f'Unknown sweep axis {axis_name!r}. '
            f'Registered axes: {sorted(axes)}.'
        )
    return axes[axis_name]


def overrides_for(axis_name, value):
    """Build the make_params override dict for one axis value.

    Parameters
    ----------
    axis_name : str
        Key in axes.
    value : float
        Numerical value of the axis in SI units.

    Returns
    -------
    overrides : dict
        Keyword arguments suitable for `make_params(**overrides)`.
    """
    return _require(axis_name)['make_overrides'](value)


def display_value(axis_name, value):
    """Convert an SI axis value to its plotting-natural number.

    Parameters
    ----------
    axis_name : str
        Key in axes.
    value : float
        Numerical value of the axis in SI units.

    Returns
    -------
    display : float
        Value multiplied by the axis's `display_scale`.
    """
    return float(value) * _require(axis_name)['display_scale']


def label_for(axis_name):
    """Return the matplotlib display label for an axis name.

    Parameters
    ----------
    axis_name : str
        Key in axes.

    Returns
    -------
    label : str
        TeX-formatted label suitable for `ax.set_xlabel`.
    """
    return _require(axis_name)['label']
