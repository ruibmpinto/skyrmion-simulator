"""Simulation parameters for SAF skyrmion dynamics.

Provides default physical constants and simulation settings
for a synthetic antiferromagnet Co/Pt multilayer system.
All effective fields are computed in Tesla.

Functions
---------
default_params
    Return default simulation parameters.
"""
#
#                                                                Modules
# =====================================================================
# Standard
from types import SimpleNamespace
# Third-party
import numpy as np

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def default_params():
    """Return default simulation parameters.

    All values correspond to a SAF Co/Pt multilayer system
    from experimental characterization.

    Returns
    -------
    p : SimpleNamespace
        Namespace with all simulation parameters.

    Notes
    -----
    Physical parameters from:
    Fast current-induced skyrmion motion in synthetic
    antiferromagnets.

    All effective fields are in Tesla. The LLG equation
    uses gamma in rad/(s*T) directly:
        dm/dt = -gamma * m x H_eff (Tesla)
    """
    p = SimpleNamespace()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Lattice
    p.nx = 256
    p.ny = 256
    p.a = 2e-9  # m, lattice constant
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Material (SAF Co/Pt)
    p.Ms = 1.43e6          # A/m, saturation magnetization
    p.A_ex = 16e-12        # J/m, exchange stiffness
    p.D = 0.62e-3          # J/m^2, DMI constant
    p.K_top = 1.294e6      # J/m^3, anisotropy (top)
    p.K_bot = 1.31e6       # J/m^3, anisotropy (bottom)
    p.alpha = 0.14         # Gilbert damping
    p.t_Co = 1.3e-9        # m, Co layer thickness
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Constants
    p.mu0 = 4.0 * np.pi * 1e-7  # T*m/A
    # Gyromagnetic ratio: 194.8 GHz/T = 194.8e9 rad/(s*T)
    p.gamma = 194.8e9      # rad/(s*T)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RKKY interlayer coupling
    p.H_RKKY = 0.205  # T, RKKY field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # External field (Tesla)
    p.H_ext = np.array([0.0, 0.0, 0.0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Spin-orbit torques
    p.DL_SOT = 2.21e-14  # T*A^-1*m^2, damping-like
    p.FL_SOT = 0.53e-14  # T*A^-1*m^2, field-like
    p.J_current = 4.0e11    # A/m^2, current density
    p.p_hat = np.array([0.0, 1.0, 0.0])  # polarization
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Time integration
    p.dt = 5e-14       # s
    p.n_relax = 10000  # relaxation steps (J=0), 500 ps
    p.n_steps = 5000   # current-driven steps
    p.dump_every = 100
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Skyrmion initial condition
    p.skyrmion_R = 80e-9   # m, skyrmion radius (mz=0)
    p.skyrmion_dw = 27e-9  # m, domain wall width (paper)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output
    p.output_dir = 'output'
    p.output_file = 'skyrmion.dump'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Precomputed prefactors
    _precompute(p)
    return p


def _precompute(p):
    """Precompute field prefactors in Tesla units.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace (modified in place).

    Notes
    -----
    All prefactors yield fields in Tesla when applied
    to the unit magnetization m. This avoids mu0
    in the LLG equation: dm/dt = -gamma * m x H(Tesla).

    Exchange: H = (2*A_ex / (Ms * a^2)) * Laplacian(m)
    DMI:      H = (D / (Ms * a)) * ...
    Anis:     H = (2*K / Ms) * mz * z_hat
    """
    a2 = p.a * p.a
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Exchange prefactor: 2*A / (Ms * a^2)  [Tesla]
    p.C_ex = 2.0 * p.A_ex / (p.Ms * a2)
    # DMI prefactor: D / (Ms * a)  [Tesla]
    p.C_dmi = p.D / (p.Ms * p.a)
    # Anisotropy prefactors: 2*K_eff / Ms  [Tesla]
    # K_eff = K - mu0*Ms^2/2 (thin-film demagnetization)
    mu0_Ms = p.mu0 * p.Ms
    p.C_anis_top = 2.0 * p.K_top / p.Ms - mu0_Ms
    p.C_anis_bot = 2.0 * p.K_bot / p.Ms - mu0_Ms
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SOT: DL_SOT * J gives field in Tesla directly
    if p.J_current != 0.0:
        p.H_DL = p.DL_SOT * p.J_current  # Tesla
        p.H_FL = p.FL_SOT * p.J_current  # Tesla
    else:
        p.H_DL = 0.0
        p.H_FL = 0.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Gamma prime for explicit LLGS
    p.gamma_p = p.gamma / (1.0 + p.alpha ** 2)
