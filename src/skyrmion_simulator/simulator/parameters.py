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
    # Square grid; periodic BCs are enforced in lattice.neighbors via roll.
    p.nx = 256
    p.ny = 256
    # a << dw=27 nm so the domain wall is well-resolved (~14 sites).
    p.a = 2e-9  # m, lattice constant
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Material (SAF Co/Pt); values taken from the SAF skyrmion paper.
    p.Ms = 1.43e6          # A/m, saturation magnetization
    p.A_ex = 16e-12        # J/m, exchange stiffness
    # Interfacial Neel DMI from Pt interface.
    # p.D = 0.62e-3        # J/m^2, legacy paper Set A
    p.D = 0.85e-3          # J/m^2, DMI constant
    # Bare K; thin-film K_eff = K - mu0 Ms^2/2 is used elsewhere.
    # p.K_top = 1.294e6    # measured (Hk_top = 12.4 mT, Table S2)
    # Paper sims raise Hk_top to 36 mT (supp. 1.1) so the domain
    # background is not reversed by SOT at the maximal current.
    p.K_top = 1.3106e6     # J/m^3, anisotropy (top)
    p.K_bot = 1.31e6       # J/m^3, anisotropy (bottom)
    # Large alpha reflects Pt proximity-enhanced damping.
    p.alpha = 0.14         # Gilbert damping
    # Co slab thickness enters the demag shape factor f(|k|*t_Co).
    p.t_Co = 1.3e-9        # m, Co layer thickness
    # Co-to-Co spacer: Ru(0.85) + Pt(0.5) = 1.35 nm (Pham 2024).
    # p.d_Ru = 0.8e-9      # legacy: Ru thickness alone
    p.d_Ru = 1.35e-9       # m, full magnetic spacer
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Constants
    p.mu0 = 4.0 * np.pi * 1e-7  # T*m/A
    # Gyromagnetic ratio: 194.8 GHz/T = 194.8e9 rad/(s*T)
    # Used directly because every H is carried in Tesla (no mu0 in LLG).
    p.gamma = 194.8e9      # rad/(s*T)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # RKKY interlayer coupling
    # AFM coupling cancels the Magnus force and kills the skyrmion Hall.
    p.H_RKKY = 0.205  # T, RKKY field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # External field (Tesla); zero by default.
    p.H_ext = np.array([0.0, 0.0, 0.0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Spin-orbit torques
    # DL_SOT * J(t) yields Tesla directly (units chosen to absorb mu0).
    p.DL_SOT = 2.21e-14  # T*A^-1*m^2, damping-like
    p.FL_SOT = 0.53e-14  # T*A^-1*m^2, field-like
    # In-plane polarization; the orientation drives skyrmion motion along x.
    p.p_hat = np.array([0.0, 1.0, 0.0])  # polarization
    # Drive J(t) in A/m^2, the only source of current for the
    # integrators. No default: set a pulse (e.g. ConstantPulse(0.0))
    # before integrating in time; the integrators refuse None.
    p.pulse = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Topological spin Hall torque
    # dm/dt += -b_j * lambda_sq * N_xy * (dm/dy), with
    # b_j = (mu_B / q_e) * J(t) * P / Ms. The mechanism is
    # disabled when lambda_sq == 0, so the existing dynamics are
    # recovered with the default parameters.
    # Length-squared coupling for the topological spin Hall effect (m^2).
    # Paper sweeps {3, 50} nm^2 for the magnitude comparison in S49.
    p.lambda_sq = 0.0
    # Spin polarization of the conduction electrons; only used when
    # lambda_sq != 0.
    p.P = 0.5
    # Precomputed ratio mu_B / q_e (J / (T C)) cached on p so the
    # llgs_rhs hot path multiplies by a single constant rather than
    # juggling two very different orders of magnitude per call.
    p.mu_B_over_q_e = 9.2740100783e-24 / 1.602176634e-19
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Demag periodic images
    # Image wraps summed per periodic direction when building the
    # Newell/racetrack kernels (0 = minimum-image truncation).
    # Residual truncation scales as ((pbc_images + 1) * L)^-3;
    # 2 puts it at ~1e-5 on the production box.
    p.pbc_images = 2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Time integration
    # Explicit RK4; dt must satisfy gamma * H_K * dt << 1 for stability.
    p.dt = 5e-14       # s
    p.n_relax = 10000  # relaxation steps (J=0), 500 ps
    p.n_steps = 5000   # current-driven steps
    p.dump_every = 100
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Skyrmion initial condition
    # Domain-wall profile centered in box; mz=0 contour radius R.
    # R derived from a measured post-relax diameter at D=0.85e-3:
    p.skyrmion_R = 93.25e-9   # m, measured for D=0.85e-3
    # dw = sqrt(A_ex / K_eff) depends only on A_ex and K_eff
    # (independent of D); paper value retained.
    p.skyrmion_dw = 27e-9     # m, domain wall width (paper)
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
    p.C_ex = 2.0 * p.A_ex / (p.Ms * a2)  # discrete Laplacian coefficient
    # DMI prefactor: D / (Ms * a)  [Tesla]
    p.C_dmi = p.D / (p.Ms * p.a)  # interfacial Neel form
    # Anisotropy prefactors: 2*K_eff / Ms  [Tesla]
    # K_eff = K - mu0*Ms^2/2 (thin-film demagnetization)
    mu0_Ms = p.mu0 * p.Ms  # shape-anisotropy correction term
    # K_eff convention here folds uniform demag into K; full demag
    # via demag.py uses bare K from fields.bare_anis_prefactors instead
    p.C_anis_top = 2.0 * p.K_top / p.Ms - mu0_Ms
    p.C_anis_bot = 2.0 * p.K_bot / p.Ms - mu0_Ms
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Gamma prime for explicit LLGS (absorbs 1/(1+a^2) prefactor)
    p.gamma_p = p.gamma / (1.0 + p.alpha ** 2)
