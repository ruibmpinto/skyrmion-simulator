"""Spin-wave equipartition validation gate.

A small (16 x 16) ferromagnetic patch, single layer (the SAF
bottom is decoupled by `H_RKKY = 0`), no DMI, no anisotropy,
weak Zeeman field along z to fix the ground state. Equilibrate
at low temperature, then sample the transverse magnetization
m_perp = (m_x, m_y) and verify that each Fourier mode of
m_perp follows the Rayleigh-Jeans / equipartition prediction

    <|M_x(k)|**2 + |M_y(k)|**2>
        = 2 * N * k_B * T / (M_s * V_cell * H_k),

with the unitless FFT defined as
`M(k) = sum_r exp(-i k . r) * m_perp(r)`, and the magnon
stiffness

    H_k = B_z + C_ex * (4 - 2 cos(k_x a) - 2 cos(k_y a)).

Pass criterion: the median ratio sim / theory over low-k modes
(|k a| < pi / 2) lies in [1 - tol_rel, 1 + tol_rel]. Failure
here indicates the per-cell noise amplitude is wrong or the
exchange Laplacian is wrong; the macrospin Langevin gate
cannot catch either since N = 1 there.

Run with:
    python -m skyrmion_simulator.stochastic_llgs.validation.test_equipartition
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
import os
import time
# Third-party
import numpy as np
# Local
from skyrmion_simulator.simulator.parameters import default_params
from skyrmion_simulator.simulator.pulses import ConstantPulse
from skyrmion_simulator.stochastic_llgs.integrator_sllg import \
    heun_stochastic_step
from skyrmion_simulator.stochastic_llgs.parameters_thermal import attach_thermal
from skyrmion_simulator.stochastic_llgs.thermal_field import \
    sample_thermal_field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def make_equipartition_params(T, alpha, B_z, nx, ny, a, t_Co,
                              Ms, A_ex, gamma, seed):
    """Construct a parameters namespace for the equipartition
    test.

    Parameters
    ----------
    T : float
        Temperature in Kelvin. Strictly positive.
    alpha : float
        Gilbert damping. Strictly positive.
    B_z : float
        Zeeman field along z in Tesla. Strictly positive.
    nx, ny : int
        Lattice size. Both strictly positive.
    a : float
        Lattice constant in meters. Strictly positive.
    t_Co : float
        Layer thickness in meters. Strictly positive.
    Ms : float
        Saturation magnetization in A/m. Strictly positive.
    A_ex : float
        Exchange stiffness in J/m. Strictly positive.
    gamma : float
        Gyromagnetic ratio in rad/(s*T). Strictly positive.
    seed : int
        Integer master seed.

    Returns
    -------
    p : SimpleNamespace
        Parameters with `K_top = K_bot = 0`, `H_RKKY = 0`,
        `pulse = ConstantPulse(0.0)`, `C_dmi = 0`, bare anisotropy
        prefactor `C_anis_top = 0`, and `sigma_noise`
        populated by `attach_thermal`.
    """
    for name, val in (('alpha', alpha), ('B_z', B_z),
                      ('a', a), ('t_Co', t_Co), ('Ms', Ms),
                      ('A_ex', A_ex), ('gamma', gamma)):
        if not (np.isfinite(val) and val > 0.0):
            raise RuntimeError(
                f'make_equipartition_params: {name} must be '
                f'finite and strictly positive, got {val!r}.'
            )
    for name, val in (('nx', nx), ('ny', ny)):
        if not isinstance(val, (int, np.integer)) or val <= 0:
            raise RuntimeError(
                f'make_equipartition_params: {name} must be '
                f'a positive int, got {val!r}.'
            )
    p = copy.deepcopy(default_params())
    p.nx = int(nx)
    p.ny = int(ny)
    p.alpha = float(alpha)
    p.gamma = float(gamma)
    p.Ms = float(Ms)
    p.a = float(a)
    p.t_Co = float(t_Co)
    p.A_ex = float(A_ex)
    p.H_ext = np.array([0.0, 0.0, float(B_z)])
    p.K_top = 0.0
    p.K_bot = 0.0
    p.H_RKKY = 0.0
    p.pulse = ConstantPulse(0.0)
    p.D = 0.0
    p.C_ex = 2.0 * p.A_ex / (p.Ms * p.a * p.a)
    p.C_dmi = 0.0
    p.C_anis_top = 0.0
    p.C_anis_bot = 0.0
    p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
    attach_thermal(p, T=T, R_th=0.0, seed=seed)
    return p


# -----------------------------------------------------------------------------
def magnon_stiffness_grid(p, ny, nx):
    """Discrete magnon stiffness H_k in Tesla.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace.
    ny, nx : int
        Lattice size.

    Returns
    -------
    H_k : numpy.ndarray(2d)
        Stiffness in Tesla on the np.fft.fft2 frequency grid,
        shape (ny, nx). At k=0, H_k = B_z (no exchange
        contribution since 1 - cos(0) = 0).
    """
    B_z = float(p.H_ext[2])
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=p.a)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=p.a)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    exchange = p.C_ex * (
        4.0 - 2.0 * np.cos(KX * p.a)
        - 2.0 * np.cos(KY * p.a)
    )
    return B_z + exchange


# -----------------------------------------------------------------------------
def main():
    """Run the spin-wave equipartition gate.

    Relaxes, accumulates per-mode transverse variance, compares
    to Rayleigh-Jeans theory, and gates on the low-k median
    sim/theory ratio.
    """
    # =========================== User Configuration =========================
    nx              = 16
    ny              = 16
    t_kelvin        = 5.0
    b_z             = 0.5            # Tesla; large enough that
    # the linearization is
    # valid
    alpha           = 0.1
    a               = 2.0e-9
    t_co            = 1.3e-9
    ms              = 1.43e6
    a_ex            = 16.0e-12
    gamma           = 194.8e9
    dt              = 5.0e-14
    n_relax_steps   = 20000          # 1 ns relaxation
    n_steps         = 200000         # 10 ns sampling
    sample_every    = 200            # 10 ps between samples
    seed            = 41
    tol_norm        = 5.0e-3
    tol_rel         = 0.20           # 20% median-ratio gate
    k_low_cut       = 0.5            # use modes with |k|/k_max < cut
    out_dir         = (
        'output/stochastic_llgs/validation'
    )
    out_npz         = 'equipartition.npz'
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    print('Spin-wave equipartition gate')
    print('-' * 56)
    print(
        f'nx={nx}, ny={ny}, T={t_kelvin} K, B_z={b_z} T, '
        f'alpha={alpha}, dt={dt:.2e}'
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    p = make_equipartition_params(
        T=t_kelvin, alpha=alpha, B_z=b_z, nx=nx, ny=ny,
        a=a, t_Co=t_co, Ms=ms, A_ex=a_ex, gamma=gamma,
        seed=seed,
    )
    H_k = magnon_stiffness_grid(p, ny, nx)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial condition: aligned with B_z
    m_top = np.zeros((ny, nx, 3), dtype=float)
    m_top[..., 2] = 1.0
    m_bot = m_top.copy()
    rng = np.random.default_rng(int(p.seed))
    sigma = float(p.sigma_noise)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Relaxation
    print('Relaxation phase...')
    t_start = time.time()
    for step in range(n_relax_steps):
        h_top = sample_thermal_field(
            rng, (ny, nx), sigma, dt,
        )
        h_bot = sample_thermal_field(
            rng, (ny, nx), sigma, dt,
        )
        m_top, m_bot, _ = heun_stochastic_step(
            m_top, m_bot, dt, p, None,
            h_top, h_bot, tol_norm, t=step * dt,
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sampling
    print('Sampling phase...')
    accum = np.zeros((ny, nx), dtype=float)
    n_samples = 0
    for step in range(n_steps):
        h_top = sample_thermal_field(
            rng, (ny, nx), sigma, dt,
        )
        h_bot = sample_thermal_field(
            rng, (ny, nx), sigma, dt,
        )
        m_top, m_bot, _ = heun_stochastic_step(
            m_top, m_bot, dt, p, None,
            h_top, h_bot, tol_norm, t=step * dt,
        )
        if step % sample_every == 0:
            # Per-mode transverse power |M_x(k)|^2 + |M_y(k)|^2.
            M_x = np.fft.fft2(m_top[..., 0])
            M_y = np.fft.fft2(m_top[..., 1])
            accum += np.abs(M_x) ** 2 + np.abs(M_y) ** 2
            n_samples += 1
    sim_mode_var = accum / n_samples
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Theory: <|M_x|^2 + |M_y|^2> = 2 N kT / (Ms V_cell H_k)
    N = nx * ny
    th_mode_var = (
        2.0 * N * p.k_B * p.T / (p.Ms * p.V_cell * H_k)
    )
    # Low-k mask (exclude k=0 mode: ratio is well-defined
    # there, but it is also the most fluctuating; keep it).
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    # normalize to 1.0
    k_mag = np.sqrt(KX ** 2 + KY ** 2) / 0.5
    low_k = k_mag < k_low_cut
    ratio = sim_mode_var / th_mode_var
    median_ratio = float(np.median(ratio[low_k]))
    max_dev = float(np.max(np.abs(ratio[low_k] - 1.0)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    out_path = os.path.join(out_dir, out_npz)
    np.savez_compressed(
        out_path,
        nx=nx, ny=ny, t_kelvin=t_kelvin, b_z=b_z,
        alpha=alpha, dt=dt, n_samples=n_samples,
        H_k=H_k, sim_mode_var=sim_mode_var,
        th_mode_var=th_mode_var, ratio=ratio,
        low_k_mask=low_k, median_ratio=median_ratio,
        max_dev=max_dev, tol_rel=tol_rel,
    )
    dt_wall = time.time() - t_start
    print('-' * 56)
    print(
        f'median(sim/theory) over low-k modes '
        f'(|k|/k_max < {k_low_cut}) = {median_ratio:.3f}, '
        f'max|ratio - 1| = {max_dev:.3f}; '
        f'gate tol = {tol_rel:.2f}; wall {dt_wall:.1f} s'
    )
    print(f'Saved {out_path}')
    if abs(median_ratio - 1.0) >= tol_rel:
        raise RuntimeError(
            f'Equipartition gate FAILED: median ratio '
            f'{median_ratio:.3f} deviates from 1 by more '
            f'than tol_rel {tol_rel:.2f}.'
        )
    print('Equipartition gate PASSED.')


# =============================================================================
if __name__ == '__main__':
    main()
