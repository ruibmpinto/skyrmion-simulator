"""FMR (uniform-mode) frequency benchmark.

Initialises a uniformly magnetised single-FM layer with a small
in-plane tilt and integrates the (lightly damped) LLG equation
with `rk4_step` + `rhs_local_keff`. The k = 0 spin precession
should occur at the analytic FMR frequency

    f_FMR = (gamma / 2 pi) * H_K,    H_K = 2 K_eff / M_s,

(no external Zeeman, no DMI, no demag; PMA easy axis along +z).
The fitted frequency is compared to the analytic value.

Functions
---------
main
    Build IC, integrate, FFT, fit f_FMR, report.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
# Third-party
import numpy as np
# Local
from skyrmion_simulator.simulator.initial_conditions import uniform_state
from skyrmion_simulator.simulator.pulses import ConstantPulse
from skyrmion_simulator.simulator.validation._helpers import (
    integrate_single_fm,
    make_single_fm_params,
    plot_ic_2d,
)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def main():
    """Run the uniform-mode FMR frequency validation.

    Drives a small uniform PMA film (no DMI, no field) from a
    slightly tilted IC, integrates the free precession, extracts the
    dominant k = 0 mode frequency, and writes the IC figure. Passes
    when the measured frequency matches the analytic
    f = gamma H_K / 2 pi (H_K = 2 K_eff / Ms) within tolerance.

    Returns
    -------
    passed : bool
        True if the measured FMR frequency is within `rtol_freq`.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Material parameters. K_eff and Ms set H_K and hence f_FMR.
    A_ex = 16.0e-12
    D = 0.0                  # no DMI; pure FMR mode
    K_eff = 5.0e5            # J/m^3
    Ms = 8.0e5               # A/m  (permalloy-like)
    alpha = 0.005            # light damping so f shows up clearly
    gamma = 1.760e11         # rad / (s T)
    H_ext = np.array([0.0, 0.0, 0.0])
    # Small lattice -- k=0 mode is dominant; 32x32 is plenty.
    nx, ny = 32, 32
    a = 2.0e-9
    dt = 1.0e-13
    # Drive duration. f_FMR is ~30 GHz here so 5 ns covers many
    # periods.
    n_steps = 50_000
    sample_every = 10        # 1 ps cadence
    # IC tilt amplitude.
    tilt = 0.05
    # Acceptance.
    rtol_freq = 5.0e-2
    ic_png = 'docs/figures/validation/fmr_initial.png'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Analytic FMR frequency for a uniaxial-K thin film (PMA),
    # no external field. H_K = 2 K_eff / M_s (Tesla).
    H_K = 2.0 * K_eff / Ms
    f_an = gamma * H_K / (2.0 * math.pi)
    print(
        f'test_fmr_dispersion: K_eff={K_eff:.2e} J/m^3, '
        f'Ms={Ms:.2e} A/m, H_K = 2K/Ms = {H_K:.3f} T')
    print(
        f'  analytic f_FMR = gamma H_K / 2 pi = '
        f'{f_an*1e-9:.3f} GHz')
    print(
        f'  lattice {nx}x{ny} at a={a*1e9:.1f} nm; integrating '
        f'{n_steps*dt*1e9:.1f} ns at dt={dt*1e15:.0f} fs.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Uniform IC with small in-plane tilt.
    m0 = uniform_state(nx=nx, ny=ny, direction=np.array(
        [tilt, 0.0, math.sqrt(1.0 - tilt * tilt)]))
    plot_ic_2d(
        m=m0, a=a, out_path=ic_png,
        title=f'FMR IC: uniform + {tilt*100:.0f}% tilt')
    print(f'  IC figure: {ic_png}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    p = make_single_fm_params(
        A_ex=A_ex, D=D, K_eff=K_eff, Ms=Ms,
        alpha=alpha, gamma=gamma, H_ext=H_ext,
        nx=nx, ny=ny, a=a, dt=dt, pulse=ConstantPulse(0.0))
    times, m_trace = integrate_single_fm(
        m_top=m0, p=p, n_steps=n_steps,
        sample_every=sample_every)
    # Spatially averaged m_x(t); the k=0 mode dominates.
    mx_t = np.mean(m_trace[..., 0], axis=(1, 2))
    # Remove mean before FFT to avoid a huge DC bin.
    mx_t = mx_t - np.mean(mx_t)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # FFT and peak detection.
    dt_sample = float(times[1] - times[0])
    spectrum = np.abs(np.fft.rfft(mx_t))
    freqs = np.fft.rfftfreq(mx_t.size, d=dt_sample)
    # Exclude the zero-frequency bin from the peak search.
    peak_idx = int(np.argmax(spectrum[1:]) + 1)
    f_fit = float(freqs[peak_idx])
    rel_err = abs(f_fit - f_an) / f_an
    print(
        f'  FFT peak f = {f_fit*1e-9:.3f} GHz '
        f'(analytic {f_an*1e-9:.3f} GHz, rel.err = '
        f'{rel_err*100:.2f} %)')
    passed = rel_err < rtol_freq
    print(
        f'  status: {"PASS" if passed else "FAIL"} '
        f'(rtol = {rtol_freq*100:.1f} %)')
    return passed


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
