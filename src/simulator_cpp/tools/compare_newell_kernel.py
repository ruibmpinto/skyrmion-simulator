"""Cross-check the C++ Newell kernel against the Python reference.

Builds the same kernel via src.simulator.demag_newell at matching nx,
ny, accuracy and compares the half-spectrum FFTW r2c slice element-wise.

Functions
---------
main
    Entry point; configuration variables sit at the top.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pathlib
import sys
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


def main():
    """Compare C++ vs Python Newell kernel arrays."""
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run configuration
    cpp_npz = ('src/simulator_cpp/build/newell_kernel.npz')
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    accuracy = 8.0
    tol_conv = 2.0e-2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load C++ kernel
    if not os.path.exists(cpp_npz):
        raise RuntimeError(
            f'compare_newell_kernel: {cpp_npz} does not exist.')
    z = np.load(cpp_npz)
    ny = int(z['ny'][0])
    nx = int(z['nx'][0])
    a = float(z['a'][0])
    t_Co = float(z['t_Co'][0])
    d_Ru = float(z['d_Ru'][0])
    mu0_Ms_cpp = float(z['mu0_Ms'][0])
    print(f'C++ kernel: ny={ny} nx={nx} a={a:.3e} t_Co={t_Co:.3e} '
          f'd_Ru={d_Ru:.3e}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build Python reference
    sys.path.insert(0, str(repo_root))
    from types import SimpleNamespace
    from src.simulator.demag_newell import precompute_demag_kernels_newell
    p = SimpleNamespace()
    p.nx = nx
    p.ny = ny
    p.a = a
    p.t_Co = t_Co
    p.d_Ru = d_Ru
    p.Ms = mu0_Ms_cpp / (4.0 * np.pi * 1e-7)
    p.mu0 = 4.0 * np.pi * 1e-7
    print(f'Building Python kernel (accuracy={accuracy}, '
          f'tol_conv={tol_conv}) ...')
    K_py = precompute_demag_kernels_newell(p,
                                           accuracy=accuracy,
                                           tol_conv=tol_conv)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Slice and rescale to compare against C++ half-spectrum + norm.
    nh = nx // 2 + 1
    norm = 1.0 / (ny * nx)
    keys = ['Nxx_self', 'Nyy_self', 'Nzz_self', 'Nxy_self',
            'Nxx_inter', 'Nyy_inter', 'Nzz_inter', 'Nxy_inter',
            'Nxz_inter', 'Nyz_inter']
    max_rel = 0.0
    for k in keys:
        py_half = K_py[k][:, :nh] * norm
        cpp = z[k]
        diff = np.abs(cpp - py_half)
        denom = np.max(np.abs(py_half))
        if denom < 1e-30:
            rel = 0.0
        else:
            rel = float(diff.max() / denom)
        print(f'  {k:<10s}  max|diff|={diff.max():.3e}  '
              f'max|py|={denom:.3e}  rel={rel:.3e}')
        if rel > max_rel:
            max_rel = rel
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Verdict
    print(f'Overall max relative error: {max_rel:.3e}')
    if max_rel > 1.0e-10:
        print('compare_newell_kernel: KERNELS DIFFER ABOVE FLOAT NOISE.')
    else:
        print('compare_newell_kernel: kernels match within float noise.')


# =============================================================================
if __name__ == '__main__':
    main()
