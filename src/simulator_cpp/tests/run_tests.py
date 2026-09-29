"""End-to-end test runner: (re)build references, configure, build,
run all C++ vs Python parity tests, summarize.

Functions
---------
main
    Entry point; configuration variables sit at the top.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import pathlib
import subprocess
import sys

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
    """Run the full parity-test pipeline.

    Run configuration sits below; edit and re-run.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run configuration
    rebuild_references = True   # regenerate references.npz from Python
    cmake_jobs = 0              # 0 => let cmake decide (-j)
    verbose_ctest = True
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    here = pathlib.Path(__file__).resolve().parent
    cpp_root = here.parent
    build_dir = cpp_root / 'build'
    repo = cpp_root.parents[1]   # simulator_cpp -> src -> repo root
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Stage 1: Python reference
    if rebuild_references:
        print('=' * 70)
        print('1. Generating references.npz ...')
        print('=' * 70)
        _run([sys.executable, str(here / 'make_references.py')])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Stage 2: configure
    print('=' * 70)
    print('2. Configuring CMake ...')
    print('=' * 70)
    _run(['cmake', '-S', str(cpp_root), '-B', str(build_dir)])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Stage 3: build
    print('=' * 70)
    print('3. Building tests ...')
    print('=' * 70)
    cmd = ['cmake', '--build', str(build_dir)]
    if cmake_jobs > 0:
        cmd += ['-j', str(cmake_jobs)]
    else:
        cmd += ['-j']
    _run(cmd)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Stage 4: run
    print('=' * 70)
    print('4. Running tests via ctest ...')
    print('=' * 70)
    cmd = ['ctest', '--output-on-failure']
    if verbose_ctest:
        cmd += ['--verbose']
    rc = subprocess.call(cmd, cwd=str(build_dir))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Stage 5: Python-side parity + error-contract harnesses.
    if rc == 0:
        print('=' * 70)
        print('5. Python error-contract + sweep-parity harnesses ...')
        print('=' * 70)
        rc = subprocess.call(
            [sys.executable, str(here / 'edge_contract.py')],
            cwd=str(repo))
        if rc == 0:
            sp = build_dir / 'sweep_parity'
            sp_py = cpp_root / 'tools' / 'sweep_parity.py'
            if sp.exists() and sp_py.exists():
                if subprocess.call([str(sp)]) == 0:
                    rc = subprocess.call(
                        [sys.executable, str(sp_py)], cwd=str(repo))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if rc == 0:
        print('=' * 70)
        print('ALL TESTS PASSED')
        print('=' * 70)
    else:
        print('=' * 70)
        print(f'FAILED (ctest exit {rc})')
        print('=' * 70)
        sys.exit(rc)


# -----------------------------------------------------------------------------
def _run(cmd):
    """Run a command, abort on non-zero exit."""
    print('  $', ' '.join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f'command failed (exit {rc}): {cmd!r}')


# =============================================================================
if __name__ == '__main__':
    main()
