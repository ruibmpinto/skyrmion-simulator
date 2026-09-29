# Contributing

Contributions, bug reports and questions are welcome.

## Reporting problems and getting support

Open an issue at
<https://github.com/ruibmpinto/skyrmion-simulator/issues>. For a bug,
include the command you ran, the full error message, your Python version
and operating system, and, if possible, a minimal script that reproduces
it. Questions about using the package are also handled through issues.

## Contributing changes

1. Fork the repository and create a branch from `main`.
2. Install the package in editable mode with the test dependencies:

   ```bash
   pip install -e ".[test]"
   ```

3. Make the change, with a test when it adds or fixes behaviour.
4. Check that the tests and lint pass:

   ```bash
   python -m pytest tests
   flake8 src studies tests docs
   ```

5. Open a pull request against `main` describing what changed and why.
   CI runs the lint, the Python regression tests and the C++ build and
   tests on every pull request.

Changes to the physics should also pass the literature benchmarks in the
`validation/` folders of the package. Changes to code that the C++ port
mirrors must keep the parity tests in `src/simulator_cpp/tests/` passing;
see [`src/simulator_cpp/README.md`](src/simulator_cpp/README.md).

## Code style

- Lines of at most 80 characters.
- NumPy-style docstrings on every module, class and function.
- No type hints in function signatures; types go in docstrings.
- No module-level global variables.
- Single quotes for strings, double quotes only for docstrings.

`flake8` is configured in `.flake8` to match these rules.

## Where code belongs

General solver features go in `src/skyrmion_simulator/`. Code specific to
one study, such as a campaign's sweeps, figures or cluster scripts, goes
in its own folder under `studies/`.
