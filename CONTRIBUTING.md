# Contributing to HPC-LLM-Forge

Thank you for your interest. This repository exists primarily as a teaching codex, but contributions that deepen the theoretical exposition or extend the reference implementations are welcome.

## Development workflow

1. Fork the repository and create a feature branch off `main`.
2. Install development dependencies:
   ```bash
   make install-dev
   ```
3. Make your changes, with tests where applicable.
4. Run the local quality gate:
   ```bash
   make lint
   make test
   ```
5. Open a pull request. Describe the change, link to any relevant paper or derivation, and include throughput / memory numbers if you touched a hot path.

## Style

- All Python code must pass `ruff` and `black` with the configured rules.
- Public functions require docstrings; complex algorithms additionally require a short *Theory* block in the docstring that cites the equations they implement.
- Notebooks should have one idea per cell and a clear narrative; avoid hidden state.

## Theory contributions

If you add a new `docs/theory/*.md`, please include:

1. A precise problem statement.
2. A proof or derivation with intermediate steps.
3. A small worked numerical example.

## Reporting issues

Open a GitHub issue with:

- A minimal reproducer.
- The output of `python -c "import hpcllmforge; hpcllmforge.print_environment()"`.
- The hardware fingerprint via `python -c "from hpcllmforge.utils.topology import fingerprint; print(fingerprint())"`.
