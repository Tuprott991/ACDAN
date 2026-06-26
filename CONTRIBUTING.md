# Contributing to ACDAN

> Placeholder contributing guide — adapt to your lab / venue conventions.

Thanks for your interest in ACDAN. This is a research reference implementation;
contributions that improve clarity, reproducibility, or extend the architecture
cleanly are very welcome.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Ground rules

- **Offline-first.** No code path may download datasets or model weights. New
  data sources must be synthetic or go behind a `registry.py` adapter that
  degrades to a mock by default.
- **Minimal dependencies.** Runtime deps are `numpy` + `PyYAML`. Do not add heavy
  ML frameworks to the core; put optional integrations behind extras.
- **Interface-driven.** New backends implement the Protocols in `rewards.py` /
  `registry.py`; do not special-case backends inside the core modules.
- **Ablation-native.** A new module gets a flag in `AblationFlags` and a faithful
  disabled-path baseline.
- **Tested gradients.** Any new differentiable term must ship a finite-difference
  test (see `tests/test_dto.py`).
- **Honest claims.** Label mocks; do not assert SOTA without a real, evaluated
  backend.

## Style

- Type hints + docstrings on public functions.
- Keep modules small and single-purpose.
- Run `pytest -q` before opening a PR; add tests for new behaviour.

## Commit / PR checklist

- [ ] Tests pass (`pytest`).
- [ ] New public API has docstrings + type hints.
- [ ] Paper mapping updated (`docs/module_to_paper_mapping.md`) if relevant.
- [ ] No new required heavy dependencies.
