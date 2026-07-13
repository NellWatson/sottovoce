# Contributing to sottovoce

Thanks for your interest in improving sottovoce. This is a research-adjacent
project: correctness and honest calibration of claims matter as much as code
quality.

## Ground rules

- **Claims must trace to evidence.** Any performance number in the README,
  docstrings, or paper references should be reproducible and cited to a specific
  experiment. If a result does not replicate, correct it — do not headline the
  most flattering single run. See `CHANGELOG.md` for the standard this project
  now holds itself to.
- **Keep the API honest.** A probe's score is P(correct); `decide()` treats a
  high score as PASS. Any new probe type must preserve that contract.

## Development setup

```bash
git clone https://github.com/NellWatson/sottovoce.git
cd sottovoce
python -m venv .venv && source .venv/bin/activate
pip install -e ".[train,dev]"
```

`torch` and `numpy` are the only runtime dependencies. The `train` extra adds
`transformers`, `datasets`, `scikit-learn`, and `accelerate` for training probes
and running cross-model transfer; the `dev` extra adds test and lint tooling.

## Running the checks

```bash
pytest            # tests
ruff check .      # lint
```

A GPU is not required for the unit tests (probe loading, dtype, routing
polarity, projection shapes all run on CPU). Training a probe end-to-end
(`examples/quickstart.py`) does need a GPU and network access to fetch the model
and TriviaQA.

## Submitting changes

1. Open an issue describing the change first for anything non-trivial.
2. Branch from `main`; keep commits focused.
3. Add or update tests for behaviour changes, and update `CHANGELOG.md` under
   `[Unreleased]`.
4. Ensure `pytest` and `ruff check .` pass.
5. Open a pull request with a clear description and, for any claim change, a
   pointer to the supporting experiment.

## Reporting issues

Please include the model, the probe (base vs. transferred), the platform, and a
minimal reproduction. For suspected numeric or calibration problems, include the
AUROC/CW numbers you observed and how you measured them.
