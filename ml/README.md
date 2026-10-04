# RepurposeMap ML: graph exploration

This folder holds the Python package for graph loading and exploration. It is currently at **Milestone 1: explore the graph**.

> **Research use only.** Outputs are hypotheses for expert review, not treatment recommendations. The current graph is **synthetic demo data**. Entity names start with `DEMO`, and the relationships are invented to exercise the code. Do not cite them as biology.

## Current limitations

- The graph is the synthetic sample in `data/sample/synthetic_demo_graph.csv`.
- **PrimeKG has not been integrated yet.** The loader reads an internal five-column edge schema. A PrimeKG adapter will come after we inspect the real file's columns.
- The CLI is for **local graph exploration only**. It has no web interface, no model, and no evidence lookup.

## Setup

Requires Python 3.12, run from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Commands

Run these from the repository root. Each command reads the synthetic demo graph by default. Add `--csv PATH` to load a different edge table.

### Graph statistics

```bash
python -m repurposemap stats
```

Prints total nodes and edges, plus counts by node type and by relation type.

### Entity search

```bash
python -m repurposemap search --query "drug"
python -m repurposemap search --query "protein" --max-results 3
```

Case-insensitive, partial-name search. Exact name matches are listed first. `--max-results` defaults to 10.

### Path finding

```bash
python -m repurposemap paths --source "DEMO Drug Alpha" --target "DEMO Disease Zeta"
python -m repurposemap paths --source "DEMO Drug Alpha" --target "DEMO Disease Eta" --max-path-length 4
python -m repurposemap paths --source "DEMO Drug Alpha" --target "DEMO Disease Zeta" --max-results 2
```

Finds directed paths, following each relation in its stored direction. Paths are printed shortest first and numbered:

```
4 path(s) from 'DEMO Drug Alpha' to 'DEMO Disease Zeta':

[1] 1 hop(s)
DEMO Drug Alpha
  --drug_indicated_for_disease-->
DEMO Disease Zeta
```

- `--max-path-length` sets the maximum number of hops. The default is 4.
- `--max-results` sets the maximum number of paths printed. The default is 10.
- If no path exists within the limit, the command says so and exits with code 0.
- Unknown entity names exit with code 1 and print an error to stderr.

### If `python -m repurposemap` cannot find the package

On macOS, a repository inside an iCloud-synced folder such as `~/Desktop` can cause the editable-install `.pth` file to be marked hidden. Python then ignores it. Create the virtual environment outside the synced folder, or run from the source directly:

```bash
PYTHONPATH=ml/src python -m repurposemap stats
```

The test suite does not need either workaround, because `pyproject.toml` sets `pythonpath = ["ml/src"]` for pytest.

## Tests

```bash
pytest
```
