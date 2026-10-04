# RepurposeMap

RepurposeMap is an explainable drug-repurposing research tool. It explores biomedical knowledge graphs to surface candidate drug-disease links, along with the biological paths that connect them. Every output is a **research hypothesis that requires expert review**.

## Important: research use only

RepurposeMap is **not** a treatment recommendation system. It does not:

- recommend drugs to patients
- claim that any drug treats any disease
- give dosing or clinical advice

## Current status: Milestone 1, "Explore the graph"

Implemented:

- **Graph loading** from a five-column edge table (`source_name, source_type, relation, target_name, target_type`) into a directed multigraph.
- **Validation**: missing columns, empty values, empty files and names used with two different types all raise clear errors.
- **Statistics**: total nodes and edges, plus counts by node type and by relation type.
- **Entity search**: case-insensitive partial-name search with a result limit.
- **Path finding**: short directed paths between two entities, with relation names preserved and a configurable hop limit.
- **Command-line interface** for local exploration of an edge table.

Not implemented yet:

- **PrimeKG is not integrated.** The loader reads our internal edge schema, not PrimeKG's file format.
- **The graph data is synthetic demo data**, not real biology.
- No machine learning, ranking models or link prediction.
- No evidence lookup (PubMed, ClinicalTrials.gov) and no drug safety information.
- No web API (FastAPI) and no frontend (React).

## Data

- **`data/sample/synthetic_demo_graph.csv`** is **synthetic demo data**: 15 entities and 23 relationships. Every entity name starts with `DEMO`. The entities and relationships are invented to exercise the code. They are **not** real biology and must not be cited as facts.
- **`data/raw/`** is reserved for real datasets such as PrimeKG. It is gitignored because the files are large. Do not commit them.

## Setup

Requires Python 3.12. Run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The dependencies are pandas and networkx. pytest is a development dependency.

## Run the tests

```bash
python -m pytest
```

## Run the CLI

The CLI reads the synthetic demo graph by default. Add `--csv PATH` to read a different edge table.

```bash
python -m repurposemap stats
python -m repurposemap search --query "drug"
python -m repurposemap search --query "protein" --max-results 3
python -m repurposemap paths --source "DEMO Drug Alpha" --target "DEMO Disease Zeta"
python -m repurposemap paths --source "DEMO Drug Alpha" --target "DEMO Disease Eta" --max-path-length 4 --max-results 5
```

Options:

- `search --max-results` (default 10)
- `paths --max-path-length` (default 4 hops) and `paths --max-results` (default 10 paths)

Example output from `paths`:

```
1 path(s) from 'DEMO Drug Alpha' to 'DEMO Disease Eta':

[1] 4 hop(s)
DEMO Drug Alpha
  --drug_targets_protein-->
DEMO Protein Kinase 1
  --protein_interacts_with_protein-->
DEMO Protein X
  --protein_interacts_with_protein-->
DEMO Protein Y
  --protein_associated_with_disease-->
DEMO Disease Eta
```

Paths follow each relation in its stored direction. Exit code 1 means an error, such as an unknown entity name or a missing file. A search or path query with no results exits with code 0.

## Troubleshooting

**`No module named repurposemap`.** The editable install did not take effect. This happened on macOS when the repository sat in an iCloud-synced folder such as `~/Desktop`. macOS marked the `.pth` file hidden, and Python ignores hidden `.pth` files. Two options:

- Create the virtual environment outside the synced folder, for example `python3.12 -m venv ~/.venvs/repurposemap`, then reinstall with `pip install -e ".[dev]"`.
- Run the CLI without installing, from the repository root: `PYTHONPATH=ml/src python -m repurposemap stats`.

Tests do not need either fix. `pyproject.toml` sets `pythonpath = ["ml/src"]` for pytest.

## Repository layout

```
RepurposeMap/
├── pyproject.toml          # package metadata, dependencies, pytest settings
├── README.md
├── data/
│   ├── raw/                # real datasets such as PrimeKG (gitignored)
│   └── sample/             # synthetic demo data (committed)
└── ml/
    ├── README.md           # graph exploration details
    ├── src/repurposemap/   # Python package
    │   ├── cli.py          # command-line interface
    │   └── graph/          # loader, stats, search, paths
    └── tests/              # pytest suite
```

## Roadmap

These are planned, not implemented:

1. Load PrimeKG through an adapter, after inspecting its real column layout.
2. Path-based drug ranking baseline.
3. TransE knowledge-graph embedding baseline.
4. Relational graph neural network for drug-disease link prediction.
5. Explanations built from biological paths.
6. Supporting evidence from PubMed and ClinicalTrials.gov.
7. Drug safety information.
8. FastAPI backend and React + TypeScript + Tailwind frontend.
