# RepurposeMap

RepurposeMap is an explainable drug-repurposing research tool. It explores biomedical knowledge graphs to surface candidate drug-disease links, along with the biological paths that connect them. Every output is a **research hypothesis that requires expert review**.

## Important: research use only

RepurposeMap is **not** a treatment recommendation system. It does not:

- recommend drugs to patients
- claim that any drug treats any disease
- give dosing or clinical advice
- treat a path in the graph as evidence that a drug works. Connectivity in a knowledge graph is not evidence of treatment efficacy.

## Current status: Milestone 3, "Proper disease-based evaluation"

Milestone 1 (exploring the graph) is complete. It covers real PrimeKG integration, canonical node IDs, statistics, entity search, bounded path finding and the CLI.

Milestone 2 added the relation policy, hub and drug-promiscuity penalties, the path-quality score, and the first drug-ranking baseline.

Milestone 3 adds:

- **Disease-based evaluation.** Known indications are split by *disease ID*, not by edge: every treatment of a held-out disease is removed, all at once, into one training graph. See "why not split by edge" in [ml/README.md](ml/README.md).
- **Explicit leakage checks** that fail loudly if a held-out treatment edge, or a train/test overlap, is found.
- **Model-agnostic metrics.** Filtered rank, Hits@1/3/10 and MRR, built to score any future ranking (the path baseline today, TransE or an R-GCN later) without change.
- **Random-order baseline**, over the same candidates the path baseline found, for a fair comparison.
- **Updated `evaluate-baseline` CLI command**, now disease-based: `--test-diseases`, `--test-fraction`, `--seed`.

On a controlled 20-disease, 147-pair benchmark, the path baseline scores Hits@10 = 0.034 and MRR = 0.0158, against Hits@10 = 0.007 and MRR = 0.0035 for a random-order baseline over the same candidates — about 5x and 4.5x better than random, respectively. Absolute performance is still weak. This is a small, controlled evaluation, not a validated benchmark, and these numbers are not clinical evidence; the baseline and the harness exist as a comparison point for TransE and R-GCN. Details in [ml/README.md](ml/README.md).

Not implemented yet:

- **No machine learning.** The ranking is a non-ML baseline. The evaluation covers a few dozen diseases, and it does not support any claim of predictive performance.
- No TransE, R-GCN, or other trained model.
- No evidence lookup (PubMed, ClinicalTrials.gov) and no drug safety information.
- No web API (FastAPI) and no frontend (React).

## Data

### PrimeKG (real dataset, local only)

- **Dataset:** PrimeKG, the Precision Medicine Knowledge Graph.
- **Source:** Harvard Dataverse, https://doi.org/10.7910/DVN/IXA7BM (persistent identifier `doi:10.7910/DVN/IXA7BM`).
- **File used:** `kg.csv`, the edge file. It holds 8,100,498 edges over 129,375 nodes.
- **Retrieved:** 5 October 2026.
- **Release version:** not yet recorded. The Dataverse page could not be read from this environment when the file was retrieved, so the version and its date must be checked and added here.
- **Dataset terms:** not yet verified. The PrimeKG GitHub README says the code is MIT licensed, but dataset use is governed by separate terms. It also says each of the 20 integrated resources has its own license. Check those terms before redistributing any PrimeKG data.
- **Citation:** Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable precision medicine. *Scientific Data* 10, 67 (2023). https://doi.org/10.1038/s41597-023-01960-3

Local-only status:

- The raw `kg.csv` is **local-only**. Place it at `data/raw/primekg/kg.csv`. The CLI reads it from there by default, or from `--csv PATH`.
- `data/raw/` is listed in `.gitignore`, so the raw file is not committed.
- No PrimeKG rows are committed to this repository.

### Test fixture (synthetic)

- **`ml/tests/fixtures/primekg_mini.csv`** is a **fully synthetic** 15-row file. It has PrimeKG's real 12-column schema, node types and relation labels. Its names and identifiers are invented (`DEMO ...`, `DEMO-...`), and its node indices are invented too. No real PrimeKG row is copied.
- The tests use this fixture so they do not need the 936 MB raw file.

### Synthetic demo graph

- **`data/sample/synthetic_demo_graph.csv`** is **synthetic demo data**: 15 entities and 23 relationships. Every entity name starts with `DEMO`. The entities and relationships are invented to exercise the code. They are **not** real biology and must not be cited as facts.

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

The tests use the synthetic sample and the small PrimeKG fixture. They run in about a second.

## Run the CLI

The CLI reads the synthetic demo graph by default. Add `--source primekg` to read PrimeKG. Add `--csv PATH` to read a different edge file.

```bash
# synthetic demo (default)
python -m repurposemap stats
python -m repurposemap search --query "drug"
python -m repurposemap paths --source-name "DEMO Drug Alpha" --target-name "DEMO Disease Zeta"

# real PrimeKG (needs data/raw/primekg/kg.csv)
python -m repurposemap stats --source primekg
python -m repurposemap search --source primekg --query "sildenafil"
python -m repurposemap search --source primekg --query "metformin" --max-results 5
python -m repurposemap paths --source primekg --source-name "sildenafil" --target-name "pulmonary arterial hypertension"
python -m repurposemap paths --source primekg --source-id 14937 --target-id 38436 --max-path-length 2

# path quality: raw shortest paths next to quality-ranked paths
python -m repurposemap explain-paths --source primekg --source-name "sildenafil" --target-id 38436

# drug-ranking baseline (research hypothesis ranking, not a treatment recommendation)
python -m repurposemap rank-drugs --source primekg --disease "pulmonary arterial hypertension" --top-k 10

# disease-based evaluation, with a random-order baseline for comparison (not a benchmark)
python -m repurposemap evaluate-baseline --source primekg --test-diseases 20 --seed 42
```

The scoring commands (`explain-paths`, `rank-drugs`, `evaluate-baseline`) need `--source primekg`. Their relation policy is written for PrimeKG. Full options and the formulas are in [ml/README.md](ml/README.md).

Options:

- `--source {sample,primekg}`: which dataset to load. Default `sample`.
- `--csv PATH`: the edge file to load. Defaults: `data/sample/synthetic_demo_graph.csv` or `data/raw/primekg/kg.csv`.
- `search --query TEXT` and `search --max-results N` (default 10).
- `paths` takes each endpoint either as `--source-name` / `--target-name`, or as `--source-id` / `--target-id`. Exactly one is required for each endpoint.
- `paths --max-path-length N` (default 4 hops), `--max-results N` (default 10 paths), and `--time-limit SECONDS` (default 20).

The `paths` options `--source` and `--target` now name the endpoints as `--source-name` and `--target-name`. `--source` selects the dataset.

### Name ambiguity

Several PrimeKG entities share a name. The fixture has two entities named `DEMO Compound Pi`, a drug and an exposure, so the same ambiguity can be reproduced on the test data. A name that matches more than one entity is an error. The error lists each candidate with its ID, so you can choose by ID:

```
error: source name 'DEMO Compound Pi' matches 2 entities. Use --source-id:
  id=105  DEMO Compound Pi (drug, external_id=DEMO-DRUG-PI)
  id=106  DEMO Compound Pi (exposure, external_id=DEMO-EXP-PI)
```

The CLI never picks one of several candidates silently.

## Node identity

- **Graph key:** PrimeKG's node index, written as a string such as `"14937"`. It is unique for each node in the file.
- **Why not the source ID:** PrimeKG's `*_id` values are not unique. Some are shared across node types, and some gene IDs are shared within a type. The index is the only reliable unique key in the file.
- **Attributes on each node:** `name`, `node_type`, `external_id` (PrimeKG's `*_id`, such as a DrugBank ID for a drug) and `provenance` (PrimeKG's `*_source`, such as `DrugBank`).
- **Search** returns the ID, name, type, and external ID.
- **Path output** shows names. The header shows each endpoint's ID when its name is not unique.

The synthetic sample has no IDs, so its node key is its name.

## Relation direction

Edges keep the direction stored in the file. The adapter does not add reverse edges, and it does not assume that a relation is symmetric. In the full file, every relation is listed in both directions, so the stored direction and the reverse edge are both present. The adapter keeps whatever the file contains.

Each edge carries `relation` and `display_relation`. Edges between one node pair are kept separately when they have different relations or labels. For example, the fixture's `DEMO Compound Pi` and `DEMO Protein Rho` are linked by both a `carrier` and a `target` edge.

## Path-search limits

Path search is bounded. It does not enumerate every path in PrimeKG.

- It returns paths **shortest first**. Within one length, the order follows the file and is deterministic.
- It stops as soon as `--max-results` paths are found.
- It never walks more than `--max-path-length` hops (default 4).
- It stops at a time limit (`--time-limit`, default 20 seconds) or an edge-check budget. When it stops early it says so. An empty result with "stopped early" does **not** mean that no path exists.
- It does not rank paths by biological relevance.

## Performance on the full PrimeKG file

Measured on this machine, an 8-core Mac with 8 GB RAM. Three full loads were timed, and the values below are the ranges seen:

- Load: about 75 to 125 seconds, from the start of the process to the finished graph. Peak memory footprint: 5.2 to 6.3 GB.
- Entity search by name: about 150 to 170 milliseconds.
- Bounded path query: under 0.3 seconds for the queries tested.

The full graph is held in memory by NetworkX. The machine needs roughly 6 GB free to load it, so close other large applications first.

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
    │   ├── adapters/       # source-specific parsing (PrimeKG)
    │   ├── graph/          # generic loader, stats, search, bounded paths
    │   ├── scoring/        # relation policy, hub penalty, path-quality score
    │   ├── ranking/        # drug-ranking baseline
    │   └── evaluation/     # held-out indication harness
    └── tests/              # pytest suite, fixtures and small test graphs
```

## Roadmap

These are planned, not implemented:

1. A larger disease-based evaluation, beyond the small, fast run used for sanity checking.
2. TransE knowledge-graph embedding baseline, scored with the same evaluation harness.
3. Relational graph neural network for drug-disease link prediction.
4. Explanations built from biological paths.
5. Supporting evidence from PubMed and ClinicalTrials.gov.
6. Drug safety information.
7. FastAPI backend and React + TypeScript + Tailwind frontend.
