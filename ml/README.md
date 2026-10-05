# RepurposeMap ML: graph exploration

This folder holds the Python package for graph loading and exploration. It is at **Milestone 1: explore the graph**. The main README has the full picture. This file lists the package layout and the commands.

> **Research use only.** Outputs are hypotheses for expert review, not treatment recommendations. A path in the graph is not evidence of efficacy. The synthetic sample is invented data. Do not cite it as biology.

## Package layout

- `repurposemap/adapters/primekg.py`: parses PrimeKG's real `kg.csv` (12 columns) into the generic graph. All PrimeKG-specific logic lives here.
- `repurposemap/graph/loader.py`: loads the synthetic five-column edge table.
- `repurposemap/graph/search.py`: name search and exact-name lookup. Returns IDs, names, types and external IDs.
- `repurposemap/graph/paths.py`: bounded directed path search (`find_paths`, `find_paths_detailed`) and formatting.
- `repurposemap/graph/stats.py`: counts by node type and relation.
- `repurposemap/cli.py`: argument parsing and output only.

## Node identity

Nodes are keyed by a canonical ID:

- **PrimeKG:** the node index from the file, as a string. Each node carries `name`, `node_type`, `external_id` (PrimeKG's `*_id`) and `provenance`.
- **Synthetic sample:** the entity name, which is also its `name`.

Names are never graph keys, because PrimeKG has names shared by several nodes. The CLI resolves a name to an ID only when it matches exactly one node. Otherwise it lists the candidates.

## Commands

Run these from the repository root. Each command reads the synthetic demo graph by default. Add `--source primekg` to read PrimeKG, and `--csv PATH` to load a different file.

```bash
python -m repurposemap stats
python -m repurposemap search --query "drug"
python -m repurposemap paths --source-name "DEMO Drug Alpha" --target-name "DEMO Disease Zeta"

python -m repurposemap stats --source primekg
python -m repurposemap search --source primekg --query "sildenafil"
python -m repurposemap paths --source primekg --source-name "sildenafil" --target-id 38436 --max-path-length 2
```

Path options: `--max-path-length` (default 4 hops), `--max-results` (default 10), `--time-limit` (default 20 seconds). If a search stops at a limit, the output says so, and an empty result is then not proof that no path exists.

## Tests

```bash
pytest
```

The tests use the synthetic sample and a fully synthetic 15-row fixture in `tests/fixtures/` that matches the PrimeKG schema. They do not need the full 936 MB file.

If `python -m repurposemap` cannot find the package on macOS, see the troubleshooting section in the main README.
