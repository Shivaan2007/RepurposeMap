"""Choose the ranking diseases and case-study pairs by a fixed rule, before any model output exists.

    python scripts/pick_case_studies.py --export-dir <export> --primekg-csv data/raw/primekg/kg.csv \\
        [--n-diseases 5] [--n-case-studies 3]

The rule (structure only; no model score, rank or metric is read, so the choice cannot be tuned to
look good):
  * candidates are fit or validation diseases with at least one known drug in the training graph.
    Locked-test diseases are excluded: their indications are held out of the training graph, so a
    masked case study on one has nothing to mask.
  * rank candidates by how many edges touch the disease in the training graph (more edges, more
    possible explanation paths), ties by disease id.
  * the ranking diseases are the top N. The case studies are the first K of them, each paired with its
    known drug of lowest PrimeKG node index (arbitrary on purpose: not the most famous or best-ranked drug).

Prints the table and the exact post_checkpoint.py commands. Override any choice by editing the commands.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _key(node_id: str):
    return (0, int(node_id)) if node_id.isdigit() else (1, node_id)


def pick(loaded, n_diseases: int, n_case_studies: int) -> list[dict]:
    """The chosen diseases, in rule order: {disease_id, degree, drug_ids, case_drug_id or None}."""

    id_to_index = loaded.entity_index.id_to_index
    index_to_id = loaded.entity_index.index_to_id
    drug_indices = {id_to_index[d] for d in loaded.drug_ids}
    allowed = set(loaded.fit_disease_ids) | set(loaded.val_disease_ids)
    allowed -= set(loaded.test_disease_ids)

    degree: dict[int, int] = {}
    known: dict[int, set[str]] = {}
    for head, relation, tail in loaded.final_triples:
        degree[int(head)] = degree.get(int(head), 0) + 1
        degree[int(tail)] = degree.get(int(tail), 0) + 1
        if int(relation) != loaded.indication_relation_idx:
            continue
        for drug, disease in ((int(head), int(tail)), (int(tail), int(head))):
            if drug in drug_indices and disease not in drug_indices:
                known.setdefault(disease, set()).add(index_to_id[drug])

    rows = []
    for disease_id in allowed:
        index = id_to_index.get(disease_id)
        if index is not None and known.get(index):
            rows.append({"disease_id": disease_id, "degree": degree.get(index, 0), "drug_ids": sorted(known[index], key=_key)})
    rows.sort(key=lambda r: (-r["degree"], _key(r["disease_id"])))
    rows = rows[:n_diseases]
    for position, row in enumerate(rows):
        row["case_drug_id"] = row["drug_ids"][0] if position < n_case_studies else None
    return rows


def names_for(csv_path: Path, wanted: set[str]) -> dict[str, str]:
    """Display names for PrimeKG node indices, streamed so the 8M-row file is never held in memory."""

    found: dict[str, str] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for side in ("x", "y"):
                node = row[f"{side}_index"]
                if node in wanted and node not in found:
                    found[node] = row[f"{side}_name"]
            if len(found) == len(wanted):
                break
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--primekg-csv", type=Path, default=None, help="only for display names")
    parser.add_argument("--n-diseases", type=int, default=5)
    parser.add_argument("--n-case-studies", type=int, default=3)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(REPO / "ml" / "src"))
    from repurposemap.embeddings.export import load_export

    rows = pick(load_export(args.export_dir), args.n_diseases, args.n_case_studies)
    if not rows:
        print("no eligible disease: no fit/validation disease has a known drug in the training graph")
        return 1
    wanted = {r["disease_id"] for r in rows} | {r["case_drug_id"] for r in rows if r["case_drug_id"]}
    names = names_for(args.primekg_csv, wanted) if args.primekg_csv else {}
    print(f"{'disease id':>10}  {'edges':>6}  {'#drugs':>6}  disease / case-study drug")
    for r in rows:
        drug = f"  -> {r['case_drug_id']} {names.get(r['case_drug_id'], '')}".rstrip() if r["case_drug_id"] else ""
        print(f"{r['disease_id']:>10}  {r['degree']:>6}  {len(r['drug_ids']):>6}  {names.get(r['disease_id'], '?')}{drug}")
    ids = " ".join(f"--disease-id {r['disease_id']}" for r in rows)
    print("\nexport:\n  python scripts/post_checkpoint.py export --checkpoint <final.pt> --export-dir <export> " + ids)
    for r in rows:
        if r["case_drug_id"]:
            print(f"case study:\n  python scripts/post_checkpoint.py case-study --checkpoint <final.pt> --export-dir <export> "
                  f"--disease-id {r['disease_id']} --known-drug-id {r['case_drug_id']} --slug disease-{r['disease_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
