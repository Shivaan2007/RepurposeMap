"""Stage 1 of the time-travel test: the three top-50 lists for 50 locked-test diseases.

    python ml/scripts/time_travel_rankings.py --export-dir <export> --checkpoint <final.pt> \\
        --primekg-csv data/raw/primekg/kg.csv --output ml/results/time_travel/rankings.json

Follows ``ml/TIME_TRAVEL_PROTOCOL.md``. Reads only disease IDs and names (no locked-test label, no manifest). Needs the
final checkpoint and the export, so it runs where they live. Writes one JSON file the collector reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ml" / "src"))
sys.path.insert(0, str(REPO / "ml" / "scripts"))
sys.path.insert(0, str(REPO / "scripts"))

from repurposemap.evaluation import time_travel as tt  # noqa: E402


def popularity_degrees(final_triples, entity_index, drug_ids) -> dict[str, int]:
    """Head-or-tail occurrences of each drug in the final training triples."""

    index_to_drug = {entity_index.id_to_index[d]: d for d in drug_ids}
    counter: Counter[int] = Counter()
    for head, _relation, tail in final_triples:
        for node in (int(head), int(tail)):
            if node in index_to_drug:
                counter[node] += 1
    return {drug: counter.get(index, 0) for index, drug in index_to_drug.items()}


def build_rankings(disease_ids, popular_ids, drug_ids, rgcn_top, names) -> list[dict]:
    """Assemble the per-disease record. ``rgcn_top(disease_id)`` returns the 50 R-GCN drug IDs in rank order."""

    def entry(drug_id: str) -> dict:
        return {"drug_id": drug_id, "drug_name": names[drug_id]}

    return [
        {
            "disease_id": disease_id,
            "disease_name": names[disease_id],
            "rgcn": [entry(d) for d in rgcn_top(disease_id)],
            "popularity": [entry(d) for d in popular_ids],
            "random": [entry(d) for d in tt.random_drugs(list(drug_ids), disease_id)],
        }
        for disease_id in disease_ids
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--primekg-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    from export_model_rankings import FinalModelRequiredError, _build_score_fn, rank_disease
    from pick_case_studies import names_for
    from repurposemap.embeddings.export import load_export
    from repurposemap.embeddings.transe_train_gpu import load_checkpoint

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if not checkpoint.get("is_final_training"):
        raise FinalModelRequiredError(f"{args.checkpoint} is not the final checkpoint")
    loaded = load_export(args.export_dir)

    disease_ids = tt.sample_diseases(loaded.test_disease_ids)
    degrees = popularity_degrees(loaded.final_triples, loaded.entity_index, loaded.drug_ids)
    popular_ids = tt.popularity_top(degrees)
    score_fn = _build_score_fn("rgcn", checkpoint, loaded)

    def rgcn_top(disease_id: str) -> list[str]:
        ranked = rank_disease(disease_id, score_fn=score_fn, drug_ids=loaded.drug_ids, drug_names={}, top_k=tt.TOP_K)
        return [r["drug_id"] for r in ranked]

    rgcn = {d: rgcn_top(d) for d in disease_ids}
    random_ids = {d: tt.random_drugs(list(loaded.drug_ids), d) for d in disease_ids}
    wanted = set(disease_ids) | set(popular_ids) | {x for v in rgcn.values() for x in v} | {x for v in random_ids.values() for x in v}
    names = names_for(args.primekg_csv, wanted)
    missing = wanted - set(names)
    if missing:
        raise SystemExit(f"{len(missing)} nodes have no name in {args.primekg_csv}, e.g. {sorted(missing)[:5]}")

    diseases = build_rankings(disease_ids, popular_ids, loaded.drug_ids, rgcn.__getitem__, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cutoff": tt.CUTOFF, "top_k": tt.TOP_K, "n_diseases": len(diseases), "diseases": diseases}, indent=1) + "\n")
    print(f"wrote {args.output}: {len(diseases)} diseases, {tt.TOP_K} drugs per ranker")
    return 0


if __name__ == "__main__":
    sys.exit(main())
