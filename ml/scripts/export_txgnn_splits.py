"""Dump TxGNN's disease splits as plain (drug_id, disease_id) lists.

    pip install TxGNN dgl torch pandas      # plus PyG for the disease-area splits
    python ml/scripts/export_txgnn_splits.py --data-dir ~/txgnn_data \\
        --out-dir ml/experiments/txgnn_splits [--export-dir ~/rgcn-export]

TxGNN builds its splits with its own code (TxData.prepare_split) on its own
copy of PrimeKG (Dataverse datafile 7144484, not the kg.csv this project
uses, datafile 6180620). The first run downloads that file, so it needs
dataverse.harvard.edu and a machine with TxGNN installed. This script only
reads what prepare_split produced; it changes nothing in TxGNN.

Output per split (<out-dir>/<split>_seed<seed>.json): the indication pairs in
train / valid / test as raw TxGNN ids, the test disease ids, counts, and the
sha256 of TxGNN's kg.csv so the exact graph is on record. Indication pairs
are read from the 'rev_indication' rows (disease -> drug), because TxGNN's
disease-area filter is applied to those rows (process_disease_area_split).

With --export-dir it also reports how many TxGNN test diseases appear in this
project's locked test and validation sets. Id formats may differ between the
two; the report prints a few of each so you can check before trusting a zero.

Using these splits for a like-for-like number means training on TxGNN's
train pairs and scoring on its test pairs. Scoring a model trained on this
project's own split against TxGNN's test diseases would leak: those diseases'
treatments can be in this project's training data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_SPLITS = ("complex_disease", "cell_proliferation", "mental_health", "cardiovascular", "anemia", "adrenal_gland")


def norm_id(value) -> str:
    """TxGNN stringifies numeric ids as floats ('1234.0'); drop the '.0'."""
    text = str(value)
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pairs_from_frame(frame, idx2id_drug: dict, idx2id_disease: dict) -> list[list[str]]:
    """(drug_id, disease_id) for every rev_indication row (x = disease, y = drug)."""
    rows = frame[frame.relation == "rev_indication"]
    out = []
    for disease_idx, drug_idx in zip(rows.x_idx.values, rows.y_idx.values):
        out.append([norm_id(idx2id_drug[float(drug_idx)]), norm_id(idx2id_disease[float(disease_idx)])])
    return out


def overlap_report(test_disease_ids: set[str], export_dir: Path) -> dict:
    metadata = json.loads((export_dir / "metadata.json").read_text())
    mine = {
        "locked_test": {norm_id(x) for x in metadata["test_disease_ids"]},
        "validation": {norm_id(x) for x in metadata["val_disease_ids"]},
    }
    return {
        name: {"overlap_with_txgnn_test": len(test_disease_ids & ids), "size": len(ids), "sample_ids": sorted(ids)[:3]}
        for name, ids in mine.items()
    } | {"txgnn_test_sample_ids": sorted(test_disease_ids)[:3]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-dir", type=Path, default=None)
    args = parser.parse_args()

    from txgnn import TxData  # imported late so --help works without TxGNN installed

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        data = TxData(data_folder_path=str(args.data_dir))
        data.prepare_split(split=split, seed=args.seed)
        maps = data.retrieve_id_mapping()
        pairs = {
            name: pairs_from_frame(frame, maps["idx2id_drug"], maps["idx2id_disease"])
            for name, frame in (("train", data.df_train), ("valid", data.df_valid), ("test", data.df_test))
        }
        test_diseases = sorted({d for _, d in pairs["test"]})
        record = {
            "split": split,
            "seed": args.seed,
            "txgnn_kg_csv_sha256": sha256_of(args.data_dir / "kg.csv"),
            "relation": "indication",
            "counts": {name: len(p) for name, p in pairs.items()} | {"test_diseases": len(test_diseases)},
            "test_disease_ids": test_diseases,
            "pairs": pairs,
        }
        if args.export_dir is not None:
            record["overlap_with_this_project"] = overlap_report(set(test_diseases), args.export_dir)
        target = args.out_dir / f"{split}_seed{args.seed}.json"
        target.write_text(json.dumps(record) + "\n")
        print(split, record["counts"], "->", target)


if __name__ == "__main__":
    main()
