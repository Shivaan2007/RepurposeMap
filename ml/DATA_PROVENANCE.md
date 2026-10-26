# PrimeKG provenance and TxGNN splits (PLAN.md gaps 1 and 2)

Both scripts need dataverse.harvard.edu, which the cloud sandbox cannot reach (403). Run them on your Mac. Neither has been run against the real services; the parsing/id helpers were tested on made-up data only.

## 1. PrimeKG build date and version

`python ml/scripts/fetch_primekg_provenance.py --csv data/raw/primekg/kg.csv`

Writes `ml/experiments/primekg_provenance.json`: every Dataverse version of the dataset, the first one containing `kg.csv` (datafile 6180620), a suggested time-travel cutoff, and whether your local file's MD5 matches Dataverse's.

Do **not** put the result into `PrimeKGFingerprint.release_version` / `snapshot_build_date`. They are hashed into `export_fingerprint`, so filling them changes it and breaks the match with the TransE locked-test result (`8e41f866…cb1aa`) and the checkpoint eligibility gate. Cite the provenance file next to results instead.

**Result (run 2026-10-26 on the owner's Mac, output in `ml/experiments/primekg_provenance.json`):** the local `kg.csv` has MD5 `aac8191d4fbc5bf09cdf8c3c78b4e75f`, which matches Dataverse's. The same MD5 appears in all three dataset versions (1.0 released 2022-04-25, 2.0 on 2022-04-28, 2.1 on 2022-05-02), so the file has not changed since the first release. **Time-travel cutoff: 2022-04-25T23:45:53Z.** The dataset record was created 2021-07-12, so the graph's content may be older than the cutoff; the cutoff is the release, not a rebuilt past (state this in one sentence in the write-up, as PLAN.md says).

Background from the README (before the run) (from PrimeKG's README, 2026-10-26, maintainers' claims): first public on Dataverse in April 2022; the Dataverse files "remain unchanged" through the July 2023 rebuild; the README says nothing about whether the Dec 2023 OMIM extension was published there; PrimeKG is now superseded by OptimusKG. The exact release date is not known yet, so the cutoff is still open.

## 2. TxGNN splits

`python ml/scripts/export_txgnn_splits.py --data-dir ~/txgnn_data --out-dir ml/experiments/txgnn_splits --export-dir ~/rgcn-export`

What the code review found:
- TxGNN does not ship split files. It generates them (`TxData.prepare_split`, seed 42; `complex_disease` plus nine disease areas) on **its own kg.csv** (Dataverse datafile 7144484), not the file this project uses (6180620). The two graphs may differ, so check the `txgnn_kg_csv_sha256` in the output and compare ids before comparing numbers.
- Its default test set is 5% of treatments by disease, a different set from this project's 100 locked-test diseases.
- A like-for-like number therefore means retraining on TxGNN's train pairs and scoring on its test pairs. Scoring the current models (trained on this project's split) on TxGNN's test diseases would leak. That retraining is the "run on TxGNN's splits" item in phase 4 and is not done.
