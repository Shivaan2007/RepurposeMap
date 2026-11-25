"""Post-checkpoint closure, one stage per step, in the order it must run.

    python scripts/post_checkpoint.py verify   --checkpoint F --final-epoch 2 [--provenance colab_provenance.json]
    python scripts/post_checkpoint.py evaluate --checkpoint F --final-epoch 2 --export-dir E --manifest M --approve-single-locked-test-run
    python scripts/post_checkpoint.py fill
    python scripts/post_checkpoint.py export   --checkpoint F --export-dir E --disease-id ID [--disease-id ID ...]
    python scripts/post_checkpoint.py case-study --checkpoint F --export-dir E --disease-id ID --known-drug-id ID --slug NAME
    python scripts/post_checkpoint.py validate-serving
    python scripts/post_checkpoint.py placeholders   # unfilled {{FILL}} / TBD markers in release-facing files
    python scripts/post_checkpoint.py qa             # pytest, frontend build, OpenAPI drift, git status, placeholders

This changes no frozen choice: the expected values below are copied from
ml/RGCN_EVALUATION_PROTOCOL.md, docs/FINAL_RELEASE_CHECKLIST.md and the
validation-search record. A mismatch stops the run; nothing is adjusted to fit.

What the code enforces (not just documents):
  * `evaluate` runs `verify` first and refuses on any failure.
  * `evaluate` needs --approve-single-locked-test-run and refuses if a locked-test
    result record already exists. If an earlier attempt started but never produced a
    record, it refuses unless --allow-restart-after-incomplete is given.
  * the raw output goes to ml/results/, and only a complete experiment record goes
    to ml/experiments/. Everything in ml/experiments/ must follow the experiment
    schema (the provenance loader and its test read every file there).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO / "ml" / "experiments"
RESULTS = REPO / "ml" / "results"
SEARCH_RECORD = EXPERIMENTS / "rgcn_sampled_validation_search_0f23fe3.json"
MARKER = RESULTS / ".rgcn_locked_test_started"
RECORD_GLOB = "rgcn_final_locked_test_*.json"

# Frozen: ml/RGCN_EVALUATION_PROTOCOL.md and docs/FINAL_RELEASE_CHECKLIST.md step 0.
FROZEN_CONFIG = {
    "hidden_dim": 128, "num_bases": 8, "num_layers": 2, "margin": 1.0, "learning_rate": 0.001,
    "batch_size": 1024, "negatives_per_positive": 4, "seed": 42, "fanout_hop1": 15, "fanout_hop2": 10,
    "early_stopping_patience": None,
}
FROZEN_SELECTED_NAME = "rgcn-wide"
FROZEN_SOURCE_COMMIT = "0f23fe3631e276f36a8ba1e481da2beb634b6ccf"
FROZEN_ARCHIVE_SHA256 = "bd26185f5db0ee0f5918936649c9f371d663034fad7bfff63939e3e9d1782085"
FROZEN_EXPORT_FINGERPRINT = "8e41f866f9fb90b0017b9056c1eb69b585004e01e65e6c6e4d17b9dbf16cb1aa"
FROZEN_PRIMEKG_SHA256 = "0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69"
FROZEN_N_EPOCHS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_checkpoint(checkpoint: dict, final_epoch: int, provenance: dict | None = None) -> list[str]:
    """Every reason this checkpoint is not the frozen final model (empty list = verified)."""

    problems: list[str] = []

    def expect(label: str, actual, expected) -> None:
        if actual != expected:
            problems.append(f"{label}: expected {expected!r}, found {actual!r}")

    if checkpoint.get("is_final_training") is not True:
        problems.append(f"is_final_training is {checkpoint.get('is_final_training')!r}, not True")
    expect("final epoch (0-indexed) in checkpoint", checkpoint.get("epoch"), final_epoch)
    expect("final epoch given vs. frozen (3 epochs, 0-indexed last = 2)", final_epoch, FROZEN_N_EPOCHS - 1)
    config = checkpoint.get("config") or {}
    for key, expected in FROZEN_CONFIG.items():
        expect(f"config.{key}", config.get(key), expected)
    expect("config.epochs", config.get("epochs"), FROZEN_N_EPOCHS)
    expect("config.validate_every", config.get("validate_every"), None)
    search = checkpoint.get("hyperparameter_search") or {}
    expect("hyperparameter_search.selected_name", search.get("selected_name"), FROZEN_SELECTED_NAME)
    expect("export_fingerprint", checkpoint.get("export_fingerprint"), FROZEN_EXPORT_FINGERPRINT)
    expect("primekg_fingerprint.sha256", (checkpoint.get("primekg_fingerprint") or {}).get("sha256"), FROZEN_PRIMEKG_SHA256)
    if not checkpoint.get("locked_test_manifest_fingerprint"):
        problems.append("locked_test_manifest_fingerprint is missing from the checkpoint")
    commit = checkpoint.get("git_commit")
    if commit is not None:  # the Colab launcher may record it only in its provenance file
        expect("git_commit", commit, FROZEN_SOURCE_COMMIT)
    losses = checkpoint.get("train_loss_curve") or []
    if len(losses) != FROZEN_N_EPOCHS or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses):
        problems.append(f"train_loss_curve should be {FROZEN_N_EPOCHS} finite values, found {losses!r}")

    if provenance is not None:
        expect("provenance.epochs", provenance.get("epochs"), FROZEN_N_EPOCHS)
        expect("provenance.final_epoch_reached", provenance.get("final_epoch_reached"), final_epoch)
        expect("provenance.stopped_for_time_budget", provenance.get("stopped_for_time_budget"), False)
        expect("provenance.is_final_training", provenance.get("is_final_training"), True)
        expect("provenance.selected_name", provenance.get("selected_name"), FROZEN_SELECTED_NAME)
        expect("provenance.source_commit", provenance.get("source_commit"), FROZEN_SOURCE_COMMIT)
        expect("provenance.source_archive_sha256", provenance.get("source_archive_sha256"), FROZEN_ARCHIVE_SHA256)
        expect("provenance.locked_test_evaluation_performed", provenance.get("locked_test_evaluation_performed"), False)
    return problems


def guard_single_evaluation(experiments: Path, marker: Path, allow_restart: bool) -> str | None:
    """None if a locked-test evaluation may start now, else the reason it may not."""

    done = sorted(experiments.glob(RECORD_GLOB))
    if done:
        return f"a locked-test result already exists ({done[0].name}); the evaluation runs exactly once"
    if marker.exists() and not allow_restart:
        return (f"{marker.name} says an evaluation was started and produced no record. Find out why before "
                "running again; pass --allow-restart-after-incomplete only if no locked-test labels were scored")
    return None


def build_experiment_record(raw: dict, *, checkpoint: dict, checkpoint_sha256: str, final_epoch: int, generated_at: str) -> dict:
    """The schema-valid experiment record for ml/experiments/. The flat metric keys
    (split, hits_at_*, mrr, macro_auprc, micro_auprc) stay at top level because
    ml/scripts/fill_final_results.py reads them from there."""

    if raw.get("split") != "locked-test":
        raise ValueError(f"raw artifact split is {raw.get('split')!r}, not 'locked-test'")
    return {
        "stage": "R-GCN final training + exactly-once locked-test evaluation",
        "purpose": "the one locked-test result for the frozen sampled R-GCN (rgcn-wide); not a tuning run and not re-runnable",
        "generated_at": generated_at,
        "provenance": {
            "source_commit": FROZEN_SOURCE_COMMIT,
            "archive_sha256": FROZEN_ARCHIVE_SHA256,
            "export_fingerprint": checkpoint.get("export_fingerprint"),
            "primekg_sha256": (checkpoint.get("primekg_fingerprint") or {}).get("sha256"),
            "locked_test_manifest_fingerprint": checkpoint.get("locked_test_manifest_fingerprint"),
            "checkpoint_sha256": checkpoint_sha256,
        },
        "safety_and_protocol": {
            "final_training_performed": True,
            "locked_test_evaluation_performed": True,
            "locked_test_evaluation_count": 1,
            "three_config_search_performed": True,
            "no_post_test_retuning": True,
        },
        "final_training": {"selected_name": FROZEN_SELECTED_NAME, "final_epoch_zero_indexed": final_epoch, "config": checkpoint.get("config")},
        **{key: raw[key] for key in ("split", "n_pairs", "hits_at_1", "hits_at_3", "hits_at_10", "mrr", "macro_auprc", "micro_auprc")},
    }


def _load_checkpoint(path: Path) -> dict:
    sys.path.insert(0, str(REPO / "ml" / "src"))
    from repurposemap.embeddings.checkpointing import load_checkpoint

    return load_checkpoint(path, map_location="cpu")


def _run(command: list[str], **kwargs) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "ml" / "src")}, **kwargs)


def cmd_verify(args) -> int:
    checkpoint = _load_checkpoint(args.checkpoint)
    provenance = json.loads(args.provenance.read_text()) if args.provenance else None
    problems = check_checkpoint(checkpoint, args.final_epoch, provenance)
    print(f"checkpoint sha256: {sha256_file(args.checkpoint)}")
    if provenance is None:
        print("note: no --provenance given; only the checkpoint's own fields were checked "
              "(commit/archive hash/time-budget flag come from colab_provenance.json)")
    if problems:
        print("VERIFY FAILED:\n  - " + "\n  - ".join(problems))
        return 1
    print("VERIFIED: matches the frozen spec.")
    return 0


def cmd_evaluate(args) -> int:
    if not args.approve_single_locked_test_run:
        print("refusing: pass --approve-single-locked-test-run to confirm this is the one locked-test evaluation")
        return 2
    if cmd_verify(args) != 0:
        return 1
    reason = guard_single_evaluation(EXPERIMENTS, MARKER, args.allow_restart_after_incomplete)
    if reason:
        print("refusing: " + reason)
        return 2
    checkpoint = _load_checkpoint(args.checkpoint)
    ckpt_sha = sha256_file(args.checkpoint)
    RESULTS.mkdir(parents=True, exist_ok=True)
    MARKER.write_text(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(), "checkpoint_sha256": ckpt_sha}))
    raw_path = RESULTS / f"rgcn_final_locked_test_{ckpt_sha[:8]}_raw.json"
    per_disease = RESULTS / f"rgcn_final_locked_test_{ckpt_sha[:8]}_per_disease.json"
    _run([sys.executable, "-m", "repurposemap", "rgcn-evaluate", "--export-dir", str(args.export_dir),
          "--checkpoint", str(args.checkpoint), "--split", "locked-test", "--manifest", str(args.manifest),
          "--artifact", str(raw_path), "--per-disease-output", str(per_disease)])
    raw = json.loads(raw_path.read_text())
    record = build_experiment_record(raw, checkpoint=checkpoint, checkpoint_sha256=ckpt_sha,
                                     final_epoch=args.final_epoch, generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    target = EXPERIMENTS / f"rgcn_final_locked_test_{ckpt_sha[:8]}.json"
    target.write_text(json.dumps(record, indent=2) + "\n")
    target.chmod(0o444)  # frozen: a record is never edited, only superseded by a new dated note
    print(f"frozen record: {target.relative_to(REPO)}\nraw outputs:   {raw_path.relative_to(REPO)}, {per_disease.relative_to(REPO)}")
    return 0


def _frozen_record() -> Path:
    found = sorted(EXPERIMENTS.glob(RECORD_GLOB))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one {RECORD_GLOB} in ml/experiments/ (run `evaluate` first), found {len(found)}")
    return found[0]


def cmd_fill(_args) -> int:
    _run([sys.executable, "ml/scripts/fill_final_results.py", "--artifact", str(_frozen_record())])
    return 0


def cmd_export(args) -> int:
    record = _frozen_record()
    command = [sys.executable, "ml/scripts/export_model_rankings.py", "--export-dir", str(args.export_dir), "--checkpoint", str(args.checkpoint),
               "--model", "rgcn", "--top-k", str(args.top_k), "--ranking-method", "rgcn",
               "--source-experiment-file", str(record), "--output-dir", "data/serving/"]
    for disease_id in args.disease_id:
        command += ["--disease-id", disease_id]
    if args.primekg_csv:
        command += ["--primekg-csv", str(args.primekg_csv)]
    _run(command)
    return 0


def cmd_case_study(args) -> int:
    _frozen_record()  # no case study before the locked-test result exists
    out = REPO / "case_studies" / f"{args.slug}.json"
    out.parent.mkdir(exist_ok=True)
    command = [sys.executable, "ml/scripts/run_case_study.py", "--export-dir", str(args.export_dir), "--checkpoint", str(args.checkpoint),
               "--model", "rgcn", "--disease-id", args.disease_id, "--known-drug-id", args.known_drug_id,
               "--ranking-method", "rgcn", "--output", str(out)]
    if args.primekg_csv:
        command += ["--primekg-csv", str(args.primekg_csv)]
    _run(command)
    return 0


def cmd_validate_serving(args) -> int:
    sys.path.insert(0, str(REPO / "ml" / "src"))
    from fastapi.testclient import TestClient

    from repurposemap.api.app import app
    from repurposemap.serving.loader import list_supported_diseases
    from repurposemap.serving.schemas import validate_bundle

    bundles = list_supported_diseases()
    if not bundles:
        print("FAILED: data/serving/ holds no valid bundle")
        return 1
    client, failures = TestClient(app), []
    listed = {d["disease_id"] for d in client.get("/api/v1/rank/diseases").json()["diseases"]}
    for bundle in bundles:
        validate_bundle(bundle)
        if bundle.disease_id not in listed:
            failures.append(f"{bundle.disease_id}: missing from GET /api/v1/rank/diseases")
        body = client.post("/api/v1/rank", json={"disease_id": bundle.disease_id}).json()
        if body.get("serving_mode") != "precomputed" or not body.get("candidates"):
            failures.append(f"{bundle.disease_id}: POST /api/v1/rank did not return precomputed candidates")
    unknown = client.post("/api/v1/rank", json={"disease_name": "no such disease zzz"}).json()
    if unknown.get("model_status") != "not_available":
        failures.append("an unsupported disease did not return model_status 'not_available'")
    print(f"{len(bundles)} bundle(s) checked")
    print("FAILED:\n  - " + "\n  - ".join(failures) if failures else "OK: bundles valid, listed, served as precomputed; unsupported disease honest")
    return 1 if failures else 0


RELEASE_FILES = ["README.md", "docs/FINAL_RESULTS.md", "docs/PORTFOLIO.md", "docs/DEMO_MATERIALS.md"]
PLACEHOLDER_MARKERS = ("{{FILL", "`TBD`", "TBD |")


def find_placeholders(root: Path, files: list[str] = RELEASE_FILES) -> list[str]:
    """`path:line: text` for every unfilled marker left in the release-facing files."""

    found = []
    for name in files:
        path = root / name
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if any(marker in line for marker in PLACEHOLDER_MARKERS):
                found.append(f"{name}:{number}: {line.strip()[:110]}")
    return found


def cmd_placeholders(_args) -> int:
    left = find_placeholders(REPO)
    print("\n".join(left) if left else "none: no {{FILL}} or TBD markers left in the release-facing files")
    return 1 if left else 0


def cmd_qa(_args) -> int:
    _run([sys.executable, "-m", "pytest", "-q"])
    subprocess.run(["npm", "run", "build"], cwd=REPO / "web" / "app", check=True)
    check = RESULTS / "openapi_check.json"
    RESULTS.mkdir(parents=True, exist_ok=True)
    _run([sys.executable, "ml/scripts/export_openapi.py", "--output", str(check)])
    drift = subprocess.run(["diff", "-q", "openapi.json", str(check)], cwd=REPO).returncode
    check.unlink(missing_ok=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True).stdout
    print("OpenAPI drift:", "none" if drift == 0 else "DRIFT")
    print("git status:", "clean" if not status.strip() else "\n" + status)
    left = cmd_placeholders(_args)
    return 0 if drift == 0 and left == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def common(p, evaluation=False):
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--final-epoch", type=int, required=evaluation, help="0-indexed last epoch; the frozen value is 2")
        p.add_argument("--provenance", type=Path, default=None, help="colab_provenance.json")

    verify = sub.add_parser("verify"); common(verify, True); verify.set_defaults(func=cmd_verify)
    evaluate = sub.add_parser("evaluate"); common(evaluate, True)
    evaluate.add_argument("--export-dir", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True, help="the LOCAL-ONLY locked_test_manifest.json")
    evaluate.add_argument("--approve-single-locked-test-run", action="store_true")
    evaluate.add_argument("--allow-restart-after-incomplete", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)
    sub.add_parser("fill").set_defaults(func=cmd_fill)
    export = sub.add_parser("export")
    export.add_argument("--checkpoint", type=Path, required=True); export.add_argument("--export-dir", type=Path, required=True)
    export.add_argument("--disease-id", action="append", required=True); export.add_argument("--top-k", type=int, default=10)
    export.add_argument("--primekg-csv", type=Path, default=None); export.set_defaults(func=cmd_export)
    case = sub.add_parser("case-study")
    case.add_argument("--checkpoint", type=Path, required=True); case.add_argument("--export-dir", type=Path, required=True)
    case.add_argument("--disease-id", required=True); case.add_argument("--known-drug-id", required=True)
    case.add_argument("--slug", required=True); case.add_argument("--primekg-csv", type=Path, default=None); case.set_defaults(func=cmd_case_study)
    sub.add_parser("validate-serving").set_defaults(func=cmd_validate_serving)
    sub.add_parser("placeholders").set_defaults(func=cmd_placeholders)
    sub.add_parser("qa").set_defaults(func=cmd_qa)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
