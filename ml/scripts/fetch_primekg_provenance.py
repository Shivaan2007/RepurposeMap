"""Record where the PrimeKG file came from, for the time-travel test's cutoff.

    python ml/scripts/fetch_primekg_provenance.py [--csv data/raw/primekg/kg.csv]

Run this on a machine that can reach dataverse.harvard.edu (the cloud
sandbox this repo was developed in cannot). It asks the Dataverse API for
every version of the PrimeKG dataset, finds the version(s) that contain the
kg.csv file this project uses (datafile 6180620), and writes
ml/experiments/primekg_provenance.json. With --csv it also compares the
local file's MD5 to the one Dataverse reports, which is how you confirm the
file on disk is that exact release.

Why a separate file and not PrimeKGFingerprint: those fields are hashed into
export_fingerprint (and from there into the locked-test manifest and the
checkpoint eligibility gate). Filling them in would change the fingerprint
and break the match with the existing TransE locked-test result. This file
sits next to the results instead and is cited with them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

DOI = "doi:10.7910/DVN/IXA7BM"
DATAFILE_ID = 6180620  # the kg.csv PrimeKG's README links to
VERSIONS_URL = f"https://dataverse.harvard.edu/api/datasets/:persistentId/versions?persistentId={DOI}"
OUTPUT = Path(__file__).resolve().parents[1] / "experiments" / "primekg_provenance.json"

# Facts read from PrimeKG's own README (github.com/mims-harvard/PrimeKG) on
# 2026-10-26. Recorded as claims by the maintainers, not independently checked.
README_CLAIMS = {
    "first_public_release": "April 2022 (README news: 'PrimeKG is live on bioRxiv and Harvard Dataverse')",
    "dataverse_files_after_july_2023_rebuild": "unchanged (README: 'the files published on Harvard DataVerse remain unchanged')",
    "dec_2023_omim_extension_on_dataverse": "not stated in the README",
    "superseded_by": "OptimusKG (README banner)",
}


def summarize_versions(payload: dict, datafile_id: int = DATAFILE_ID) -> dict:
    """Pure function over the Dataverse /versions response (easy to test)."""
    versions = payload.get("data", [])
    rows, md5s = [], set()
    for v in versions:
        match = [f for f in v.get("files", []) if f.get("dataFile", {}).get("id") == datafile_id]
        if match:
            checksum = match[0]["dataFile"].get("checksum", {})
            if checksum.get("type") == "MD5":
                md5s.add(checksum["value"])
        rows.append(
            {
                "version": f"{v.get('versionNumber')}.{v.get('versionMinorNumber', 0)}",
                "state": v.get("versionState"),
                "release_time": v.get("releaseTime"),
                "create_time": v.get("createTime"),
                "contains_datafile": bool(match),
            }
        )
    containing = [r for r in rows if r["contains_datafile"] and r["release_time"]]
    first = min(containing, key=lambda r: r["release_time"]) if containing else None
    return {
        "versions": rows,
        "first_version_containing_datafile": first,
        "datafile_md5_values_seen": sorted(md5s),
    }


def local_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=None, help="local kg.csv to check against Dataverse's MD5")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    # Dataverse answers 403 to urllib's default "Python-urllib" User-Agent.
    request = urllib.request.Request(VERSIONS_URL, headers={"User-Agent": "Mozilla/5.0 RepurposeMap-provenance", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Dataverse returned HTTP {error.code}: {error.read()[:500].decode(errors='replace')}")
    summary = summarize_versions(payload)

    record = {
        "dataverse_doi": DOI,
        "datafile_id": DATAFILE_ID,
        "local_sha256": "0a5a010fcd88dcec572a573d98c26c1c244a11567092b6161d3ded611cfb5c69",
        "local_retrieved_date": "2026-10-05",
        "readme_claims": README_CLAIMS,
        "dataverse": summary,
    }
    first = summary["first_version_containing_datafile"]
    record["suggested_time_travel_cutoff"] = first["release_time"] if first else None
    record["cutoff_note"] = (
        "Dataverse release time of the first version containing this file. The graph has no per-edge dates, "
        "so the cutoff is the snapshot's release, not a rebuilt past. The graph's true content date may be "
        "earlier, which only makes this cutoff conservative."
    )
    if args.csv is not None:
        mine = local_md5(args.csv)
        record["local_md5"] = mine
        record["local_file_matches_dataverse"] = mine in summary["datafile_md5_values_seen"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
