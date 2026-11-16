"""Attach self-reported home regions using exact TripAdvisor usernames.

Profiles come from a separate corpus/snapshot. Age/gender are unavailable;
profile availability at the historical HotelRec review date is not guaranteed.
"""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

from prepare_hotelrec import INVALID_AUTHORS

PROFILE_PAPER = "https://ceur-ws.org/Vol-3498/paper8.pdf"


def attach_demographics(folder="datasets_raw/hotelrec_full_sample",
                        authors="datasets_raw/tripadvisor-us/authors.json.gz"):
    folder, authors = Path(folder), Path(authors)
    profiles_path = folder / "users_demographics.csv.gz"
    metadata_path = folder / "demographics_source.json"
    if profiles_path.exists() or metadata_path.exists():
        raise FileExistsError("HotelRec demographics already exist; preserve them before preparing another snapshot")
    source = json.loads((folder / "source.json").read_text(encoding="utf8"))
    ratings_path = folder / "reviews_slim.csv.gz"
    with ratings_path.open("rb") as stream:
        ratings_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if ratings_sha != source["prepared_sha256"][ratings_path.name]:
        raise ValueError("HotelRec ratings checksum mismatch")
    with gzip.open(ratings_path, "rt", encoding="utf8", newline="") as stream:
        user_ids = {row["user_id"] for row in csv.DictReader(stream)
                    if row["user_id"].strip().lower() not in INVALID_AUTHORS}
    if not user_ids or "" in user_ids:
        raise ValueError("HotelRec target users are empty or invalid")

    matches = defaultdict(set)
    scanned, invalid_matches = 0, 0
    with gzip.open(authors, "rt", encoding="utf8") as stream:
        for scanned, line in enumerate(stream, 1):
            profile = json.loads(line)
            username = profile.get("username")
            if not isinstance(username, str) or username not in user_ids:
                continue
            profile_id, hometown = profile.get("id"), profile.get("hometown")
            if (not isinstance(profile_id, str) or not profile_id.strip()
                    or profile_id.lower() in INVALID_AUTHORS
                    or (hometown is not None and not isinstance(hometown, str))):
                invalid_matches += 1
                continue
            hometown = " ".join((hometown or "").split())
            if hometown.lower() in INVALID_AUTHORS or hometown.lower() in {"none", "n/a", "nan"}:
                hometown = ""
            matches[username].add((profile_id, hometown))

    rows, statuses, regions = [], Counter(), Counter()
    for user_id in sorted(user_ids):
        candidates = matches.get(user_id, set())
        hometowns = {row[1] for row in candidates if row[1]}
        profile_id = hometown = region = ""
        if not candidates:
            status = "unmatched"
        elif len({row[0] for row in candidates}) > 1:
            status = "ambiguous_username"
        elif len(hometowns) > 1:
            status = "conflicting_hometown"
        else:
            profile_id = next(iter(candidates))[0]
            hometown = next(iter(hometowns), "")
            region = hometown.rsplit(",", 1)[-1].strip().casefold()
            if region in INVALID_AUTHORS or region in {"none", "n/a", "nan"}:
                region = ""
            status = "matched" if region else "matched_without_hometown"
        statuses[status] += 1
        if region:
            regions[region] += 1
        rows.append((user_id, profile_id, hometown, region, status))

    temporary = profiles_path.with_suffix(".tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["user_id", "profile_id", "hometown", "home_region", "match_status"])
            writer.writerows(rows)
        with temporary.open("rb") as stream:
            prepared_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        with authors.open("rb") as stream:
            authors_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        meta = {
            "feature": "home_region", "target_reviews_sha256": ratings_sha,
            "profiles_sha256": prepared_sha,
            "profile_source": {"file": str(authors), "sha256": authors_sha,
                               "paper": PROFILE_PAPER, "rows_scanned": scanned},
            "join": "Exact case-sensitive HotelRec author = profile username; ambiguous IDs/conflicting locations excluded",
            "transform": "Last comma-separated part of self-reported hometown; whitespace normalized and casefolded",
            "semantics": "Geographic home region: may be a state, country or city; not nationality",
            "counts": {"target_users": len(user_ids), "users_with_region": statuses["matched"],
                       "known_region_groups": len(regions), "invalid_matching_profiles": invalid_matches,
                       "match_status": dict(statuses)},
            "region_counts": dict(regions.most_common()),
            "unavailable_attributes": ["age", "gender"],
            "profile_observed_at": None,
            "temporal_limitation": "Separate profile snapshot; availability at historical HotelRec review dates is not guaranteed",
        }
        temporary.replace(profiles_path)
        metadata_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf8")
        print(json.dumps(meta, indent=2, ensure_ascii=False), flush=True)
        return meta
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=Path("datasets_raw/hotelrec_full_sample"))
    parser.add_argument("--authors", type=Path, default=Path("datasets_raw/tripadvisor-us/authors.json.gz"))
    args = parser.parse_args()
    attach_demographics(args.folder, args.authors)
