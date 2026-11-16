"""Extract a bounded HotelRec prefix without downloading/unpacking the full 50 GB.

Academic research source: https://github.com/diegoantognini/HotelRec
The prefix is ordered by hotel, NOT a random sample of the full dataset.
"""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import struct
from datetime import datetime
from urllib.request import urlopen
import zlib
from functools import lru_cache

SOURCE_URL = "https://drive.switch.ch/public.php/dav/files/n48smsdufhRA7fR"
INVALID_AUTHORS = frozenset({"unknown", "anonymous", "null", "undefined", "/undefined", "a tripadvisor member"})


@lru_cache(maxsize=1024)
def hotel_features(url):
    # Consecutive reviews share a hotel URL; parse its static features once.
    hotel = re.fullmatch(r"Hotel_Review-g\d+-(d\d+)-Reviews-(.+)\.html", url)
    return (hotel[1], re.sub(r"[_-]+", " ", hotel[2])) if hotel else None


def parse_review(review):
    """Return only stable IDs, rating, date and URL features; reject invalid reviews."""
    author = str(review.get("author") or "").strip()
    hotel = hotel_features(review.get("hotel_url") or "")
    try:
        rating = float(review.get("rating"))
        date = datetime.fromisoformat(review["date"])
        timestamp = (date - datetime(1970, 1, 1)).total_seconds()
    except (ValueError, TypeError, KeyError):
        return None
    if (not author or author.lower() in INVALID_AUTHORS
            or not hotel or not math.isfinite(rating) or not 1 <= rating <= 5):
        return None
    return author, hotel[0], rating, int(timestamp), hotel[1]


def zip_lines(stream):
    """Stream the first deflated ZIP member; a prefix cannot verify full ZIP CRC."""
    header = stream.read(30)
    if len(header) != 30:
        raise ValueError("Incomplete ZIP header")
    signature, _, flags, method, *_, name_len, extra_len = struct.unpack("<4s5H3I2H", header)
    name = stream.read(name_len)
    extra = stream.read(extra_len)
    if (signature != b"PK\x03\x04" or flags & 1 or method != 8
            or name != b"HotelRec.txt" or len(extra) != extra_len):
        raise ValueError("Expected unencrypted deflated HotelRec.txt as first ZIP member")
    decoder = zlib.decompressobj(-15)
    pending = b""
    while chunk := stream.read(64 * 1024):
        lines = (pending + decoder.decompress(chunk)).split(b"\n")
        pending = lines.pop()
        yield from lines
        if decoder.eof:
            if pending:
                yield pending
            return
    raise ValueError("ZIP prefix ended before the requested number of reviews")


def prepare(stream, folder, max_reviews):
    if max_reviews < 1:
        raise ValueError("max_reviews must be positive")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    targets = [folder / name for name in ("reviews_slim.csv.gz", "hotels.csv.gz", "source.json")]
    if any(path.exists() for path in targets):
        raise FileExistsError("Prepared HotelRec files already exist; use another output folder")
    temporary = targets[0].with_suffix(".tmp")
    digest, hotels, skipped, written, scanned = hashlib.sha256(), {}, 0, 0, 0
    try:
        with gzip.open(temporary, "wt", encoding="utf8", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["user_id", "item_id", "rating", "timestamp"])
            for scanned, line in enumerate(itertools.islice(zip_lines(stream), max_reviews), 1):
                digest.update(line + b"\n")
                parsed = parse_review(json.loads(line))
                if parsed is None:
                    skipped += 1
                    continue
                author, item_id, rating, timestamp, description = parsed
                hotels[item_id] = description
                writer.writerow([author, item_id, rating, timestamp])
                written += 1
                if scanned % 100000 == 0:
                    print(f"Scanned {scanned:,} reviews; kept {written:,}; hotels {len(hotels):,}", flush=True)
        if scanned != max_reviews or not written:
            raise ValueError(f"Expected {max_reviews} reviews; scanned {scanned}, kept {written}")
        with gzip.open(targets[1], "wt", encoding="utf8", newline="") as out:
            writer = csv.writer(out)
            writer.writerow(["item_id", "description"])
            writer.writerows(sorted(hotels.items()))
        temporary.replace(targets[0])
        checksums = {}
        for path in targets[:2]:
            with path.open("rb") as handle:
                checksums[path.name] = hashlib.file_digest(handle, "sha256").hexdigest()
        meta = {
            "source_url": SOURCE_URL, "source_repository": "https://github.com/diegoantognini/HotelRec",
            "sampling": "First N JSONL records in HotelRec.txt; hotel-ordered prefix, not random",
            "reviews_scanned": scanned, "reviews_kept": written, "invalid_reviews_skipped": skipped,
            "hotels": len(hotels), "raw_lines_sha256": digest.hexdigest(),
            "prepared_sha256": checksums,
            "content": "Hotel name/location from URL only; no review text or ratings",
            "demographics": "Unavailable; unknown fallback only",
            "integrity": "Prefix SHA256 recorded; full archive CRC/checksum not verified",
            "usage": "Academic research only",
        }
        targets[2].write_text(json.dumps(meta, indent=2), encoding="utf8")
        print(json.dumps(meta, indent=2), flush=True)
        return meta
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Existing full ZIP or downloaded ZIP prefix")
    parser.add_argument("--max-reviews", type=int, default=1000000)
    parser.add_argument("--output", type=Path, default=Path("datasets_raw/hotelrec"))
    args = parser.parse_args()
    with args.archive.open("rb") if args.archive else urlopen(SOURCE_URL, timeout=60) as source:
        prepare(source, args.output, args.max_reviews)
