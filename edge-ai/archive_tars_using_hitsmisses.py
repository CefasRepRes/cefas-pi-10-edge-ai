from gui.application_validation.azure_utils import get_blob_service_client
from azure.storage.blob import StandardBlobTier
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
import csv
import io
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

ACCOUNT_URL = "https://citprodc8603uksa.blob.core.windows.net"
CONTAINERS = [
    "cend06-25-herring-south-northsea-spring",
    "cend-10-25-megsneps-earlysummer-northsea",
    "cend-11-25-csemp-summer-northsea",
    "cend-12-25-dbts-summer-northsea",
    "cend05-25-ecos-celticsea-winter",
    "cend08-25-scallop-echannel-springsummer",
    "cend09-25-mpa-echannel-earlysummer",
    "cend15-25-nsgf-northsea-summer",
    "cend16-25-abts-irishsea-bchannel-autumn",
    "cend17-24-peltic-autumn-celticsea",
    "cend17-25-peltic-autumn",
]



# Don't do these ones as they aren't tarred:
#cend08-24-nephrops-summer-northsea
#cend10-24-meggs-summer-northsea
#cend12-23-mpa-autumn

# Start small and automatically widen the tail read if HitsMisses.txt is not found.
TAIL_BYTE_ATTEMPTS = [128 * 1024, 512 * 1024, 2 * 1024 * 1024, 8 * 1024 * 1024, 32 * 1024 * 1024]
TARGET_FILE_NAMES = ("HitsMisses.txt",)
TEN_MINUTE_TAR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}/\d{4}\.tar$", re.IGNORECASE)
DRY_RUN = True  # Set False when you are happy with the printed decisions.
ARCHIVE_WHEN_ANY_MISS_GT_ZERO = True
REPORT_CSV = "hitsmisses_archive_decisions.csv"


def _tar_checksum_ok(header: bytes) -> bool:
    if len(header) != 512:
        return False
    if header == b"\0" * 512:
        return False
    raw_name = header[:100].split(b"\0", 1)[0]
    if not raw_name:
        return False
    stored = header[148:156].decode("ascii", errors="ignore").strip("\0 ")
    if not stored:
        return False
    try:
        stored_sum = int(stored, 8)
    except ValueError:
        return False
    unsigned = bytearray(header)
    unsigned[148:156] = b"        "
    return sum(unsigned) == stored_sum


def _tar_member_size(header: bytes) -> int:
    size_text = header[124:136].decode("ascii", errors="ignore").strip("\0 ")
    try:
        return int(size_text, 8)
    except ValueError:
        return 0


def extract_files_from_tar_tail(tar_tail_bytes: bytes, wanted_files: Iterable[str] = TARGET_FILE_NAMES) -> Dict[str, bytes]:
    """Extract target members from a block-aligned tail section of an uncompressed TAR.

    The previous version assumed byte zero of the downloaded tail was a TAR header.
    That is usually not true. This version walks block-by-block and only trusts
    blocks that pass the TAR header checksum, so it can recover target files from
    the tail even when the first blocks are payload or padding from an earlier file.
    """
    wanted_files = tuple(wanted_files)
    block = 512
    results: Dict[str, bytes] = {}
    i = 0
    n = len(tar_tail_bytes)
    while i + block <= n:
        header = tar_tail_bytes[i:i + block]
        if not _tar_checksum_ok(header):
            i += block
            continue
        name = header[:100].split(b"\0", 1)[0].decode("utf-8", errors="ignore")
        prefix = header[345:500].split(b"\0", 1)[0].decode("utf-8", errors="ignore")
        if prefix:
            name = f"{prefix}/{name}"
        size = _tar_member_size(header)
        start = i + block
        end = start + size
        padded_end = start + ((size + block - 1) // block) * block
        if end <= n and any(name.endswith(wanted) for wanted in wanted_files):
            results[name] = tar_tail_bytes[start:end]
        i = padded_end if padded_end > i else i + block
    return results


def download_tail_aligned(blob_client, blob_size: int, requested_tail_bytes: int) -> bytes:
    """Download a tail range whose start is aligned to a 512-byte TAR block boundary."""
    requested_tail_bytes = int(requested_tail_bytes)
    raw_offset = max(0, blob_size - requested_tail_bytes)
    aligned_offset = raw_offset - (raw_offset % 512)
    length = blob_size - aligned_offset
    return blob_client.download_blob(offset=aligned_offset, length=length, max_concurrency=1).readall()


def grab_hitsmisses_from_tar_tail(blob_client, blob_size: int, tail_attempts: Iterable[int] = TAIL_BYTE_ATTEMPTS) -> Tuple[Optional[str], Optional[bytes], int]:
    """Try increasingly large tail reads and return the first HitsMisses.txt found."""
    last_tail = 0
    for tail_bytes in tail_attempts:
        last_tail = min(int(tail_bytes), int(blob_size))
        tail = download_tail_aligned(blob_client, blob_size, last_tail)
        files = extract_files_from_tar_tail(tail, TARGET_FILE_NAMES)
        for name, content in files.items():
            if name.endswith("HitsMisses.txt"):
                return name, content, last_tail
    return None, None, last_tail


def parse_hitsmisses(content: bytes) -> Tuple[List[Tuple[int, int]], int, int, int]:
    """Return rows, total hits, total misses and number of rows with misses > 0."""
    text = content.decode("utf-8-sig", errors="replace")
    rows: List[Tuple[int, int]] = []
    reader = csv.reader(io.StringIO(text))
    for raw in reader:
        if len(raw) < 2:
            continue
        try:
            hit = int(float(raw[0].strip()))
            miss = int(float(raw[1].strip()))
        except ValueError:
            continue
        rows.append((hit, miss))
    total_hits = sum(hit for hit, _miss in rows)
    total_misses = sum(miss for _hit, miss in rows)
    rows_with_misses = sum(1 for _hit, miss in rows if miss > 0)
    return rows, total_hits, total_misses, rows_with_misses


def should_archive_from_hitsmisses(rows: List[Tuple[int, int]]) -> Tuple[bool, str]:
    if not rows:
        return False, "no_parseable_hitsmisses_rows"
    if ARCHIVE_WHEN_ANY_MISS_GT_ZERO:
        if any(miss > 0 for _hit, miss in rows):
            return True, "archive_any_miss_gt_zero"
        return False, "skip_all_misses_zero"
    total_misses = sum(miss for _hit, miss in rows)
    if total_misses > 0:
        return True, "archive_total_misses_gt_zero"
    return False, "skip_total_misses_zero"


def set_archive_tier(blob_client, dry_run: bool = DRY_RUN) -> str:
    if dry_run:
        return "dry_run_not_archived"
    blob_client.set_standard_blob_tier(StandardBlobTier.ARCHIVE)
    return "archived"


def iter_candidate_tar_blobs(container_client, prefix: str = ""):
    """Yield ten-minute TAR blobs like YYYY-MM-DD/HHMM.tar."""
    for blob in container_client.list_blobs(name_starts_with=prefix or None):
        name = getattr(blob, "name", "")
        if TEN_MINUTE_TAR_RE.match(name):
            yield blob


def archive_tars_using_hitsmisses(
    containers: Iterable[str] = CONTAINERS,
    account_url: str = ACCOUNT_URL,
    prefix: str = "",
    dry_run: bool = DRY_RUN,
    report_csv: str = REPORT_CSV,
):
    """Browser-authenticate once, scan TAR tails, and archive TARs only when HitsMisses says to.

    For each container/date/ten-minute TAR:
      1. stream only the TAR tail;
      2. recover HitsMisses.txt if present near the end;
      3. parse hit,miss rows;
      4. archive the TAR blob only if at least one miss value is > 0.

    A HitsMisses.txt with ten rows such as '1858,0' ... '2096,0' will therefore be skipped.
    """
    bsc = get_blob_service_client(account_url)  # uses your application_validation browser auth flow/cache
    decisions = []
    fieldnames = [
        "container", "blob_name", "blob_size", "hitsmisses_member", "tail_bytes_used",
        "rows", "total_hits", "total_misses", "rows_with_misses", "archive_decision",
        "reason", "tier_action", "error",
    ]
    with open(report_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for container in containers:
            print(f"\nContainer: {container}")
            container_client = bsc.get_container_client(container)
            for blob in iter_candidate_tar_blobs(container_client, prefix=prefix):
                blob_name = blob.name
                blob_size = int(getattr(blob, "size", 0) or 0)
                row = {
                    "container": container,
                    "blob_name": blob_name,
                    "blob_size": blob_size,
                    "hitsmisses_member": "",
                    "tail_bytes_used": "",
                    "rows": 0,
                    "total_hits": 0,
                    "total_misses": 0,
                    "rows_with_misses": 0,
                    "archive_decision": False,
                    "reason": "",
                    "tier_action": "",
                    "error": "",
                }
                try:
                    blob_client = container_client.get_blob_client(blob_name)
                    member_name, content, tail_used = grab_hitsmisses_from_tar_tail(blob_client, blob_size)
                    row["tail_bytes_used"] = tail_used
                    if content is None:
                        row["reason"] = "no_hitsmisses_found_in_tail"
                        row["tier_action"] = "skipped"
                        print(f"  SKIP no HitsMisses in tail: {blob_name}")
                    else:
                        rows, total_hits, total_misses, rows_with_misses = parse_hitsmisses(content)
                        archive, reason = should_archive_from_hitsmisses(rows)
                        row.update({
                            "hitsmisses_member": member_name,
                            "rows": len(rows),
                            "total_hits": total_hits,
                            "total_misses": total_misses,
                            "rows_with_misses": rows_with_misses,
                            "archive_decision": archive,
                            "reason": reason,
                        })
                        if archive:
                            action = set_archive_tier(blob_client, dry_run=dry_run)
                            row["tier_action"] = action
                            print(f"  ARCHIVE {action}: {blob_name} rows={len(rows)} total_misses={total_misses}")
                        else:
                            row["tier_action"] = "skipped"
                            print(f"  SKIP {reason}: {blob_name} rows={len(rows)} total_misses={total_misses}")
                except (ResourceNotFoundError, HttpResponseError, Exception) as e:
                    row["reason"] = "error"
                    row["tier_action"] = "failed"
                    row["error"] = f"{type(e).__name__}: {e}"
                    print(f"  FAILED {blob_name}: {row['error']}")
                writer.writerow(row)
                f.flush()
                decisions.append(row)
    print(f"\nWrote decision report: {os.path.abspath(report_csv)}")
    if dry_run:
        print("DRY_RUN is True, so no blobs were actually archived. Set DRY_RUN = False to apply Archive tier.")
    return decisions


if __name__ == "__main__":
    archive_tars_using_hitsmisses(
        containers=CONTAINERS,
        account_url=ACCOUNT_URL,
        dry_run=DRY_RUN,
    )
