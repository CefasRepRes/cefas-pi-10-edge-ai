import hashlib
import os
import re
import tarfile
from typing import Dict, Iterable, List, Optional, Tuple


def _strip_tar_extension(name: str) -> str:
    """Return name without a TAR archive suffix, preserving any parent path."""
    for ext in (".tar.gz", ".tgz", ".tar"):
        if name.lower().endswith(ext):
            return name[:-len(ext)]
    return name


def _safe_path_segment(segment: str) -> str:
    """Make one blob path segment safe to use as a local directory or filename."""
    segment = str(segment or "").strip()
    segment = re.sub(r"[^A-Za-z0-9._-]+", "_", segment)
    segment = segment.strip("._")
    return segment or "unknown"


def safe_stem_from_tar_name(tar_blob_name: str) -> str:
    """Return the safe basename stem for a TAR blob.

    This is intentionally kept as a basename-only helper because it is used in
    places that need a single filename component. For output folders that must
    distinguish same-named TARs in different date prefixes, use
    safe_relative_stem_from_tar_name().
    """
    base = os.path.basename(str(tar_blob_name).replace("\\", "/"))
    return _safe_path_segment(_strip_tar_extension(base))


def safe_relative_stem_from_tar_name(tar_blob_name: str) -> str:
    """Return a safe relative path for a TAR blob, preserving date/prefix folders.

    Example:
        2024-10-13/1700.tar -> 2024-10-13/1700

    This prevents outputs for blobs such as 2024-10-13/1700.tar and
    2024-10-14/1700.tar from both being written to per_tar/1700.
    """
    normalised = str(tar_blob_name or "").replace("\\", "/").strip("/")
    no_ext = _strip_tar_extension(normalised)
    parts = [_safe_path_segment(part) for part in no_ext.split("/") if part not in ("", ".", "..")]
    if not parts:
        parts = [safe_stem_from_tar_name(tar_blob_name)]
    return os.path.join(*parts)


def safe_file_id_from_tar_name(tar_blob_name: str) -> str:
    """Return a single safe filename identifier that includes the blob prefix."""
    return "__".join(safe_relative_stem_from_tar_name(tar_blob_name).split(os.sep))


def per_tar_paths(run_dir: str, tar_blob_name: str):
    relative_stem = safe_relative_stem_from_tar_name(tar_blob_name)
    out_dir = os.path.join(run_dir, "per_tar", relative_stem)
    os.makedirs(out_dir, exist_ok=True)
    return (
        out_dir,
        os.path.join(out_dir, "predictions.json"),
        os.path.join(out_dir, "summary.json"),
    )


def iter_per_tar_prediction_paths(per_tar_dir: str):
    """Yield predictions.json files under a per-TAR output tree, including nested folders."""
    if not per_tar_dir or not os.path.isdir(per_tar_dir):
        return []

    matches = []
    for root, dirs, files in os.walk(per_tar_dir):
        dirs.sort()
        files.sort()
        if "predictions.json" in files:
            matches.append(os.path.join(root, "predictions.json"))
    return matches


def tar_member_sample_rank(tar_blob_name: str, member_name: str) -> int:
    """Stable per-member rank used for deterministic within-TAR sampling."""
    digest = hashlib.sha256(f"{tar_blob_name}\n{member_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sample_ratio_threshold(sample_ratio: float) -> int:
    """Convert a ratio in [0, 1] to an unsigned 64-bit sampling threshold."""
    if sample_ratio <= 0.0:
        return -1
    if sample_ratio >= 1.0:
        return (1 << 64) - 1
    return int(sample_ratio * ((1 << 64) - 1))


def should_sample_tar_member(
    tar_blob_name: str,
    member_name: str,
    sample_ratio: float,
    sample_rank: int = None,
    sample_threshold: int = None,
) -> bool:
    """Return True when the TAR member should be kept for the requested sample ratio."""
    if sample_ratio <= 0.0:
        return False
    if sample_ratio >= 1.0:
        return True
    sample_rank = tar_member_sample_rank(tar_blob_name, member_name) if sample_rank is None else sample_rank
    sample_threshold = sample_ratio_threshold(sample_ratio) if sample_threshold is None else sample_threshold
    return sample_rank <= sample_threshold


class _BlobChunkStream:
    """Small read() adapter around Azure StorageStreamDownloader.chunks()."""

    def __init__(self, downloader):
        self._chunks = iter(downloader.chunks())
        self._buffer = bytearray()
        self._eof = False
        self.bytes_read = 0

    def readable(self):
        return True

    def read(self, size=-1):
        if size is None or size < 0:
            out = [bytes(self._buffer)]
            self._buffer.clear()
            for chunk in self._chunks:
                out.append(chunk)
            self._eof = True
            return b"".join(out)

        while len(self._buffer) < size and not self._eof:
            try:
                self._buffer.extend(next(self._chunks))
            except StopIteration:
                self._eof = True
                break

        out = bytes(self._buffer[:size])
        del self._buffer[:size]
        self.bytes_read += len(out)
        return out

    def close(self):
        """Mark the stream as closed so downstream code can stop reading promptly."""
        self._eof = True
        self._buffer.clear()
        self._chunks = iter(())


def _align_tar_offset(offset: int, block_size: int = 512) -> int:
    """Align an offset to a TAR block boundary so segmented reads start at member headers."""
    offset = int(offset or 0)
    if offset <= 0:
        return 0
    if block_size <= 0:
        return offset
    return offset - (offset % block_size)


def build_tar_stream_segments(blob_size: int, target_bytes: int, segment_count: int = 5) -> List[Tuple[int, int]]:
    """Build evenly spaced TAR stream segments as (offset, length) pairs."""
    blob_size = int(blob_size or 0)
    target_bytes = int(target_bytes or 0)
    segment_count = max(1, int(segment_count or 1))
    if blob_size <= 0 or target_bytes <= 0:
        return []

    segments: List[Tuple[int, int]] = []
    remaining = target_bytes
    for index in range(segment_count):
        if remaining <= 0:
            segments.append((0, 0))
            continue

        offset = int(blob_size * index / segment_count)
        offset = _align_tar_offset(offset)
        if index == segment_count - 1:
            length = remaining
        else:
            length = max(1, remaining // (segment_count - index))
            if length > remaining:
                length = remaining

        if offset >= blob_size:
            segments.append((offset, 0))
            continue

        available = max(0, blob_size - offset)
        if length > available:
            length = available
        if length <= 0:
            segments.append((offset, 0))
            continue

        segments.append((offset, length))
        remaining -= length

    return segments


def _tar_checksum_ok(header: bytes) -> bool:
    """Return True when a 512-byte block looks like a valid TAR header."""
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
    """Parse the size field from a TAR header block."""
    size_text = header[124:136].decode("ascii", errors="ignore").strip("\0 ")
    try:
        return int(size_text, 8)
    except ValueError:
        return 0


def _find_valid_tar_header_offset(blob_client, start_offset: int, scan_bytes: int) -> Tuple[int, Optional[bytes]]:
    """Return the next checksum-valid TAR header offset after the requested start."""
    offset = max(0, int(start_offset or 0))
    scan_length = max(512 * 2, int(scan_bytes or 0))
    if scan_length <= 0:
        return offset, None

    aligned_offset = offset - (offset % 512)
    scan_length = max(scan_length, 512 * 2)
    try:
        downloader = blob_client.download_blob(offset=aligned_offset, length=scan_length, max_concurrency=1)
        payload = downloader.readall()
    except Exception:
        return offset, None

    if not payload:
        return offset, None

    start_index = max(0, offset - aligned_offset)
    for index in range(start_index, len(payload) - 511, 512):
        block = payload[index:index + 512]
        if _tar_checksum_ok(block):
            return aligned_offset + index, block

    return offset, None


def iter_tar_members_from_blob(
    blob_client,
    tar_blob_name: str,
    log_cb=None,
    max_bytes: int = None,
    offset: int = 0,
):
    """Yield tarfile members and member file objects directly from Azure Blob Storage.

    When a byte offset is supplied we first scan forward from that point for the
    next checksum-valid TAR header block, so we do not start reading from the
    middle of a member payload or padding.
    """
    offset = int(offset or 0)
    if log_cb:
        stream_note = f" (offset {offset:,}, first {max_bytes:,} bytes)" if max_bytes is not None else ""
        if offset:
            stream_note = f" (offset {offset:,})" if max_bytes is None else stream_note
        log_cb(f"Streaming TAR directly from blob store: {tar_blob_name}{stream_note}")

    download_kwargs = {"max_concurrency": 1}
    if max_bytes is not None:
        requested_length = max(1, int(max_bytes))
    else:
        requested_length = None

    resolved_offset = offset
    resolved_header = None
    if offset:
        scan_length = max(512 * 8, (requested_length or 0) + 512 * 16)
        resolved_offset, resolved_header = _find_valid_tar_header_offset(blob_client, offset, scan_length)
        if resolved_header is not None and resolved_offset != offset:
            if log_cb:
                log_cb(
                    f"Aligned TAR stream start to checksum-valid header at byte {resolved_offset:,} "
                    f"(requested {offset:,})"
                )

    if max_bytes is not None:
        download_length = requested_length
        if resolved_header is not None:
            member_size = _tar_member_size(resolved_header)
            if member_size > 0:
                padded_member_size = member_size + ((512 - (member_size % 512)) % 512)
                download_length = max(download_length, 512 + padded_member_size)
        download_kwargs.update({"offset": resolved_offset, "length": max(1, int(download_length))})
    elif resolved_offset != offset:
        download_kwargs.update({"offset": resolved_offset})
    elif offset:
        download_kwargs.update({"offset": offset})

    try:
        downloader = blob_client.download_blob(**download_kwargs)
    except Exception as exc:
        if log_cb:
            log_cb(f"Failed to open TAR stream at offset {resolved_offset:,}: {exc}")
        return

    stream = _BlobChunkStream(downloader)

    try:
        with tarfile.open(fileobj=stream, mode="r|*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                fh = tf.extractfile(member)
                if fh is None:
                    continue
                yield member, fh
    except tarfile.ReadError as e:
        if max_bytes is None:
            raise
        if log_cb:
            log_cb(
                f"Stopped TAR stream at byte limit ({max_bytes:,}) for {tar_blob_name} "
                f"after reading {stream.bytes_read:,} bytes ({type(e).__name__})"
            )
    finally:
        try:
            stream.close()
        except Exception:
            pass
        try:
            downloader.close()
        except Exception:
            pass


def iter_tar_image_bytes_from_blob(
    blob_client,
    tar_blob_name: str,
    log_cb=None,
    member_filter=None,
    max_bytes: int = None,
    offset: int = 0,
):
    """Yield (member_name, image_bytes) for image members streamed directly from a blob TAR."""
    image_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".webp")

    for member, extracted in iter_tar_members_from_blob(
        blob_client,
        tar_blob_name,
        log_cb=log_cb,
        max_bytes=max_bytes,
        offset=offset,
    ):
        member_name = member.name
        if not member_name.lower().endswith(image_exts):
            continue
        if member_filter is not None and not member_filter(member_name):
            continue
        try:
            yield member_name, extracted.read()
        except Exception as e:
            if (
                max_bytes is not None
                and (
                    "unexpected end of data" in str(e).lower()
                    or type(e).__name__ == "ReadError"
                )
            ):
                # expected when intentionally truncating a TAR
                return
            if log_cb:
                log_cb(
                    f"Skipping image member read in TAR "
                    f"({type(e).__name__}): {member_name} - {e}"
                )
