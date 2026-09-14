from gui.application_validation.azure_utils import get_blob_service_client
from azure.storage.blob import ContentSettings
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, HttpResponseError, ServiceRequestError, ServiceResponseError
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import datetime, timedelta
from io import BytesIO
import csv
import logging
import os
import re
import tarfile
import tempfile
import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

ACCOUNT_URL = "https://citprodc8603uksa.blob.core.windows.net"
SOURCE_CONTAINERS = [
    "databox-2483495b-0221-43b8-584a-444434c92e9c",
]

# Inclusive date range used to probe for HitsMisses.txt files without scanning the whole container.
SURVEY_START_DATE = "2023-10-28"
SURVEY_END_DATE = "2023-10-30"
DRY_RUN = False
OVERWRITE_EXISTING_TARS = False
OUTPUT_TAR_AT_DESTINATION_ROOT = True
REPORT_CSV = "tiny_files_fast_hitmisses_report.csv"
LOG_FILE = "tiny_files_fast_hitmisses.log"
HITSMISSES_BASENAME = "HitsMisses.txt"
RAW_IMAGES_DIR = "RawImages"
IMAGE_STEM = "pia1"
TIF_DIGITS = 8
# Probe every possible minute folder, not just 10-minute boundaries. This avoids a full blob listing while still catching offset tenbins such as 1345.
PROBE_EVERY_MINUTE = False
HITSMISSES_PROBE_WORKERS = 4
TENBIN_WORKERS = 1
TIF_DOWNLOAD_WORKERS = 8
MAX_PENDING_TIF_DOWNLOADS = TIF_DOWNLOAD_WORKERS * 4
DOWNLOAD_MAX_CONCURRENCY_PER_BLOB = 1
UPLOAD_MAX_CONCURRENCY = 8
PROGRESS_EVERY_N_TIFS = 100
MAX_RETRIES = 8
RETRY_SLEEP_SECONDS = 1
TAR_CONTENT_TYPE = "application/x-tar"


def setup_logging(log_file: str = LOG_FILE) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"), logging.StreamHandler()],
    )
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
    logging.getLogger("azure.identity").setLevel(logging.WARNING)
    logging.getLogger("msal").setLevel(logging.WARNING)
    logging.info("RUN_LOG file=%s", os.path.abspath(log_file))


def destination_container_name(source_container: str) -> str:
    name = source_container.lower()
    name = f"{name[:4]}-{name[4:]}"
    name = re.sub(r"[^a-z0-9-]", "-", name)
    name = re.sub(r"-+", "-", name).strip("-")
    if len(name) < 3 or len(name) > 63:
        raise ValueError(f"Destination container name is invalid after inserting hyphen: {name}")
    return name


def ensure_destination_container(bsc, dest_container_name: str, dry_run: bool):
    dest = bsc.get_container_client(dest_container_name)
    if dry_run:
        logging.info("DEST_CONTAINER dry_run_not_created name=%s", dest_container_name)
        return dest, "dry_run_container_not_created"
    try:
        bsc.create_container(dest_container_name)
        logging.info("DEST_CONTAINER created name=%s", dest_container_name)
        return dest, "created"
    except ResourceExistsError:
        logging.info("DEST_CONTAINER exists name=%s", dest_container_name)
        return dest, "already_exists"


def blob_exists(container_client, blob_name: str) -> bool:
    try:
        container_client.get_blob_client(blob_name).get_blob_properties()
        return True
    except ResourceNotFoundError:
        return False


def parse_date(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d")


def iter_candidate_bins(start_date: str, end_date: str, every_minute: bool = True) -> Iterator[Tuple[str, str]]:
    start = parse_date(start_date)
    end = parse_date(end_date)
    step = timedelta(minutes=1 if every_minute else 10)
    current = start
    final_exclusive = end + timedelta(days=1)
    while current < final_exclusive:
        yield current.strftime("%Y-%m-%d"), current.strftime("%H%M")
        current += step


def output_tar_name(root: str, date_text: str, bin_text: str) -> str:
    clean_root = root.strip("/")
    if OUTPUT_TAR_AT_DESTINATION_ROOT or not clean_root:
        return f"{date_text}/{bin_text}.tar"
    return f"{clean_root}/{date_text}/{bin_text}.tar"


def hitsmisses_blob_name(root: str, date_text: str, bin_text: str) -> str:
    clean_root = root.strip("/")
    if clean_root:
        return f"{clean_root}/{date_text}/{bin_text}/{HITSMISSES_BASENAME}"
    return f"{date_text}/{bin_text}/{HITSMISSES_BASENAME}"


def tif_blob_name(root: str, date_text: str, bin_text: str, index: int) -> str:
    name = f"{IMAGE_STEM}.{date_text}.{bin_text}+N{index:0{TIF_DIGITS}d}.tif"
    clean_root = root.strip("/")
    if clean_root:
        return f"{clean_root}/{date_text}/{bin_text}/{RAW_IMAGES_DIR}/{name}"
    return f"{date_text}/{bin_text}/{RAW_IMAGES_DIR}/{name}"


def tif_member_name(date_text: str, bin_text: str, index: int) -> str:
    return f"{RAW_IMAGES_DIR}/{IMAGE_STEM}.{date_text}.{bin_text}+N{index:0{TIF_DIGITS}d}.tif"


def retry_call(label: str, func):
    last_error = None
    retryable_status_codes = {408, 429, 500, 502, 503, 504}

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func()

        except ResourceNotFoundError:
            # Missing blobs are expected during HitsMisses probing.
            # They should not be retried or logged as warnings.
            raise

        except (ServiceRequestError, ServiceResponseError, TimeoutError, OSError) as exc:
            last_error = exc

        except HttpResponseError as exc:
            status_code = getattr(exc, "status_code", None)

            if status_code == 404:
                raise

            if status_code not in retryable_status_codes:
                raise

            last_error = exc

        if attempt < MAX_RETRIES:
            sleep_for = RETRY_SLEEP_SECONDS * attempt

            logging.warning(
                "RETRY label=%s attempt=%s/%s error=%r sleep_s=%.2f",
                label,
                attempt,
                MAX_RETRIES,
                last_error,
                sleep_for,
            )
            time.sleep(sleep_for)

    if last_error is None:
        raise RuntimeError(
            f"Operation failed without capturing an exception: {label}"
        )

    raise last_error

def download_blob_bytes(
    container_client,
    blob_name: str,
    max_concurrency: int = DOWNLOAD_MAX_CONCURRENCY_PER_BLOB,
) -> bytes:
    blob_client = container_client.get_blob_client(blob_name)

    def do_download() -> bytes:
        downloader = blob_client.download_blob(
            max_concurrency=max_concurrency,
        )
        return downloader.readall()

    return retry_call(
        label=f"download:{blob_name}",
        func=do_download,
    )


def parse_hitsmisses_bytes(data: bytes) -> Tuple[List[int], int]:
    text = data.decode("utf-8-sig", errors="replace")
    counts: List[int] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        first = line.split(",", 1)[0].strip()
        if not first:
            continue
        counts.append(int(first))
    return counts, sum(counts)


def probe_one_hitsmisses(source_cc, root: str, date_text: str, bin_text: str) -> Optional[Dict[str, object]]:
    blob_name = hitsmisses_blob_name(root, date_text, bin_text)
    try:
        data = download_blob_bytes(source_cc, blob_name, max_concurrency=1)
    except ResourceNotFoundError:
        return None
    counts, total = parse_hitsmisses_bytes(data)
    return {
        "root": root,
        "date_text": date_text,
        "bin_text": bin_text,
        "hitsmisses_blob": blob_name,
        "hitsmisses_bytes": data,
        "hitsmisses_counts": counts,
        "expected_tifs": total,
    }


def discover_tenbins_from_hitsmisses(source_cc, root: str, start_date: str, end_date: str) -> List[Dict[str, object]]:
    candidates = list(iter_candidate_bins(start_date, end_date, every_minute=PROBE_EVERY_MINUTE))
    logging.info("HITSMISSES_PROBE_START root=%s candidates=%s workers=%s every_minute=%s", root or "<root>", len(candidates), HITSMISSES_PROBE_WORKERS, PROBE_EVERY_MINUTE)
    found: List[Dict[str, object]] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=HITSMISSES_PROBE_WORKERS) as pool:
        futures = [pool.submit(probe_one_hitsmisses, source_cc, root, d, b) for d, b in candidates]
        for i, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            if result is not None:
                found.append(result)
                logging.info(
                    "HITSMISSES_FOUND blob=%s expected_tifs=%s",
                    result["hitsmisses_blob"],
                    result["expected_tifs"],
                )
            if i % 5000 == 0:
                logging.info("HITSMISSES_PROBE_PROGRESS checked=%s/%s found=%s", i, len(candidates), len(found))
    found.sort(key=lambda x: (str(x["date_text"]), str(x["bin_text"])))
    logging.info("HITSMISSES_PROBE_END found=%s elapsed_s=%.1f", len(found), time.time() - started)
    return found


def add_bytes_to_tar(tar: tarfile.TarFile, member_name: str, data: bytes) -> None:
    info = tarfile.TarInfo(member_name)
    info.size = len(data)
    info.mtime = int(time.time())
    tar.addfile(info, BytesIO(data))


def download_one_tif(source_cc, root: str, date_text: str, bin_text: str, index: int) -> Tuple[int, str, bytes]:
    blob_name = tif_blob_name(root, date_text, bin_text, index)
    data = download_blob_bytes(source_cc, blob_name, max_concurrency=DOWNLOAD_MAX_CONCURRENCY_PER_BLOB)
    return index, blob_name, data


def write_tenbin_tar_to_local_temp(source_cc, tenbin: Dict[str, object], temp_dir: str) -> Dict[str, object]:
    root = str(tenbin["root"])
    date_text = str(tenbin["date_text"])
    bin_text = str(tenbin["bin_text"])
    expected_tifs = int(tenbin["expected_tifs"])
    hitsmisses_data = bytes(tenbin["hitsmisses_bytes"])
    hitsmisses_blob = str(tenbin["hitsmisses_blob"])
    out_tar = output_tar_name(root, date_text, bin_text)
    safe_tar_name = out_tar.replace("/", "_")
    local_tar_path = os.path.join(temp_dir, safe_tar_name)
    started = time.time()
    source_payload_bytes = 0
    tif_count = 0
    failed_downloads: List[str] = []
    logging.info("TENBIN_START date=%s bin=%s expected_tifs=%s out_tar=%s", date_text, bin_text, expected_tifs, out_tar)
    if expected_tifs <= 0:
        with tarfile.open(local_tar_path, mode="w") as tar:
            with ThreadPoolExecutor(max_workers=TIF_DOWNLOAD_WORKERS) as pool:
                next_index = 0
                pending = set()

                initial_submission_count = min(
                    MAX_PENDING_TIF_DOWNLOADS,
                    expected_tifs,
                )

                while (
                    next_index < expected_tifs
                    and len(pending) < initial_submission_count
                ):
                    pending.add(
                        pool.submit(
                            download_one_tif,
                            source_cc,
                            root,
                            date_text,
                            bin_text,
                            next_index,
                        )
                    )
                    next_index += 1

                while pending:
                    done, pending = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )

                    for fut in done:
                        try:
                            index, blob_name, data = fut.result()

                        except Exception as exc:
                            failed_downloads.append(
                                f"{type(exc).__name__}: {exc}"
                            )

                            if len(failed_downloads) <= 20:
                                logging.exception(
                                    "TIF_DOWNLOAD_FAILED date=%s bin=%s",
                                    date_text,
                                    bin_text,
                                )

                        else:
                            add_bytes_to_tar(
                                tar,
                                tif_member_name(date_text, bin_text, index),
                                data,
                            )

                            tif_count += 1
                            source_payload_bytes += len(data)

                            if tif_count % PROGRESS_EVERY_N_TIFS == 0:
                                logging.info(
                                    "TIF_PROGRESS date=%s bin=%s "
                                    "downloaded=%s/%s",
                                    date_text,
                                    bin_text,
                                    tif_count,
                                    expected_tifs,
                                )

                    # Replenish the bounded queue regardless of whether completed
                    # futures succeeded or failed.
                    while (
                        next_index < expected_tifs
                        and len(pending) < MAX_PENDING_TIF_DOWNLOADS
                    ):
                        pending.add(
                            pool.submit(
                                download_one_tif,
                                source_cc,
                                root,
                                date_text,
                                bin_text,
                                next_index,
                            )
                        )
                        next_index += 1

            add_bytes_to_tar(
                tar,
                HITSMISSES_BASENAME,
                hitsmisses_data,
            )        
        return {
            "source_prefix": f"{root.strip('/') + '/' if root.strip('/') else ''}{date_text}/{bin_text}/",
            "output_tar": out_tar,
            "decision": "created",
            "reason": "zero_tifs_hitsmisses_only",
            "file_count": 1,
            "tif_count": 0,
            "source_payload_bytes": len(hitsmisses_data),
            "tar_file_size": os.path.getsize(local_tar_path),
            "hitsmisses_blob": hitsmisses_blob,
            "hitsmisses_member_last": True,
            "local_tar_path": local_tar_path,
            "error": "",
        }
    with tarfile.open(local_tar_path, mode="w") as tar:
        with ThreadPoolExecutor(max_workers=TIF_DOWNLOAD_WORKERS) as pool:
            next_index = 0
            pending = set()
            while next_index < expected_tifs and len(pending) < min(MAX_PENDING_TIF_DOWNLOADS, expected_tifs):
                pending.add(pool.submit(download_one_tif, source_cc, root, date_text, bin_text, next_index))
                next_index += 1
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    try:
                        index, blob_name, data = fut.result()
                    except Exception as e:
                        failed_downloads.append(f"{type(e).__name__}: {e}")
                        if len(failed_downloads) <= 20:
                            logging.exception("TIF_DOWNLOAD_FAILED date=%s bin=%s", date_text, bin_text)
                        continue
                    add_bytes_to_tar(tar, tif_member_name(date_text, bin_text, index), data)
                    tif_count += 1
                    source_payload_bytes += len(data)
                    if tif_count % PROGRESS_EVERY_N_TIFS == 0:
                        logging.info("TIF_PROGRESS date=%s bin=%s downloaded=%s/%s", date_text, bin_text, tif_count, expected_tifs)
                    while next_index < expected_tifs and len(pending) < MAX_PENDING_TIF_DOWNLOADS:
                        pending.add(pool.submit(download_one_tif, source_cc, root, date_text, bin_text, next_index))
                        next_index += 1
        add_bytes_to_tar(tar, HITSMISSES_BASENAME, hitsmisses_data)
    tar_file_size = os.path.getsize(local_tar_path)
    elapsed = time.time() - started
    if failed_downloads:
        decision = "failed"
        reason = f"missing_or_failed_tifs={len(failed_downloads)}"
    elif tif_count != expected_tifs:
        decision = "failed"
        reason = f"tif_count_mismatch_expected_{expected_tifs}_got_{tif_count}"
    else:
        decision = "created"
        reason = "ok"
    logging.info(
        "TENBIN_TAR_DONE date=%s bin=%s decision=%s tifs=%s/%s tar_bytes=%s elapsed_s=%.1f",
        date_text,
        bin_text,
        decision,
        tif_count,
        expected_tifs,
        tar_file_size,
        elapsed,
    )
    return {
        "source_prefix": f"{root.strip('/') + '/' if root.strip('/') else ''}{date_text}/{bin_text}/",
        "output_tar": out_tar,
        "decision": decision,
        "reason": reason,
        "file_count": tif_count + 1,
        "tif_count": tif_count,
        "source_payload_bytes": source_payload_bytes + len(hitsmisses_data),
        "tar_file_size": tar_file_size,
        "hitsmisses_blob": hitsmisses_blob,
        "hitsmisses_member_last": True,
        "local_tar_path": local_tar_path,
        "error": " | ".join(failed_downloads[:20]),
    }

def upload_tar(
    dest_cc,
    local_tar_path: str,
    output_tar: str,
    overwrite: bool,
    dry_run: bool,
) -> str:
    if dry_run:
        logging.info(
            "UPLOAD_SKIPPED_DRY_RUN output_tar=%s local=%s",
            output_tar,
            local_tar_path,
        )
        return "dry_run_upload_skipped"

    if not overwrite and blob_exists(dest_cc, output_tar):
        logging.info(
            "UPLOAD_SKIPPED_EXISTS output_tar=%s",
            output_tar,
        )
        return "skipped_existing"

    blob_client = dest_cc.get_blob_client(output_tar)
    size = os.path.getsize(local_tar_path)

    logging.info(
        "UPLOAD_START output_tar=%s bytes=%s max_concurrency=%s",
        output_tar,
        size,
        UPLOAD_MAX_CONCURRENCY,
    )

    def do_upload() -> None:
        # Open the file afresh for every retry so that each attempt begins
        # at byte zero.
        with open(local_tar_path, "rb") as fh:
            blob_client.upload_blob(
                fh,
                overwrite=True,
                max_concurrency=UPLOAD_MAX_CONCURRENCY,
                content_settings=ContentSettings(
                    content_type=TAR_CONTENT_TYPE,
                ),
            )

    retry_call(
        label=f"upload:{output_tar}",
        func=do_upload,
    )

    logging.info(
        "UPLOAD_DONE output_tar=%s bytes=%s",
        output_tar,
        size,
    )

    return "uploaded"

def process_tenbin(source_cc, dest_cc, tenbin: Dict[str, object], temp_dir: str, dry_run: bool, overwrite: bool) -> Dict[str, object]:
    result = write_tenbin_tar_to_local_temp(source_cc, tenbin, temp_dir)
    if result["decision"] == "created":
        upload_action = upload_tar(dest_cc, str(result["local_tar_path"]), str(result["output_tar"]), overwrite, dry_run)
        if upload_action == "skipped_existing":
            result["decision"] = "skipped"
            result["reason"] = "destination_tar_exists"
        else:
            result["reason"] = f"{result['reason']}; {upload_action}"
    try:
        if os.path.exists(str(result["local_tar_path"])):
            os.remove(str(result["local_tar_path"]))
    finally:
        result.pop("local_tar_path", None)
    return result


def tar_tiny_files_to_new_containers(
    source_containers: Iterable[str] = SOURCE_CONTAINERS,
    account_url: str = ACCOUNT_URL,
    dry_run: bool = DRY_RUN,
    overwrite_existing_tars: bool = OVERWRITE_EXISTING_TARS,
    report_csv: str = REPORT_CSV,
    survey_start_date: str = SURVEY_START_DATE,
    survey_end_date: str = SURVEY_END_DATE,
):
    setup_logging(LOG_FILE)
        
    logging.info(
        "RUN_START dry_run=%s overwrite_existing_tars=%s "
        "survey_start=%s survey_end=%s probe_every_minute=%s "
        "hitsmisses_workers=%s tenbin_workers=%s tif_workers=%s "
        "tif_max_pending=%s download_concurrency_per_blob=%s "
        "upload_max_concurrency=%s",
        dry_run,
        overwrite_existing_tars,
        survey_start_date,
        survey_end_date,
        PROBE_EVERY_MINUTE,
        HITSMISSES_PROBE_WORKERS,
        TENBIN_WORKERS,
        TIF_DOWNLOAD_WORKERS,
        MAX_PENDING_TIF_DOWNLOADS,
        DOWNLOAD_MAX_CONCURRENCY_PER_BLOB,
        UPLOAD_MAX_CONCURRENCY,
    )    

    bsc = get_blob_service_client(account_url)
    fieldnames = [
        "source_container",
        "destination_container",
        "destination_container_action",
        "source_prefix",
        "output_tar",
        "decision",
        "reason",
        "file_count",
        "tif_count",
        "source_payload_bytes",
        "tar_file_size",
        "hitsmisses_blob",
        "hitsmisses_member_last",
        "error",
    ]
    rows: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="tiny_tars_") as temp_dir:
        with open(report_csv, "w", newline="", encoding="utf-8") as report_fh:
            writer = csv.DictWriter(report_fh, fieldnames=fieldnames)
            writer.writeheader()
            for source_container in source_containers:
                container_start = time.time()
                dest_name = destination_container_name(source_container)
                logging.info("CONTAINER_START source=%s destination=%s", source_container, dest_name)
                source_cc = bsc.get_container_client(source_container)
                try:
                    dest_cc, dest_action = ensure_destination_container(bsc, dest_name, dry_run=dry_run)
                except Exception as e:
                    row = {
                        "source_container": source_container,
                        "destination_container": dest_name,
                        "destination_container_action": "failed",
                        "source_prefix": "",
                        "output_tar": "",
                        "decision": "failed",
                        "reason": "create_destination_container_failed",
                        "file_count": 0,
                        "tif_count": 0,
                        "source_payload_bytes": 0,
                        "tar_file_size": 0,
                        "hitsmisses_blob": "",
                        "hitsmisses_member_last": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                    writer.writerow(row)
                    report_fh.flush()
                    rows.append(row)
                    logging.exception("CONTAINER_FAILED source=%s destination=%s", source_container, dest_name)
                    continue
                try:
                    tenbins = discover_tenbins_from_hitsmisses(source_cc, root="", start_date=survey_start_date, end_date=survey_end_date)
                    if not tenbins:
                        logging.info("CONTAINER_NO_TENBINS source=%s", source_container)
                    with ThreadPoolExecutor(max_workers=TENBIN_WORKERS) as pool:
                        future_to_tenbin = {
                            pool.submit(process_tenbin, source_cc, dest_cc, tenbin, temp_dir, dry_run, overwrite_existing_tars): tenbin
                            for tenbin in tenbins
                        }
                        for fut in as_completed(future_to_tenbin):
                            tenbin = future_to_tenbin[fut]
                            try:
                                result = fut.result()
                                row = {
                                    "source_container": source_container,
                                    "destination_container": dest_name,
                                    "destination_container_action": dest_action,
                                    **{k: result.get(k, "") for k in fieldnames if k not in ("source_container", "destination_container", "destination_container_action")},
                                }
                            except Exception as e:
                                row = {
                                    "source_container": source_container,
                                    "destination_container": dest_name,
                                    "destination_container_action": dest_action,
                                    "source_prefix": f"{tenbin.get('date_text')}/{tenbin.get('bin_text')}/",
                                    "output_tar": output_tar_name(str(tenbin.get("root", "")), str(tenbin.get("date_text")), str(tenbin.get("bin_text"))),
                                    "decision": "failed",
                                    "reason": "tenbin_failed",
                                    "file_count": 0,
                                    "tif_count": 0,
                                    "source_payload_bytes": 0,
                                    "tar_file_size": 0,
                                    "hitsmisses_blob": str(tenbin.get("hitsmisses_blob", "")),
                                    "hitsmisses_member_last": False,
                                    "error": f"{type(e).__name__}: {e}",
                                }
                                logging.exception("TENBIN_FAILED source=%s date=%s bin=%s", source_container, tenbin.get("date_text"), tenbin.get("bin_text"))
                            writer.writerow(row)
                            report_fh.flush()
                            rows.append(row)
                except Exception as e:
                    row = {
                        "source_container": source_container,
                        "destination_container": dest_name,
                        "destination_container_action": dest_action,
                        "source_prefix": "",
                        "output_tar": "",
                        "decision": "failed",
                        "reason": "container_failed",
                        "file_count": 0,
                        "tif_count": 0,
                        "source_payload_bytes": 0,
                        "tar_file_size": 0,
                        "hitsmisses_blob": "",
                        "hitsmisses_member_last": False,
                        "error": f"{type(e).__name__}: {e}",
                    }
                    writer.writerow(row)
                    report_fh.flush()
                    rows.append(row)
                    logging.exception("CONTAINER_FAILED source=%s", source_container)
                logging.info(
                    "CONTAINER_END source=%s destination=%s elapsed_min=%.1f",
                    source_container,
                    dest_name,
                    (time.time() - container_start) / 60.0,
                )
    logging.info("RUN_END report_csv=%s rows=%s", os.path.abspath(report_csv), len(rows))
    return rows


if __name__ == "__main__":
    tar_tiny_files_to_new_containers(
        source_containers=SOURCE_CONTAINERS,
        account_url=ACCOUNT_URL,
        dry_run=DRY_RUN,
        overwrite_existing_tars=OVERWRITE_EXISTING_TARS,
        report_csv=REPORT_CSV,
        survey_start_date=SURVEY_START_DATE,
        survey_end_date=SURVEY_END_DATE,
    )
