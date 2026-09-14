from __future__ import annotations

from typing import Optional


def _normalize_gps_value(value):
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "error":
        return None
    return value


def extract_gps_payload(image_bytes: Optional[bytes], gps_module) -> Optional[dict]:
    if image_bytes is None or gps_module is None:
        return None
    try:
        # gps.py uses the string "error" when EXIF extraction fails,
        # so normalize that to None and omit the GPS block entirely
        # when every coordinate is unavailable.
        lat, lon, image_datetime = gps_module.extract_gps(image_bytes)
        lat = _normalize_gps_value(lat)
        lon = _normalize_gps_value(lon)
        image_datetime = _normalize_gps_value(image_datetime)
    except Exception:
        return None
    if any(value is not None for value in (lat, lon, image_datetime)):
        return {
            "lat": lat,
            "lon": lon,
            "image_datetime": image_datetime,
        }
    return None


def build_prediction_record(
    *,
    container: Optional[str],
    tar_name: str,
    member_name: str,
    predicted_label: str,
    confidence: float,
    image_bytes: Optional[bytes] = None,
    gps_module=None,
    gps_data: Optional[dict] = None,
) -> dict:
    record = {
        "container": container,
        "source_tar_blob": tar_name,
        "tar_member": member_name,
        "blob_member_uri": f"{container}/{tar_name}!/{member_name}" if container else f"{tar_name}!/{member_name}",
        "predicted_label": predicted_label,
        "confidence": float(confidence),
    }    
    if gps_data is not None:
        record["gps"] = gps_data
    else:
        gps_payload = extract_gps_payload(image_bytes, gps_module)
        if gps_payload is not None:
            record["gps"] = gps_payload
    return record
