import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prediction_records import build_prediction_record


class _FakeGPSModule:
    def extract_gps(self, image_bytes):
        return 51.1234, -2.5678, "2025-01-01 10:20:30"


def test_build_prediction_record_includes_compact_fields_and_gps():
    record = build_prediction_record(
        container="source-container",
        tar_name="2025-01-01/1000.tar",
        member_name="RawImages/pi2.2025-01-01.1000+N00000001.tif",
        predicted_label="fish_larvae",
        confidence=0.91,
        image_bytes=b"image-bytes",
        gps_module=_FakeGPSModule(),
    )

    assert record == {
        "source_tar_blob": "2025-01-01/1000.tar",
        "tar_member": "RawImages/pi2.2025-01-01.1000+N00000001.tif",
        "blob_member_uri": "source-container/2025-01-01/1000.tar!/RawImages/pi2.2025-01-01.1000+N00000001.tif",
        "predicted_label": "fish_larvae",
        "confidence": 0.91,
        "gps": {
            "lat": 51.1234,
            "lon": -2.5678,
            "image_datetime": "2025-01-01 10:20:30",
        },
    }


def test_build_prediction_record_omits_gps_when_extraction_is_unavailable():
    class _EmptyGPSModule:
        def extract_gps(self, _image_bytes):
            return "error", "error", "error"

    record = build_prediction_record(
        container="source-container",
        tar_name="2025-01-01/1000.tar",
        member_name="RawImages/pi2.2025-01-01.1000+N00000001.tif",
        predicted_label="fish_larvae",
        confidence=0.91,
        image_bytes=b"image-bytes",
        gps_module=_EmptyGPSModule(),
    )

    assert "gps" not in record


def test_build_prediction_record_uses_explicit_gps_data():
    gps_payload = {
        "lat": 51.1234,
        "lon": -2.5678,
        "image_datetime": "2025-01-01 10:20:30",
    }
    record = build_prediction_record(
        container="source-container",
        tar_name="2025-01-01/1000.tar",
        member_name="RawImages/pi2.2025-01-01.1000+N00000001.tif",
        predicted_label="fish_larvae",
        confidence=0.91,
        image_bytes=b"image-bytes",
        gps_module=None,
        gps_data=gps_payload,
    )

    assert record["gps"] == gps_payload


def test_build_prediction_record_omits_gps_without_gps_module():
    record = build_prediction_record(
        container="source-container",
        tar_name="2025-01-01/1000.tar",
        member_name="RawImages/pi2.2025-01-01.1000+N00000001.tif",
        predicted_label="fish_larvae",
        confidence=0.91,
        image_bytes=b"image-bytes",
        gps_module=None,
    )

    assert "gps" not in record
