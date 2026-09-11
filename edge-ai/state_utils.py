import time
import config

def init_state(LABEL_GROUPS):
    data = {
        "version": config.protocol_version,
        "system_serial_no": config.system_serial_no,
        "equispherdiameter_standard": [],
        "equispherdiameter_otsu": [],
        "thresholdArea_standard": [],
        "thresholdArea_otsu": [],
        "objectlength": [],
        "time_start": time.time(),
        "hits": [],
        "misses": [],
        "randomimage": []
    }
    for label in LABEL_GROUPS:
        count_key = f"{label}Count"
        uncorrected_count_key = f"uncorrected_{label}Count"
        data[count_key] = 0
        data[uncorrected_count_key] = 0
    return data