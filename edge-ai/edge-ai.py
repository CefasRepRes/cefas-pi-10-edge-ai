#!/usr/bin/env python

# Portions copyright (c) 2023 Plankton Analytics Ltd.
#
# For Pi_Imager format UDP data stream processing, see
# UserGuide/UDP-data-format.pdf

# To-do:
#need to hold back the reporting until it has the hitsmisses, then only send it when the hitsmisses correction has been applied
#and also correct it by the edgesubrate
#and also multiply by the 34l/min

from datetime import datetime, timezone
from PIL import Image, ImageDraw
from io import BytesIO
from io import StringIO
from logging.handlers import TimedRotatingFileHandler
from struct import *
import argparse
import background
import classifier
from display import display
import cv2
import exif
import extractor
import gps
import logging
import math
import os
import pathlib
import queue
import ring_buffer
import sender
import socket
import state_utils
import statistics
import storage
import sys
import tempfile
import threading
import tifffile as tiff
import time
import csv
import base64
import copy
import json
import subprocess

def shorten_and_unique_labels(labels):
    if isinstance(labels, list):
        # Handle list input
        labels = [label[:30] for label in labels]  # Shorten to 30 characters
        unique_labels = []
        for label in labels:
            suffix = 1
            new_label = label
            while new_label in unique_labels:  # Ensure uniqueness
                new_label = label[:29] + str(suffix)
                suffix += 1
            unique_labels.append(new_label)
        return unique_labels

    elif isinstance(labels, dict):
        # Handle dictionary input
        keys = [key[:30] for key in labels.keys()]  # Shorten to 30 characters
        unique_labels = {}
        for original_key, value in labels.items():
            new_key = original_key[:30]
            suffix = 1
            while new_key in unique_labels:  # Ensure uniqueness
                new_key = original_key[:29] + str(suffix)
                suffix += 1
            unique_labels[new_key] = value
        return unique_labels

    else:
        raise TypeError("Input must be a list or dictionary")


LABEL_GROUPS = {
    "Detritus": ["Detritus"],
    "Copepod": ["Copepod"],
    "Noncopepod": ["Noncopepod"]
}

#LABEL_GROUPS = {
#    "Detritus": ["Detritus"],
#    "Phyto_diatom": ["Phyto_diatom"],
#    "Phyto_diatom_chaetocerotanae_Chaetoceros": ["Phyto_diatom_chaetocerotanae_Chaetoceros"],
#    "Phyto_diatom_rhisoleniales_Guinardia flaccida": ["Phyto_diatom_rhisoleniales_Guinardia flaccida"],
#    "Phyto_diatom_rhisoleniales_Rhizosolenia": ["Phyto_diatom_rhisoleniales_Rhizosolenia"],
#    "Phyto_dinoflagellate_gonyaulacales_Tripos": ["Phyto_dinoflagellate_gonyaulacales_Tripos"],
#    "Phyto_dinoflagellate_gonyaulacales_Tripos macroceros": ["Phyto_dinoflagellate_gonyaulacales_Tripos macroceros"],
#    "Phyto_dinoflagellate_gonyaulacales_Tripos muelleri": ["Phyto_dinoflagellate_gonyaulacales_Tripos muelleri"],
#    "Zoo_cnidaria": ["Zoo_cnidaria"],
#    "Zoo_crustacea_copepod": ["Zoo_crustacea_copepod"],
#    "Zoo_crustacea_copepod_calanoida": ["Zoo_crustacea_copepod_calanoida"],
#    "Zoo_crustacea_copepod_calanoida_Acartia": ["Zoo_crustacea_copepod_calanoida_Acartia"],
#    "Zoo_crustacea_copepod_calanoida_Centropages": ["Zoo_crustacea_copepod_calanoida_Centropages"],
#    "Zoo_crustacea_copepod_cyclopoida": ["Zoo_crustacea_copepod_cyclopoida"],
#    "Zoo_crustacea_copepod_cyclopoida_Oithona": ["Zoo_crustacea_copepod_cyclopoida_Oithona"],
#    "Zoo_crustacea_copepod_nauplii": ["Zoo_crustacea_copepod_nauplii"],
#    "Zoo_other": ["Zoo_other"],
#    "Zoo_tintinnidae": ["Zoo_tintinnidae"]
#}
LABEL_GROUPS = shorten_and_unique_labels(LABEL_GROUPS)

MODEL_DIR = pathlib.Path(__file__).resolve().parent / "models"


def normalise_model_name(model_name):
    """Accept model<timestamp> or model<timestamp>.pt and return the .pt filename."""
    model_name = pathlib.Path(str(model_name).strip()).name
    if not model_name:
        raise ValueError("No model name supplied")
    if not model_name.endswith(".pt"):
        model_name = f"{model_name}.pt"
    if not model_name.startswith("model"):
        raise ValueError("Model name should look like model2026-07-21T09-12-07Z or model2026-07-21T09-12-07Z.pt")
    return model_name


def model_timestamp(model_name):
    model_name = normalise_model_name(model_name)
    return model_name[len("model"):-len(".pt")]


def training_report_name(model_name):
    return f"training_report{model_timestamp(model_name)}.json"


def model_bundle_paths(model_name):
    model_name = normalise_model_name(model_name)
    return {
        "model": MODEL_DIR / model_name,
        "training_report": MODEL_DIR / training_report_name(model_name),
        "settings": MODEL_DIR / "modeltrainsettings.json",
    }


def ensure_model_bundle_available(model_name, sas_token=None):
    """Ensure model and training report are local, calling grabmodel.py if missing."""
    paths = model_bundle_paths(model_name)
    required = [paths["model"], paths["training_report"]]

    if all(path.exists() and path.stat().st_size > 0 for path in required):
        return paths

    logging.info("Model bundle is not fully present locally; calling grabmodel.py")
    grabmodel_path = pathlib.Path(__file__).resolve().parent / "grabmodel.py"
    cmd = [sys.executable, str(grabmodel_path), normalise_model_name(model_name)]
    if sas_token:
        cmd += ["--sas-token", sas_token]
    subprocess.check_call(cmd)

    missing = [str(path) for path in required if not (path.exists() and path.stat().st_size > 0)]
    if missing:
        raise FileNotFoundError("Required model files are still missing after grabmodel.py: " + ", ".join(missing))

    return paths


def configure_classifier_from_model_name(model_name, sas_token=None):
    """Load model, architecture and labels from the selected model's training report."""
    global LABEL_GROUPS

    paths = ensure_model_bundle_available(model_name, sas_token=sas_token)
    with open(paths["training_report"], "r", encoding="utf-8") as f:
        report = json.load(f)

    classes = report.get("classes", [])
    if not classes:
        raise ValueError(f"No classes found in {paths['training_report']}")

    arch = report.get("arch", "resnet18")
    classifier.set_labels(classes)
    LABEL_GROUPS = {label: [label] for label in classifier.LABELS}

    device = classifier.get_device()
    model = classifier.load_model(
        str(paths["model"]),
        device,
        arch,
        labels=classes,
        training_report=report,
    )

    logging.info(f"Using model {paths['model'].name}")
    logging.info(f"Using training report {paths['training_report'].name}")
    logging.info(f"Model architecture: {arch}")
    logging.info(f"Model classes: {classifier.LABELS}")

    return device, model


def configure_legacy_classifier(args, python_path):
    """Retain previous --model_version behaviour for backwards compatibility."""
    device = classifier.get_device()
    print("Model numbers longer implemented. Please use the form -m model2026-07-21T09-12-07Z")
    if args.model_version == 2:
        model = classifier.load_model(
            os.path.normpath(os.path.join(os.path.dirname(python_path), "./models/model_18_21May.pth")),
            device,
            "resnet18",
        )
    elif args.model_version == 3:
        model = classifier.load_model(
            os.path.normpath(os.path.join(os.path.dirname(python_path), "./models/model_18_3classes_RGB.pth")),
            device,
            "resnet18",
        )
    elif args.model_version == 4:
        model = classifier.load_model(
            os.path.normpath(os.path.join(os.path.dirname(python_path), "./models/model_18_18classes_RGB.pth")),
            device,
            "resnet18",
        )
    else:
        raise ValueError(
            "Classification now needs a model name. Use, for example: "
            "python edge-ai.py --classify model_name model2026-07-21T09-12-07Z.pt"
        )

    return device, model


def reset_counters(state):
    logging.debug("Resetting counters..")
    for label in LABEL_GROUPS.keys():
        state[f"uncorrected_{label}Count"] = 0


def apply_hitsmisses_correction(packet, hit, miss):
    """Apply hits/misses correction to a single report packet in-place."""
    logging.debug("Correcting counts in report packet and dividing by fixed flow rate of 34l per min..")
    if hit > 0:
        packet["hits"] = hit
        packet["misses"] = miss
        tot = 0
        for label in LABEL_GROUPS.keys():
            packet[f"{label}Count"] = round(
                ((hit + miss) / hit) * packet["edgeSubRate"] * packet[f"uncorrected_{label}Count"] / 34, 3
            )
            tot += packet["edgeSubRate"] * packet[f"uncorrected_{label}Count"]
        packet["totalCount"] = tot


def send_diag_log(enabled, message):
    if enabled:
        logging.info(f"[send-diagnostics] {message}")


# Execute config to define IP
config = dict(line.strip().split('=') for line in open('ipconfig.txt') if line.strip())
exec("\n".join(f"{key.strip()} = '{value.strip()}'" for key, value in config.items()))
UDP_PORT = int(UDP_PORT)


exiting = False

def get_label_group(label, label_groups=LABEL_GROUPS):
    for pattern, group in label_groups.items():
        if pattern == label or (pattern.endswith('*') and label.startswith(pattern[:-1])):
            return group
    return "Other"


def parse(data, ring):
    """Parse a PI UDP packet, update the ring buffer and return an image if available."""

    logging.debug("Parsing ..")

    image = None
    hitsmisses = None

    hash, field, part, unique_id, total_parts, data_size, tag, pack1, pack2 = unpack(
        "IHHLHHHcc", data[0:24]
    )

    logging.debug(f"{hash},{field},{part},{unique_id},{total_parts},{data_size},{tag}")

    buffer = data[24 : (data_size + 24)]

    if ring.unique_ids[field] != unique_id:
        # This is a new UniqueID, so start over
        ring.unique_ids[field] = unique_id
        ring.counts[field] = 0

    if tag == 0:
        logging.warn(f"{tag} NoTag - shouldn't be sent")

    elif tag == 1:
        logging.debug(f"{tag} Filename - the filename (first packet)")

        filename = buffer.decode("ascii")

        logging.debug("received filename: %s" % filename)

        filename = filename.replace(
            "\\", os.path.sep
        )  # Convert Windows style paths to the host OS convention
        path, filename = os.path.split(filename)
        ring.filenames[field] = filename

    elif tag == 2:
        logging.debug(f"{tag} TiffIfd - a tiff header (second packet)")

    elif tag == 3:
        logging.debug(f"{tag} FileBody - ordinary file data (not a tiff file)")

    elif tag == 4:
        logging.debug(f"{tag} TiffBody - tiff file image data")

    else:
        logging.warn(f"Unknown tag: {tag}")

    # Before assigning ring.buffers[field][part], check there is an index for each part within the buffer. If the total_parts being sent over does not match the actual number of part perhaps that could cause our list assignment index error?
    if part >= len(ring.buffers[field]):
        logging.error(f"Part index {part} out of range for field {field}")
        return None, None, None

    ring.buffers[field][part] = buffer # I suspect this is the line which throws an IndexError: list assignment index out of range

    ring.counts[field] += 1

    if ring.counts[field] >= total_parts:  # All packets received
        filename = ring.filenames[field]

        # We assume at most once delivery from UDP (which is probably
        # reasonable on a simple LAN). We can thus assume that all the
        # buffers are filled and the image is complete.

        if filename != "":
            root, ext = os.path.splitext(filename)
            if ext == ".tif":
                logging.debug(f"Received {filename} ...")
                with tempfile.SpooledTemporaryFile(max_size=100, mode="w+b") as f:
                    for i in range(1, total_parts):
                        f.write(ring.buffers[field][i])
                    f.seek(0)
                    bs = f.read()
                    if filename.lower() != "background.tif":  # Ignore background images
                        image = bs
            elif filename == "HitsMisses.txt":
                logging.info(f"Received {filename} ...")
                with tempfile.SpooledTemporaryFile(max_size=100, mode="w+b") as f:
                    for i in range(1, total_parts):
                        f.write(ring.buffers[field][i])
                    f.seek(0)
                    text = f.read().decode('utf-8')
                    logging.info(f"Text content: {text}")
                    hitsmisses = text

        return image, filename, hitsmisses

    return None, None, None


def parse_offline(sock):
    received_content = bytearray()
    data, address = sock.recvfrom(65536)
    received_content.extend(data)

    return received_content


def my_mean(x):
    if not x:
        return math.nan
    return round(statistics.mean(x),3)

from PIL import Image, ImageDraw
from io import BytesIO

from PIL import Image, ImageDraw
from io import BytesIO

def make_fake_bubble_image(size=46, padding=10):
    """
    Create a slightly squashed synthetic bubble.
    Returns: image_bytes, width, height
    """

    img = Image.new("RGB", (size, size), (220, 220, 220))
    draw = ImageDraw.Draw(img)

    # Slight horizontal squash
    left   = padding + 2
    top    = padding
    right  = size - padding - 4
    bottom = size - padding

    draw.ellipse(
        (left, top, right, bottom),
        fill=(25, 25, 25),
    )

    buf = BytesIO()
    img.save(buf, format="TIFF")

    return buf.getvalue(), size, size
    
def flush_classifier_queue(device, model, classifier, classifier_queue, state, args):
    if classifier_queue.qsize() > 0:
        image_list = list(classifier_queue.queue)
        labels, scores = classifier.classify_batch(
            image_list, device, model, args.gray, batch_size=len(image_list)
        )
        
        #print(labels)
        #print(scores)

        logging.debug(f"Labels: {labels}")
        last_zoo_other_image = None
        for image, label in zip(image_list, labels):
            state_key = f"uncorrected_{label}Count"
            state[state_key] = state.get(state_key, 0) + 1
            if label == "Zoo_crustacea_copepod":
                print("we got one")
                last_zoo_other_image = image
        if last_zoo_other_image is not None:
            if args.sendoneimage:
                image_width, image_length, image_time = exif.getexif(last_zoo_other_image)
                if image_width * image_length < 120000:
                    image_pil = Image.open(BytesIO(last_zoo_other_image))
                    buffered = BytesIO()
                    image_pil.save(buffered, format="PNG", optimize=True)
                    state["randomimage"] = base64.b64encode(buffered.getvalue()).decode('utf-8')
        classifier_queue.queue.clear()




def start_listener(message_queue, args):
    """Start listening on the UDP socket, parsing datagrams and
    queueing images for processing."""
    logging.info(f"Starting listener on : {socket.gethostname()}")

    ring = ring_buffer.RingBuffer()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    logging.debug("Connected")

    BUFFER_SIZE = 8 * 1024 + 24
    buffer = bytearray(BUFFER_SIZE)

    dropped = 0
    packet_count = 0
    subsample_counter = 0
    

    while not exiting:
        if args.debug:
            image = parse_offline(sock)
        else:
            n, addr = sock.recvfrom_into(buffer)
            image, filename, hitsmisses = parse(buffer[0:n], ring)
        if image is not None:
            subsample_counter += 1
            if subsample_counter % args.subsampling_rate != 0:
                continue#skip this image to subsample
                dropped += 1
                logging.debug(f"Dropping image {filename}, count {dropped}")
            if message_queue.full():
                dropped += 1
                logging.debug(f"Dropping image {filename}, count {dropped}")
            else:
                message_queue.put((filename, image, hitsmisses))
        if hitsmisses is not None:
            while True:  # Keep trying indefinitely, the data are useless without these
                if message_queue.full():
                    logging.info("Queue is full and cannot save hits and misses. Retrying in 1 second...")
                    time.sleep(1)  
                else:
                    message_queue.put((filename, image, hitsmisses))
                    logging.info(f"Received {filename}, contents: {hitsmisses}")
                    break

def start_processor(message_queue, args):
    """Start processing images from the queue."""

    python_path = os.path.abspath(__file__)
    device = None
    model = None
    selected_model = getattr(args, "model", None) or getattr(args, "model_name", None)
    if selected_model and not args.classify:
        logging.info("A model was supplied, so classification has been enabled")
        args.classify = True
    if args.classify:
        if selected_model:
            sas_token = getattr(args, "sas_token", None)
            device, model = configure_classifier_from_model_name(selected_model, sas_token=sas_token)
        else:
            device, model = configure_legacy_classifier(args, python_path)
        logging.info("Model initialised")

    state = state_utils.init_state(LABEL_GROUPS)
    state.update({
        "survey": args.survey,
        "edgeSubRate": args.subsampling_rate,
        "totalCount": 0,
        "model_name": pathlib.Path(selected_model).name if selected_model else "legacy_model"
    })

    # Window-based state registry: list of report packets, each stamped with
    # window_start/window_end and status ('pending' or 'validated').
    report_registry = []
    send_attempts_by_window = {}
    registry_lock = threading.Lock()
    # Period-end timestamp of the last received HitsMisses.txt batch.
    last_hitsmisses_period_end = None
    # Tracks whether the current minute window contained any fake-bubble data.
    window_has_fake_bubble = False
    send_diagnostics_enabled = bool(
        getattr(args, "verbose", False)
        or getattr(args, "debug", False)
        or str(getattr(args, "loglevel", "")).upper() == "DEBUG"
    )
    send_diag_log(
        send_diagnostics_enabled,
        "Verbose send diagnostics enabled "
        "(--verbose, --debug, or --loglevel DEBUG).",
    )

    if args.display:
        cv2.namedWindow("PI", cv2.WINDOW_NORMAL)

    # Another queue for batching images, how many images should be held in memory
    classifier_queue = queue.Queue(5000)

    image_count = 0
    rate = 0
    latitude, longitude, image_datetime = [0,0, None]

    def _drain_image_queue():
        """Remove pending image items from message_queue, keeping HitsMisses items."""
        kept = []
        removed_images = 0
        while True:
            try:
                item = message_queue.get_nowait()
                _, _img, _hm = item
                if _hm is not None:
                    kept.append(item)
                else:
                    removed_images += 1
            except queue.Empty:
                break
        for item in kept:
            try:
                message_queue.put_nowait(item)
            except queue.Full:
                logging.warning(
                    "Could not re-queue HitsMisses item after drain: queue full "
                    f"(queue size={message_queue.qsize()})"
                )
        send_diag_log(
            send_diagnostics_enabled,
            f"Drain queue complete: removed_images={removed_images}, "
            f"kept_hitsmisses={len(kept)}, queue_size_now={message_queue.qsize()}",
        )

    def _expire_pending_packets():
        """Expire and print pending packets whose hits/misses window has passed.

        Must be called while holding registry_lock.
        """
        if last_hitsmisses_period_end is None:
            return
        # The HitsMisses batch covers 10 one-minute windows ending at period_end.
        # Any pending packet whose window closed before the start of that batch
        # will never receive correction data; expire it now.
        hm_window_start = last_hitsmisses_period_end - 10 * report_interval
        send_diag_log(
            send_diagnostics_enabled,
            f"Expire check: last_hitsmisses_period_end={last_hitsmisses_period_end}, "
            f"window_start_cutoff={hm_window_start}, registry_size={len(report_registry)}",
        )
        to_remove = [
            p for p in report_registry
            if p["status"] == "pending" and p["window_end"] < hm_window_start
        ]
        for packet in to_remove:
            print(
                f"EXPIRED report packet (window {packet.get('window_start')} - "
                f"{packet.get('window_end')}): {packet}"
            )
            report_registry.remove(packet)
            send_attempts_by_window.pop(packet.get("window_start"), None)
            send_diag_log(
                send_diagnostics_enabled,
                f"Expired pending packet removed: window_start={packet.get('window_start')}, "
                f"window_end={packet.get('window_end')}",
            )

    def process_summary_state_stats():
        while True:
            time.sleep(1)
            if not args.send:
                continue
            with registry_lock:
                _expire_pending_packets()
                pending_count = sum(1 for p in report_registry if p["status"] == "pending")
                validated_count = sum(1 for p in report_registry if p["status"] == "validated")
                send_diag_log(
                    send_diagnostics_enabled,
                    f"Summary send tick: registry_size={len(report_registry)}, "
                    f"pending={pending_count}, validated={validated_count}",
                )
                for packet in list(report_registry):
                    if packet["status"] == "validated":
                        window_key = packet.get("window_start")
                        attempt = send_attempts_by_window.get(window_key, 0) + 1
                        send_attempts_by_window[window_key] = attempt
                        send_diag_log(
                            send_diagnostics_enabled,
                            f"Send decision=validated packet selected: "
                            f"window_start={packet.get('window_start')}, "
                            f"window_end={packet.get('window_end')}, attempt={attempt}",
                        )
                        logging.info(
                            f"Sending queued data window_start={packet.get('window_start')}"
                        )
                        communication_attempt = sender.send(
                            packet,
                            verbose_diagnostics=send_diagnostics_enabled,
                            diagnostics_context={
                                "window_start": packet.get("window_start"),
                                "window_end": packet.get("window_end"),
                                "attempt": attempt,
                            },
                        )
                        if communication_attempt == 1:
                            logging.info(
                                f"Sent queued data window_start={packet.get('window_start')}"
                            )
                            report_registry.remove(packet)
                            send_attempts_by_window.pop(window_key, None)
                            send_diag_log(
                                send_diagnostics_enabled,
                                f"Send succeeded and packet removed from registry: "
                                f"window_start={packet.get('window_start')}, "
                                f"remaining_registry={len(report_registry)}",
                            )
                        else:
                            logging.info(
                                f"Failed to send, will retry: "
                                f"window_start={packet.get('window_start')}"
                            )
                            send_diag_log(
                                send_diagnostics_enabled,
                                f"Send failed; packet kept for retry: "
                                f"window_start={packet.get('window_start')}, attempt={attempt}",
                            )
    
    summary_thread = threading.Thread(target=process_summary_state_stats)
    summary_thread.daemon = True
    summary_thread.start()
    
    # Calculate the next report time
    report_interval = 60#args.report_interval# we want hits and misses to be handled in this loop therefore we need to force 1 min intervals
    next_report_time = (time.time() // report_interval + 1) * report_interval
        
    while True:
        image = None
        hitsmisses = None
        filename = None
        is_fake_bubble = False

        try:
            filename, image, hitsmisses = message_queue.get(timeout=30)

        except queue.Empty:
            if args.onefakebubble:
                logging.warning(
                    f"No messages received from message_queue for 30 seconds "
                    f"(qsize={message_queue.qsize()}); injecting one fake bubble image "
                    f"(--onefakebubble is set)"
                )
                image, fake_width, fake_length = make_fake_bubble_image()
                filename = f"fake_bubble_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.tif"
                is_fake_bubble = True
                window_has_fake_bubble = True
            # Fall through to minute-boundary check even when no image is injected.

        if hitsmisses is not None:
            # A HitsMisses.txt batch has arrived.  Stamp with the current time as
            # the period-end, match each row to the closest pending window packet
            # by time, apply corrections, and clear the image backlog.
            period_end = time.time()
            last_hitsmisses_period_end = period_end
            send_diag_log(
                send_diagnostics_enabled,
                f"HitsMisses batch received: filename={filename}, period_end={period_end}",
            )

            contents = csv.reader(StringIO(hitsmisses))
            rows = [(int(r[0]), int(r[1])) for r in contents if r]
            send_diag_log(
                send_diagnostics_enabled,
                f"HitsMisses parsed rows={len(rows)}",
            )

            if len(rows) == 10:
                with registry_lock:
                    for i, (hit, miss) in enumerate(rows):
                        # Row 0 is the oldest window (period_end - 9 min),
                        # row 9 is the most recent (period_end).
                        target_window_end = period_end - (9 - i) * report_interval
                        best_match = None
                        best_diff = float("inf")
                        for packet in report_registry:
                            if packet["status"] == "pending":
                                diff = abs(packet["window_end"] - target_window_end)
                                # Allow up to half a report interval of drift between the
                                # HitsMisses arrival time and the stored window boundary.
                                if diff < report_interval / 2 and diff < best_diff:
                                    best_diff = diff
                                    best_match = packet
                        if best_match is not None and hit > 0:
                            apply_hitsmisses_correction(best_match, hit, miss)
                            best_match["status"] = "validated"
                            send_diag_log(
                                send_diagnostics_enabled,
                                f"Packet validated from HitsMisses row={i}: "
                                f"target_window_end={target_window_end}, "
                                f"matched_window_start={best_match.get('window_start')}, "
                                f"hit={hit}, miss={miss}",
                            )
                        else:
                            send_diag_log(
                                send_diagnostics_enabled,
                                f"No validation applied for HitsMisses row={i}: "
                                f"target_window_end={target_window_end}, hit={hit}, miss={miss}",
                            )
            else:
                send_diag_log(
                    send_diagnostics_enabled,
                    f"Unexpected HitsMisses row count={len(rows)} (expected 10); "
                    "no packet validation changes applied.",
                )

            _drain_image_queue()
            continue

        if image is not None:
            image_count += 1
            rate = image_count / (time.time() - state["time_start"])

            logging.debug(
                f"Total images {image_count}, rate {rate}/s, queue {message_queue.qsize()}"
            )

            # N.B. background correction does not preserve EXIF data,
            # so if you need metadata, you'd better grab it now.
            # Synthetic fake bubble images have no camera EXIF, so provide
            # dimensions directly and use the current UTC time.
            if is_fake_bubble:
                image_width, image_length = fake_width, fake_length
                image_time = datetime.now(timezone.utc)
            else:
                image_width, image_length, image_time = exif.getexif(image)

            if args.gps:
                latitude, longitude, image_datetime = gps.extract_gps(image)
                state["latitude"] = latitude
                state["longitude"] = longitude
                logging.debug(f"GPS: {latitude}, {longitude}")

            if args.store:
                storage.store(filename, image)

            if args.save:
                if args.classify:
                    label, score = classifier.classify(image, device, model, args.gray)
                    label = label + "-"
                else:
                    label = ""
                with open(f"{label}{filename}", "wb") as file:
                    file.write(image)

            if args.background_correction:
                image = background.background_correction(image)
                # No EXIF data from this point

            if args.classify:
                classifier_queue.put(image)
                #send_diag_log(
                #    send_diagnostics_enabled,
                #    f"Classifier queue add: qsize={classifier_queue.qsize()}, "
                #    f"batch_size={args.batch_size}",
                #)
                if classifier_queue.qsize() >= args.batch_size:
                    # If we've reached the specified batch size then go
                    # ahead and classify:
                    send_diag_log(
                        send_diagnostics_enabled,
                        "Flush classifier queue decision=true (batch threshold reached)",
                    )
                    flush_classifier_queue(
                        device, model, classifier, classifier_queue, state, args
                    )
                #else:
                    #send_diag_log(
                    #    send_diagnostics_enabled,
                    #    "Flush classifier queue decision=false (below batch threshold)",
                    #)

            if args.extract:
                data = extractor.extract(image)
                logging.debug(f"Features : {data}")
                esd_1, esd_2, threshold_area, threshold_area_james_osho, object_length, max_points = data
                state["equispherdiameter_standard"].append(esd_1)
                state["equispherdiameter_otsu"].append(esd_2)
                state["thresholdArea_standard"].append(threshold_area)
                state["thresholdArea_otsu"].append(threshold_area_james_osho)
                state["objectlength"].append(object_length)

            if args.display:
                if args.classify:
                    label, score = classifier.classify(image, device, model, args.gray)
                else:
                    label = ""
                display(image, label, args.gray)

        # Check if it's time to report
        current_time = time.time()
        if current_time >= next_report_time:
            # At this point, we are at the end of the bin, so we want
            # to finish any classification and summarisation before
            # reporting, and possibly sending, the data.

            if args.classify:
                # Classify any outstanding samples, irrespective of
                # whether we have reached the batch size:
                send_diag_log(
                    send_diagnostics_enabled,
                    f"Minute boundary flush: classifier_queue_qsize={classifier_queue.qsize()}",
                )
                flush_classifier_queue(
                    device, model, classifier, classifier_queue, state, args
                )

            logging.info(
                f"Total images {image_count}, rate {rate}/s, queue {message_queue.qsize()}"
            )

            state["time_end"] = current_time

            if args.classify:
                for key, value in state.items():
                    if key not in {"objectlength", "thresholdArea_standard", "thresholdArea_otsu","equispherdiameter_otsu","equispherdiameter_standard"}:
                        logging.info(f"{key}: {value}")

            if args.extract:
                # We have lists of particle area and axes dimensions,
                # and we must now summarise the distribution for this
                # bin.
                state["objectlength"] = my_mean(state["objectlength"])
                state["thresholdArea_otsu"] = my_mean(state["thresholdArea_otsu"])
                state["thresholdArea_standard"] = my_mean(state["thresholdArea_standard"])
                state["equispherdiameter_otsu"] = my_mean(state["equispherdiameter_otsu"])
                state["equispherdiameter_standard"] = my_mean(state["equispherdiameter_standard"])

            state["timestamp"] = datetime.now(timezone.utc).isoformat()

            logging.debug("Closing minute window and adding to report registry ..")

            # Stamp the window and deep-copy into the registry as a pending packet.
            window_packet = copy.deepcopy(state)
            window_packet["window_start"] = state["time_start"]
            window_packet["window_end"] = current_time
            window_packet["status"] = "pending"
            send_diag_log(
                send_diagnostics_enabled,
                f"Window packet queued: window_start={window_packet['window_start']}, "
                f"window_end={window_packet['window_end']}, status={window_packet['status']}",
            )

            if window_has_fake_bubble:
                # Spoof hits/misses for this fake-bubble window and validate immediately.
                apply_hitsmisses_correction(window_packet, hit=1, miss=0)
                window_packet["status"] = "validated"
                send_diag_log(
                    send_diagnostics_enabled,
                    "Window packet marked validated immediately due to fake bubble",
                )

            with registry_lock:
                report_registry.append(window_packet)
                send_diag_log(
                    send_diagnostics_enabled,
                    f"Report registry append complete: registry_size={len(report_registry)}",
                )
                _expire_pending_packets()

            # Clear image backlog at each minute boundary (keep HitsMisses items).
            _drain_image_queue()

            model_name = state.get("model_name", "unknown")

            logging.debug("Resetting counters..")
            state["time_start"] = current_time
            reset_counters(state)
            state["latitude"] = 0  # FIXME
            state["longitude"] = 0  # FIXME

            state["objectlength"] = []
            state["thresholdArea_otsu"] = []
            state["thresholdArea_standard"] = []
            state["equispherdiameter_otsu"] = []
            state["equispherdiameter_standard"] = []
            state["hits"] = 0
            state["misses"] = []
            state["totalCount"] = 0
            state["model_name"] = model_name
            send_diag_log(
                send_diagnostics_enabled,
                f"Post-send/reset cleanup complete: next_window_start={state['time_start']}, "
                f"registry_size={len(report_registry)}, queue_size={message_queue.qsize()}",
            )

            image_count = 0
            window_has_fake_bubble = False

            next_report_time = (current_time // report_interval + 1) * report_interval
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            return


def main(args):
    # Configure logging
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    # Create the logger and set the log level
    logger = logging.getLogger()
    logger.setLevel(args.loglevel)

    # Create the console handler and set the log format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

    # Create the file handler and set the log format
    file_handler = TimedRotatingFileHandler(
        "edge-ai.log", when="midnight", backupCount=21
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)

    logging.info("Plankton Imager Edge-AI starting ..")
    logging.info(f"Survey : {args.survey}")

    message_queue = queue.Queue(100)  # For inter-thread communication

    # Start a background thread for the UDP listener which retrieves
    # images and queues them on message_queue

    listener_thread = threading.Thread(
        target=start_listener, args=(message_queue, args), daemon=True
    )
    listener_thread.start()

    # Start a processor on the main thread to dequeue images and
    # process them. N.B. Python requires us to run the GUI on this
    # foreground thread.

    start_processor(message_queue, args)

    logging.info("Plankton Imager Edge-AI stopping ..")

    listener_thread.join(timeout=10)
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plankton Imager Edge-AI System")
    parser.add_argument("-c", "--classify", action="store_true", help="classify images")
    parser.add_argument(
        "-d", "--display", action="store_true", help="display images (warning slow!)"
    )
    parser.add_argument(
        "-e", "--extract", action="store_true", help="extract morphological features"
    )
    parser.add_argument(
        "-s", "--store", action="store_true", help="store images on disk"
    )
    parser.add_argument(
        "--save", action="store_true", help="store images individually on disk"
    )

    parser.add_argument(
        "-t",
        "--send",
        action="store_true",
        help="send (transmit) data to the dashboard",
    )

    parser.add_argument(
        "--model",
        help="Model name, e.g. model2026-07-21T09-12-07Z or model2026-07-21T09-12-07Z.pt. Downloads it with grabmodel.py if missing.",
    )
    
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Optional positional form of --model, e.g. python edge-ai.py model2026-07-21T09-12-07Z.pt",
    )

    parser.add_argument(
        "-g",
        "--gray",
        action="store_true",
        help="load lighter weight(Resnet 18) with gray scale",
    )
    parser.add_argument(
        "--background-correction",
        action="store_true",
        help="apply background correction to images",
    )
    parser.add_argument(
        "--gps", action="store_true", help="extract GPS position from EXIF data"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable verbose console logging and send diagnostics. Also see --loglevel",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="use Mojtaba's UDP protocol rather than the PIA protocol.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=255,
        help="batch size for bulk classification, default=255",
    )
    parser.add_argument(
        "--survey",
        default="not specified",
        help="name of the survey",
    )
    parser.add_argument(
        "--report_interval",
        type=int,
        default=60,
        help="how often the system reports statistics (how often it sends to the dashboard) in seconds, default=20s",
    )
    parser.add_argument(
        "-l",
        "--loglevel",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--sendoneimage",
        action="store_true",
        help="apply background correction to images",
    )
    parser.add_argument(
        "--onefakebubble",
        action="store_true",
        help=(
            "When the message queue is idle for 30 seconds, inject one synthetic "
            "fake-bubble image and spoof its hits/misses (1 hit, 0 misses) so it "
            "follows the same validation and sending pipeline as real data. "
            "Without this flag, queue timeouts are silently ignored."
        ),
    )
    parser.add_argument(
    "--subsampling_rate",
    type=int,
    default=1,
    help="Positive number N for the subsampling rate for UDP packets processing. E.g., 1 in N packets will be processed. Default is 1 (no subsampling).",
    )
    parser.add_argument(
        "--sas-token",
        metavar="TOKEN",
        help=(
            "SAS token for the Azure Blob Storage container used by grabmodel.py "
            "to download the model bundle. The leading '?' is optional. "
            "If omitted and the model needs to be downloaded, grabmodel.py will "
            "prompt interactively."
        ),
    )
    main(parser.parse_args())
