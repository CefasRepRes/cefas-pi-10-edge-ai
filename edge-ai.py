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
from PIL import Image
from io import BytesIO
from io import StringIO
from logging.handlers import TimedRotatingFileHandler
from struct import *
import argparse
import background
import classifier
import display
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
    "Phyto_diatom": ["Phyto_diatom"],
    "Phyto_diatom_chaetocerotanae_Chaetoceros": ["Phyto_diatom_chaetocerotanae_Chaetoceros"],
    "Phyto_diatom_rhisoleniales_Guinardia flaccida": ["Phyto_diatom_rhisoleniales_Guinardia flaccida"],
    "Phyto_diatom_rhisoleniales_Rhizosolenia": ["Phyto_diatom_rhisoleniales_Rhizosolenia"],
    "Phyto_dinoflagellate_gonyaulacales_Tripos": ["Phyto_dinoflagellate_gonyaulacales_Tripos"],
    "Phyto_dinoflagellate_gonyaulacales_Tripos macroceros": ["Phyto_dinoflagellate_gonyaulacales_Tripos macroceros"],
    "Phyto_dinoflagellate_gonyaulacales_Tripos muelleri": ["Phyto_dinoflagellate_gonyaulacales_Tripos muelleri"],
    "Zoo_cnidaria": ["Zoo_cnidaria"],
    "Zoo_crustacea_copepod": ["Zoo_crustacea_copepod"],
    "Zoo_crustacea_copepod_calanoida": ["Zoo_crustacea_copepod_calanoida"],
    "Zoo_crustacea_copepod_calanoida_Acartia": ["Zoo_crustacea_copepod_calanoida_Acartia"],
    "Zoo_crustacea_copepod_calanoida_Centropages": ["Zoo_crustacea_copepod_calanoida_Centropages"],
    "Zoo_crustacea_copepod_cyclopoida": ["Zoo_crustacea_copepod_cyclopoida"],
    "Zoo_crustacea_copepod_cyclopoida_Oithona": ["Zoo_crustacea_copepod_cyclopoida_Oithona"],
    "Zoo_crustacea_copepod_nauplii": ["Zoo_crustacea_copepod_nauplii"],
    "Zoo_other": ["Zoo_other"],
    "Zoo_tintinnidae": ["Zoo_tintinnidae"]
}
LABEL_GROUPS = shorten_and_unique_labels(LABEL_GROUPS)

def reset_counters(state):
    logging.debug("Resetting counters..")
    for label in LABEL_GROUPS.keys():
        state[f"uncorrected_{label}Count"] = 0


def calculate_concentrations(state_list,n,hit, miss):
    logging.debug("Correcting counts in previous states and dividing by fixed flow rate of 34l per min..")
    if n in state_list:
        if hit>0:
            tot=0
            state_list[n]["hits"] = hit
            state_list[n]["misses"] = miss
            for label in LABEL_GROUPS.keys():
                state_list[n][f"{label}Count"] = round(((hit+miss)/hit) * state_list[n]["edgeSubRate"] * state_list[n][f"uncorrected_{label}Count"] / 34, 3)
                tot=tot+(state_list[n]["edgeSubRate"] * state_list[n][f"uncorrected_{label}Count"])
            state_list[n][f"totalCount"] = tot


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

def flush_classifier_queue(device, model, classifier, classifier_queue, state, args):
    if classifier_queue.qsize() > 0:
        image_list = list(classifier_queue.queue)
        labels, scores = classifier.classify_batch(
            image_list, device, model, args.gray, batch_size=len(image_list)
        )
        logging.debug(f"Label: {labels}")
        for label in labels:
            #group = get_label_group(label)
            state_key = f"uncorrected_{label}Count"
            state[state_key] = state.get(state_key, 0) + 1
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

    state = state_utils.init_state(LABEL_GROUPS)    
    state.update({
        "survey": args.survey,
        "edgeSubRate": args.subsampling_rate,
        "totalCount": 0
    })
    
    stateminus1 = copy.deepcopy(state)
    stateminus2 = copy.deepcopy(state)
    stateminus3 = copy.deepcopy(state)
    stateminus4 = copy.deepcopy(state)
    stateminus5 = copy.deepcopy(state)
    stateminus6 = copy.deepcopy(state)
    stateminus7 = copy.deepcopy(state)
    stateminus8 = copy.deepcopy(state)
    stateminus9 = copy.deepcopy(state)
    stateminus10 = copy.deepcopy(state)
    
    
    python_path = os.path.abspath(__file__)

    if args.classify:
        # If you make any changes here, make the same changes in classifier.py!
        device = classifier.get_device()

        if args.model_version == 2:
            model = classifier.load_model(
                os.path.normpath(os.path.join(os.path.dirname(python_path), "./models/model_18_21May.pth")),
                device,
                "resnet18"
            )
        if args.model_version == 3:
            model = classifier.load_model(
                os.path.normpath(os.path.join(os.path.dirname(python_path), "./models/model_18_3classes_RGB.pth")),
                device,
                "resnet18"
            )
        if args.model_version == 4:
            model = classifier.load_model(
                os.path.normpath(os.path.join(os.path.dirname(python_path), "./models/model_18_18classes_RGB.pth")),
                device,
                "resnet18"
            )            

        logging.info("Model initialised")

    if args.display:
        cv2.namedWindow("PI", cv2.WINDOW_NORMAL)

    # Another queue for batching images, how many images should be held in memory
    classifier_queue = queue.Queue(5000)

    # Queue for summary statistics in case of internet drop
    summary_stats_queue = queue.Queue(5000)

    image_count = 0
    latitude, longitude, image_datetime = [None, None, None]

    def process_summary_state_stats():
        while True:
            if not summary_stats_queue.empty():
                time.sleep(1)
                #with summary_stats_queue.mutex:
                #    logging.info(f"Queued data {summary_stats_queue.queue}")       # To look at what is in the queue         
                stats = summary_stats_queue.get()
                logging.info(f"Sending queued data {stats['time_start']}")
                communication_attempt = sender.send(stats)
                if communication_attempt == 0:
                    logging.info(f"Failed to send queued data, putting back in the queue: {stats['time_start']}")
                    summary_stats_queue.put(stats)
                if communication_attempt == 1:
                    logging.info(f"Sent queued data {stats['time_start']}")
                    summary_stats_queue.task_done()
            time.sleep(1)  # Retry checking and sending a json every 3 seconds if there are unsent data
    
    summary_thread = threading.Thread(target=process_summary_state_stats)
    summary_thread.daemon = True
    summary_thread.start()
    
    # Calculate the next report time
    report_interval = 60#args.report_interval# we want hits and misses to be handled in this loop therefore we need to force 1 min intervals
    next_report_time = (time.time() // report_interval + 1) * report_interval

    while True:
        filename, image, hitsmisses = message_queue.get()
        if hitsmisses is not None:
            contents = csv.reader(StringIO(hitsmisses))
            hits = []
            misses = []
            for row in contents:
                if row:
                    row_hits, row_misses = row
                    hits.append(int(row_hits))
                    misses.append(int(row_misses))
            state["hits"] = hits
            state["misses"] = misses
            continue
            
        image_count += 1
        rate = image_count / (time.time() - state["time_start"])

        logging.debug(
            f"Total images {image_count}, rate {rate}/s, queue {message_queue.qsize()}"
        )

        # N.B. background correction does not preserve EXIF data,
        # so if you need metadata, you'd better grab it now:
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
            if classifier_queue.qsize() >= args.batch_size:
                # If we've reached the specified batch size then go
                # ahead and classify:
                flush_classifier_queue(
                    device, model, classifier, classifier_queue, state, args
                )

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

            logging.debug("Check for hitsmisses and increment along previous observations by +1 minute ..")
            stateminus11 = copy.deepcopy(stateminus10)
            stateminus10 = copy.deepcopy(stateminus9)
            stateminus9 = copy.deepcopy(stateminus8)
            stateminus8 = copy.deepcopy(stateminus7)
            stateminus7 = copy.deepcopy(stateminus6)
            stateminus6 = copy.deepcopy(stateminus5)
            stateminus5 = copy.deepcopy(stateminus4)
            stateminus4 = copy.deepcopy(stateminus3)
            stateminus3 = copy.deepcopy(stateminus2)
            stateminus2 = copy.deepcopy(stateminus1)
            stateminus1 = copy.deepcopy(state)

            state_list = {
                0: stateminus11,
                1: stateminus10,# In our hitsmisses list, index 1 = 10 minutes ago
                2: stateminus9,
                3: stateminus8,
                4: stateminus7,
                5: stateminus6,
                6: stateminus5,
                7: stateminus4,
                8: stateminus3,
                9: stateminus2,
                10: stateminus1 # In our hitsmisses list, index 10 = 1 minutes ago
            }

            if isinstance(state["hits"], list) and len(state["hits"]) == 10:
                for n, (hit, miss) in enumerate(zip(state["hits"], state["misses"]), start=1): # In our hitsmisses list, index 1 = 10 minutes ago               
                    calculate_concentrations(state_list,n,hit, miss)
            if args.send:
#                    if args.sendoneimage:
#                        if image_width + image_length < 1500:
#                            state["randomimage"] = base64.b64encode(image).decode('utf-8')
                try:
                    accounted= int(state_list[0]["hits"] + state_list[0]["misses"])
                    counted= int(state_list[0]["totalCount"])
                    if True:#(counted / accounted) > 0.5: # If we can't account for more than 50% of the images, don't upload.
                        logging.info(f"Dashboard - queuing upload of data delayed by mins: 10")
                        summary_stats_queue.put(copy.deepcopy(state_list[0]))
    #                    logging.info(f"Dashboard - queuing upload of data delayed by mins: 10")
    #                    summary_stats_queue.put(copy.deepcopy(state_list[10]))
    #                    logging.info(f"Dashboard - queuing upload of data delayed by mins: 9")
    #                    summary_stats_queue.put(copy.deepcopy(state_list[9]))
    #                    logging.info(f"Dashboard - queuing upload of data delayed by mins: 8")
    #                    summary_stats_queue.put(copy.deepcopy(state_list[8]))
    #                    logging.info(f"Dashboard - queuing upload of data delayed by mins: 7")
    #                    summary_stats_queue.put(copy.deepcopy(state_list[7]))
    #                    logging.info(f"Dashboard - queuing upload of data delayed by mins: 6")
    #                    summary_stats_queue.put(copy.deepcopy(state_list[6]))
    #                    logging.info(f"Dashboard - queuing upload of data delayed by mins: 5")
    #                    summary_stats_queue.put(copy.deepcopy(state_list[5]))
    #                    logging.info(f"Dashboard - queuing upload of data delayed by mins: 4")
    #                    summary_stats_queue.put(copy.deepcopy(state_list[4]))
    #                    logging.info(f"Dashboard - queuing upload of data delayed by mins: 3")
    #                    summary_stats_queue.put(copy.deepcopy(state_list[3]))
    #                    logging.info(f"Dashboard - queuing upload of data delayed by mins: 2")
    #                    summary_stats_queue.put(copy.deepcopy(state_list[2]))
    #                    logging.info(f"Dashboard - queuing upload of data delayed by mins: 1")
    #                    summary_stats_queue.put(copy.deepcopy(state_list[1]))
                except:
                        print("unable to validate counts against hits & misses")


            logging.debug("Resetting counters..")
            state["time_start"] = current_time
            reset_counters(state)
            state["latitude"] = latitude  # FIXME
            state["longitude"] = longitude  # FIXME

            state["objectlength"] = []
            state["thresholdArea_otsu"] = []
            state["thresholdArea_standard"] = []
            state["equispherdiameter_otsu"] = []
            state["equispherdiameter_standard"] = []
            state["hits"] = 0
            state["misses"] = []
            state["totalCount"] = 0


            image_count = 0



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
        "-m",
        "--model_version",
        type=int,
        default=0,
        help="model version zero means Resnet50, and 1 means Resnet15, default=0",
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
        help="log to the console (as well as file). Also see --loglevel",
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
    "--subsampling_rate",
    type=int,
    default=1,
    help="Positive number N for the subsampling rate for UDP packets processing. E.g., 1 in N packets will be processed. Default is 1 (no subsampling).",
    )   
    main(parser.parse_args())
