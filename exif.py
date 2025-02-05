#!/usr/bin/env python

from PIL import Image
from PIL.Image import Exif
from io import BytesIO
from PIL.ExifTags import TAGS, GPSTAGS
import logging


def get_labeled_exif(exif):
    return {TAGS.get(key, key): value for key, value in exif.items()}


def get_geo(labeled_exif):
    gps_info = labeled_exif.get("GPSInfo", {})
    # print(gps_info)
    return {GPSTAGS.get(key, key): value for key, value in gps_info.items()}


def getexif(image):
    image = Image.open(BytesIO(image))

    exif_data = image.getexif()
    labeled_exif = get_labeled_exif(exif_data)

    # logging.info(labeled_exif)
    # logging.info(labeled_exif["ImageWidth"])
    # logging.info(labeled_exif["ImageLength"])
    # logging.info(labeled_exif["DateTime"])

    return (
        labeled_exif["ImageWidth"],
        labeled_exif["ImageLength"],
        labeled_exif["DateTime"],
    )
