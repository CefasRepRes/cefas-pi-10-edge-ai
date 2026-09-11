#!/usr/bin/env python

from io import BytesIO
import cv2
import tifffile as tiff
import logging


def display(image_bytes, label, gray):
    """Display an image with a label."""

    logging.debug(f"Displaying ..")

    image = tiff.imread(BytesIO(image_bytes))

    scale = 2000 / image.shape[1]

    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)

    image = cv2.resize(image, (width, height))

    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.putText(
        image,
        label,
        (50, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("PI", image)
