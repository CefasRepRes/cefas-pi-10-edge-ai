#!/usr/bin/env python3

# Good enough background subtraction?
import argparse
import numpy as np
import tifffile as tiff
from io import BytesIO
import logging

# Warning - does not preserve EXIF data!


def background_correction(image_bytes):
    # Read the TIFF image

    stream = BytesIO(image_bytes)
    image = tiff.imread(stream)
    height, width, channels = image.shape

    # The background line occurs after the main tiff file as an
    # additional line. We assume it's `width` wide and consists of BGR
    # triplets

    ptr = stream.tell()

    bg = image_bytes[ptr : ptr + width * 3]
    bg = np.frombuffer(bg, dtype=np.uint8)
    bg = 255 - bg  # invert
    bg = bg.reshape(1, width, 3)

    # Make a full sized background image by stacking rows
    bg = np.vstack([bg] * height)

    # Subtract the background from the image
    d = (image.astype(int) - bg.astype(int)).clip(0, 255).astype(np.uint8)

    stream = BytesIO()
    tiff.imwrite(stream, d)
    stream.seek(0)

    return stream.read()


def main():
    parser = argparse.ArgumentParser(description="PI background correction")

    parser.add_argument(
        "filename", type=str, help="Path to the tiff file you want to correct"
    )

    args = parser.parse_args()
    f = open(args.filename, "rb")
    ba = bytearray(f.read())
    image = background_correction(ba)
    with open("output.tif", "wb") as file:
        file.write(image)


if __name__ == "__main__":
    main()
