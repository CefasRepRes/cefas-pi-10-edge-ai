#!/usr/bin/env python3

import datetime
import logging

# We define a simple binary file that consists of a series of image
# name, newline and image bytes.


def store(image_name, image):
    now = datetime.datetime.now()

    # Filename based on a ten minute bin
    filename = (
        "pi-" + now.strftime("%Y-%m-%d-%H-") + str(now.minute // 10) + "0" + ".bin"
    )

    logging.debug(f"Store: {filename}")

    with open(filename, "ab") as f:
        f.write(int(len(image_name)).to_bytes(2, "little"))
        f.write(image_name.encode("utf-8"))
        f.write(int(len(image)).to_bytes(4, "little"))
        f.write(image)
