#!/usr/bin/env python3

import logging
import sys


# Extracts TIFF files from our bin files


def split_bin(filename):
    with open(filename, mode="rb") as f:
        while True:
            n = int.from_bytes(f.read(2), "little")

            filename = f.read(n).decode("utf-8")

            if filename == "":
                break

            n = int.from_bytes(f.read(4), "little")

            b = f.read(n)

            with open(filename, "wb") as g:
                g.write(b)


def main():
    filename = sys.argv[1]
    split_bin(filename)


if __name__ == "__main__":
    main()
