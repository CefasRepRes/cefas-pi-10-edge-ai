#!/usr/bin/env python3

import argparse
import os
import csv
from io import BytesIO
import exifread

def extract_gps(filename_or_bytes):
    try:
        try:
            tags = exifread.process_file(BytesIO(filename_or_bytes)) # If bytes
        except:
            with open(filename_or_bytes, "rb") as file:
                image_bytes = file.read()
            stream = BytesIO(image_bytes)
            tags = exifread.process_file(stream)

        latitude = tags.get("GPS GPSLatitude").values
        longitude = tags.get("GPS GPSLongitude").values
        latitude_ref = tags.get("GPS GPSLatitudeRef").values[0]
        longitude_ref = tags.get("GPS GPSLongitudeRef").values[0]
        image_datetime = tags.get("Image DateTime").values

        latitude = float(latitude[0]) + float(latitude[1]) / 60 + float(latitude[2]) / 3600
        if latitude_ref == "S":
            latitude *= -1

        longitude = float(longitude[0]) + float(longitude[1]) / 60 + float(longitude[2]) / 3600
        if longitude_ref == "W":
            longitude *= -1
    except:
        latitude = 'error'
        longitude = 'error'
        image_datetime = 'error'
    return latitude, longitude, image_datetime

def extract_gps_from_folder(folder):
    gps_info = []
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.tif', '.tiff', '.png')):  # Filter only image files
            filepath = os.path.join(folder, filename)
            latitude, longitude, image_datetime = extract_gps(filepath)
            gps_info.append((filename, latitude, longitude, image_datetime))
    return gps_info

# If we are calling this script from command line, parse the command line arguments to run either extract_gps() or extract_gps_from_folder() then save the output to csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get GPS information from image(s)")
    parser.add_argument("filename", type=str, help="Path to the image file or folder")
    parser.add_argument("-o", "--output", type=str, help="Output CSV file")
    parser.add_argument("--folder", action="store_true", help="Process a folder")
    args = parser.parse_args()

    if args.folder:
        gps_info = extract_gps_from_folder(args.filename)
        output_file = args.output or "gps_info.csv"
        with open(output_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "Latitude", "Longitude", "Image DateTime"])
            for info in gps_info:
                writer.writerow(info)
    else:
        filename = args.filename
        latitude, longitude, image_datetime = extract_gps(filename)
        gps_info = [(os.path.basename(filename), latitude, longitude, image_datetime)]

        # Write GPS information to CSV
        output_file = args.output or "gps_info.csv"
        with open(output_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Filename", "Latitude", "Longitude", "Image DateTime"])
            for info in gps_info:
                writer.writerow(info)

    print(f"GPS information written to {output_file}")
