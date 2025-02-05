#!/usr/bin/env python3

# extract() finds the biggest contour after applying a dynamic threshold to the image. It is dilated and eroded to get rid of small details. 
# A second erosion and dilation is performed to get rid of more detail, with the intention of providing a size "range". The c_2 should be a little smaller than c_1 after the additional dilations and erosions, and is less likely to also contain the antennae, etc.
# extract_visually is the same function with imshow statements so you can see erosions and dilations being performed.



import argparse
import cv2
import numpy as np
import tifffile as tiff
from io import BytesIO
import csv
import os
import datetime
import logging
import sys
from scipy.spatial import distance as dist
import imutils
from imutils import contours
from itertools import combinations

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def extract_visually(filename_or_bytes, return_contours = False):
    try:
        # Load the image
        try:
            image = tiff.imread(BytesIO(filename_or_bytes))  # If bytes
        except:
            image = cv2.imread(filename_or_bytes)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Grayscale', gray)
        cv2.moveWindow('Grayscale', 100, 100)  # Set the position of the window
        cv2.waitKey(0)
        cv2.destroyWindow('Grayscale')

        # Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imshow('Thresholded', thresh)
        cv2.moveWindow('Thresholded', 100, 100)  # Set the position of the window
        cv2.waitKey(0)
        cv2.destroyWindow('Thresholded')

        # Erosions and dilations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('Morph Close', thresh)
        cv2.moveWindow('Morph Close', 100, 100)  # Set the position of the window
        cv2.waitKey(0)
        cv2.destroyWindow('Morph Close')

        thresh = cv2.dilate(thresh, None, iterations=3)
        cv2.imshow('Dilated', thresh)
        cv2.moveWindow('Dilated', 100, 100)  # Set the position of the window
        cv2.waitKey(0)
        cv2.destroyWindow('Dilated')

        thresh = cv2.erode(thresh, None, iterations=3)
        cv2.imshow('Eroded', thresh)
        cv2.moveWindow('Eroded', 100, 100)  # Set the position of the window
        cv2.waitKey(0)
        cv2.destroyWindow('Eroded')

        # Find contours
        cnts = imutils.grab_contours(cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
        cnts = contours.sort_contours(cnts)[0]

        if not cnts:
            return None, None, None, None, None

        # Draw and show the largest contour (c_1)
        c_1 = max(cnts, key=cv2.contourArea)
        contour_img = image.copy()
        cv2.drawContours(contour_img, [c_1], -1, (0, 255, 0), 2)
        cv2.imshow('Largest Contour', contour_img)
        cv2.moveWindow('Largest Contour', 100, 100)  # Set the position of the window
        cv2.waitKey(0)
        cv2.destroyWindow('Largest Contour')

        threshold_area_james_osho = cv2.contourArea(c_1)
        threshold_area_james_osho_um2 = threshold_area_james_osho * (10 ** 2)

        # Create a mask and find contours in the mask
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c_1], -1, (255), thickness=cv2.FILLED)
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=3)
        cv2.imshow('Mask', mask)
        cv2.moveWindow('Mask', 100, 100)  # Set the position of the window
        cv2.waitKey(0)
        cv2.destroyWindow('Mask')

        c_2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_2 = max(c_2, key=cv2.contourArea)

        # Draw and show the second largest contour (c_2)
        contour_img = image.copy()
        cv2.drawContours(contour_img, [c_2], -1, (0, 255, 255), 2)
        cv2.imshow('Second Largest Contour', contour_img)
        cv2.moveWindow('Second Largest Contour', 100, 100)  # Set the position of the window
        cv2.waitKey(0)
        cv2.destroyWindow('Second Largest Contour')

        threshold_area = cv2.contourArea(c_2)
        threshold_area_um2 = threshold_area * (10 ** 2)  # Convert the contour area from pixels to square micrometers.

        # Calculate ESDs
        esd_1 = 2 * np.sqrt(threshold_area_um2 / np.pi)
        esd_2 = 2 * np.sqrt(threshold_area_james_osho_um2 / np.pi)

        # Find the maximum distance in convex hull
        c = cv2.convexHull(c_2)
        object_length = 0
        max_distance = 0
        max_points = None
        for pair in combinations(c, 2):
            distance = np.linalg.norm(pair[0][0] - pair[1][0])
            if distance > max_distance:
                max_distance = distance
                max_points = pair
        object_length = max_distance

    except:
        esd_1 = 0
        esd_2 = 0
        threshold_area = 0
        threshold_area_james_osho = 0
        c_1 = 0
        c_2 = 0
        object_length = 0
        max_points = 0

    if return_contours:
        return [esd_1, esd_2, threshold_area, threshold_area_james_osho, object_length, max_points, c_1, c_2]
    else:
        return [esd_1, esd_2, threshold_area, threshold_area_james_osho, object_length, max_points]



def extract(filename_or_bytes, return_contours = False):
    try:
        try:
            image = tiff.imread(BytesIO(filename_or_bytes)) # If bytes
        except:
            image = cv2.imread(filename_or_bytes)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, None, iterations=3)
        thresh = cv2.erode(thresh, None, iterations=3)
        cnts = imutils.grab_contours(cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
        cnts = contours.sort_contours(cnts)[0]
        if not cnts:
            return None, None, None, None, None
        c_1 = max(cnts, key=cv2.contourArea)
        threshold_area_james_osho = cv2.contourArea(c_1)
        threshold_area_james_osho_um2 = threshold_area_james_osho * (10 ** 2)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c_1], -1, (255), thickness=cv2.FILLED)
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=3)
        c_2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c_2 = max(c_2, key=cv2.contourArea)
        threshold_area = cv2.contourArea(c_2)
        threshold_area_um2 = threshold_area * (10 ** 2) # Convert the contour area from pixels to square micrometers. pixel_area_um2 = 10 ** 2  ... pixel_side_length_um = 10um
        esd_1 = 2 * np.sqrt(threshold_area_um2 / np.pi)
        esd_2 = 2 * np.sqrt(threshold_area_james_osho_um2 / np.pi)
        c = cv2.convexHull(c_2)
        object_length = 0
        max_distance = 0
        max_points = None
        for pair in combinations(c, 2):
            distance = np.linalg.norm(pair[0][0] - pair[1][0])
            if distance > max_distance:
                max_distance = distance
                max_points = pair
        object_length = max_distance
    except:
        esd_1 = 0
        esd_2 = 0
        threshold_area = 0
        threshold_area_james_osho = 0
        c_1 = 0
        c_2 = 0
        object_length = 0
        max_points = 0
    if return_contours:
        return [esd_1, esd_2, threshold_area, threshold_area_james_osho, object_length, max_points, c_1, c_2]
    if return_contours==False:
        return [esd_1, esd_2, threshold_area, threshold_area_james_osho, object_length, max_points]

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract information from an image")
    parser.add_argument("filename_or_bytes", type=str, help="Path to the image file or folder, alternatively this should also accept byte data.")
    parser.add_argument("-o", "--output", type=str, help="Output CSV file")
    parser.add_argument("--folder", action="store_true", help="Process a folder")
    args = parser.parse_args()

    if args.folder:
        image_files = []
        for filename in os.listdir(args.filename_or_bytes):
            if filename.endswith(('.jpg', '.jpeg', '.tif', '.tiff', '.png')):
                filepath = os.path.join(args.filename_or_bytes, filename)
                result = extract(filepath)
#                result = extract_visually(filepath)
                output_file = args.output or "extracted_info.csv"
                with open(output_file, "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([filename] + result)
        try:
            print(f"Information extracted and written to {output_file}")
        except:
            print(f"Nothing saved for folder {args.filename_or_bytes}")
    else:
        result = extract(args.filename_or_bytes)

        # Write result to CSV
        output_file = args.output or "extracted_info.csv"
        with open(output_file, "rb", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["esd_1", "esd_2", "threshold_area", "threshold_area_james_osho", "object_length", "max_points"])
            writer.writerow(result)

        print(f"Information extracted and written to {output_file}")
