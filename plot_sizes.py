import pandas as pd
import cv2
import numpy as np
import os
import re
import sys

def preprocess_max_points(max_points_str):
    try:
        match = re.search(r'array\(\[\[(.*?)\]\], dtype=int32\), array\(\[\[(.*?)\]\], dtype=int32\)\)', max_points_str)
        if not match:
            raise ValueError("No valid array structure found")
        
        points_str_1, points_str_2 = match.groups()
        
        points_list_1 = [[int(num) for num in re.findall(r'-?\d+', point)] for point in points_str_1.split('], [')]
        points_list_2 = [[int(num) for num in re.findall(r'-?\d+', point)] for point in points_str_2.split('], [')]
        
        c_1 = np.array(points_list_1, dtype=np.int32).reshape((-1, 1, 2))
        c_2 = np.array(points_list_2, dtype=np.int32).reshape((-1, 1, 2))

        return (c_1, c_2)
    
    except Exception as e:
        print(f"Error preprocessing max_points: {e}")
        return None

def process_images(input_csv, image_dir, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_csv, header=None)
    df.columns = ["Filename", "esd_1", "esd_2", "threshold_area", "threshold_area_james_osho", "object_length", "max_points"]

    for index, row in df.iterrows():
        filename = row["Filename"]
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue
        
        esd_1 = row["esd_1"]
        esd_2 = row["esd_2"]
        threshold_area = row["threshold_area"]
        threshold_area_james_osho = row["threshold_area_james_osho"]
        
        max_points = preprocess_max_points(row["max_points"])
        if max_points is None:
            print(f"Skipping row due to parsing error: {row}")
            continue
        
        c_1, c_2 = max_points
        
        print(f"Thresholded Area: {min(threshold_area, threshold_area_james_osho)} - {max(threshold_area, threshold_area_james_osho)}")
        print(f"ESD: {min(esd_1, esd_2)} - {max(esd_1, esd_2)}")
        
        if c_1.size > 0:
        
            centroid = [int(np.mean([c_1[0][0][0],c_2[0][0][0]])),int(np.mean([c_1[0][0][1],c_2[0][0][1]]))]
            cv2.circle(image, centroid, 5, (0, 0, 255), -1)
            cv2.circle(image, centroid, int(esd_1 / 2), (0, 0, 255), 1)
            cv2.circle(image, centroid, int(esd_2 / 2), (0, 255, 0), 1)
            cv2.line(image, tuple(c_1[0][0]), tuple(c_2[0][0]), (0, 0, 255), 2)
        
        output_path = os.path.join(output_folder, os.path.basename(filename))
        cv2.imwrite(output_path, image)
        print(f"Image saved to {output_path}")

    print("Processing complete.")

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_csv> <image_dir> <output_folder>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    image_dir = sys.argv[2]
    output_folder = sys.argv[3]
    
    process_images(input_csv, image_dir, output_folder)

if __name__ == "__main__":
    main()
