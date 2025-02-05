# Rewrite of plot_sizes because it was not working with the output from azure
import pandas as pd
import cv2
import numpy as np
import os
import re
import sys
import ast

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

def process_images(input_csv, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    df = pd.read_csv(input_csv)
#    df.columns = ["Filename", "Predicted Class", "score", "Latitude", "Longitude", "Image DateTime", "esd_1", "esd_2", "threshold_area", "threshold_area_james_osho", "object_length", "max_points", "contour_1", "contour_2"]


    def parse_complex_cell(cell):
        try:
            # Remove newlines and extra spaces from the cell
            cleaned_cell = re.sub(r'\s+', ',', cell).strip().replace('[,', '[')
            return cleaned_cell
        except (ValueError, SyntaxError):
            return cell

    # Apply parsing function to columns with complex data
    columns_to_parse = ['contour_1', 'contour_2']
    for column in columns_to_parse:
        df[column] = df[column].apply(parse_complex_cell)
        

    # Example: Print out the DataFrame to verify
    print(df.head())
    

    for index, row in df.iterrows():
        filename = row["Filename"]
        image_path = filename.replace("mnt/c/Users/JR13/", "C:/Users/JR13/").replace("\\", "/").replace("/", "\\")
        print(image_path)
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
#            cv2.circle(image, centroid, 5, (0, 0, 255), -1)
#            cv2.circle(image, centroid, int(esd_1 / 2), (0, 0, 255), 1)
#            cv2.circle(image, centroid, int(esd_2 / 2), (0, 255, 0), 1)
            cv2.line(image, tuple(c_1[0][0]), tuple(c_2[0][0]), (0, 0, 255), 2)
            #"contour_1", "contour_2"
            contour_1 = np.array(eval(row["contour_1"]), dtype=np.int32).reshape((-1, 1, 2))
            contour_2 = np.array(eval(row["contour_2"]), dtype=np.int32).reshape((-1, 1, 2))
            cv2.drawContours(image, [contour_1], -1, (0, 255, 0), 2)
            cv2.drawContours(image, [contour_2], -1, (0, 0, 255), 2)
        
        output_path = os.path.join(output_folder, os.path.basename(filename))
        cv2.imwrite(output_path, image)
        print(f"Image saved to {output_path}")

    print("Processing complete.")

def main():
    if len(sys.argv) != 3:
        print("Usage: python plot_sizes.py <input_csv> <output_folder>")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_folder = sys.argv[2]
    
    process_images(input_csv, output_folder)

if __name__ == "__main__":
    main()
