import os
import csv
import subprocess
import classifier
from extractor import extract  # Ensure this is correct
from gps import extract_gps  # Ensure this is correct
import torch
import torchvision

base_folder = "C:/Users/JR13/Downloads/surveydata/"

combined_output_file = os.path.join(base_folder, "sizes.csv")

def resnet18(num_classes):
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model

device = classifier.get_device()
model = resnet18(num_classes=3)
model_state_dict = torch.load("./models/model_18_21May.pth", map_location="cpu")
model.load_state_dict(model_state_dict)
model = model.to(device)

LABELS = [
    "copepod",
    "detritus",
    "noncopepod",
]

      
# Iterate over each day folder
for day_folder in os.listdir(base_folder):
    day_folder_path = os.path.join(base_folder, day_folder)
    if os.path.isdir(day_folder_path):
        # Iterate over each minute folder within the day folder
        for minute_folder in os.listdir(day_folder_path):
            minute_folder_path = os.path.join(day_folder_path, minute_folder, "RawImages")
            print(minute_folder)
            if os.path.isdir(minute_folder_path):
                combined_output_file = os.path.join(day_folder_path, minute_folder, "outputs.csv")
                with open(combined_output_file, "a", newline="") as combined_csvfile:
                    combined_writer = csv.writer(combined_csvfile)
                    for filename in os.listdir(minute_folder_path):
                        if filename.lower().endswith(('.jpg', '.jpeg', '.tif', '.tiff', '.png')):
                            filepath = os.path.join(minute_folder_path, filename)
                            sz1, sz2, sz3, sz4, sz5, maxpts1 = extract(filepath)
                            latitude, longitude, image_datetime = extract_gps(filepath)
                            with open(filepath, "rb") as file:
                                [ignoreme, resultstensor] = classifier.classify(file.read(),device,model)
                            score, index = torch.max(resultstensor, dim=0)
                            species = LABELS[index]
                            combined_writer.writerow([sz1, sz2, sz3, sz4, sz5, maxpts1 , latitude, longitude, image_datetime , score, species])