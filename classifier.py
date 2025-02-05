#!/usr/bin/env python3
import pandas as pd

# Classify images of copepod, non-copepod and detritus.
#
# Run the ResNet50 classifier that was trained on Plankton Analytics
# PIA data supplied by James Scott for the CEFAS Data Study Group at
# the Alan Turing Institute, December 2021.
#
# Based on
# https://github.com/alan-turing-institute/plankton-dsg-challenge/blob/main/notebooks/python/dsg2021/cnn_demo.ipynb
#

from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.transforms import functional
import argparse
import logging
import sys
import torch
import torchvision
import tifffile as tiff
from io import BytesIO
import numpy as np
import os
import csv


def shorten_and_unique_labels(labels):
    if isinstance(labels, list):
        # Handle list input
        labels = [label[:30] for label in labels]  # Shorten to 30 characters
        unique_labels = []
        for label in labels:
            suffix = 1
            new_label = label
            while new_label in unique_labels:  # Ensure uniqueness
                new_label = label[:29] + str(suffix)
                suffix += 1
            unique_labels.append(new_label)
        return unique_labels

    elif isinstance(labels, dict):
        # Handle dictionary input
        keys = [key[:30] for key in labels.keys()]  # Shorten to 30 characters
        unique_labels = {}
        for original_key, value in labels.items():
            new_key = original_key[:30]
            suffix = 1
            while new_key in unique_labels:  # Ensure uniqueness
                new_key = original_key[:29] + str(suffix)
                suffix += 1
            unique_labels[new_key] = value
        return unique_labels

    else:
        raise TypeError("Input must be a list or dictionary")


#LABELS = [    "copepod",    "detritus",    "noncopepod"   ]


LABELS = [    "Detritus",    "Phyto_diatom",    "Phyto_diatom_chaetocerotanae_Chaetoceros",    "Phyto_diatom_rhisoleniales_Guinardia flaccida",    "Phyto_diatom_rhisoleniales_Rhizosolenia",    "Phyto_dinoflagellate_gonyaulacales_Tripos",    "Phyto_dinoflagellate_gonyaulacales_Tripos macroceros",    "Phyto_dinoflagellate_gonyaulacales_Tripos muelleri",    "Zoo_cnidaria",    "Zoo_crustacea_copepod",    "Zoo_crustacea_copepod_calanoida",    "Zoo_crustacea_copepod_calanoida_Acartia",    "Zoo_crustacea_copepod_calanoida_Centropages",    "Zoo_crustacea_copepod_cyclopoida",    "Zoo_crustacea_copepod_cyclopoida_Oithona",    "Zoo_crustacea_copepod_nauplii",    "Zoo_other",    "Zoo_tintinnidae"]  
LABELS = shorten_and_unique_labels(LABELS)

def resnet50(num_classes):
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model


def resnet18(num_classes):
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model


#def resnet18_gray(num_classes):
#    model = torchvision.models.resnet18()
#    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
#    model.eval()
#    return model


def get_device():
    device = torch.device("cpu")

    # Check if GPU is available ..
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logging.info(f"Device : {device}")
    return device


def load_model(filename, device, model_version):
    if model_version == "resnet18":
        model = resnet18(num_classes=len(LABELS))
    if model_version == "resnet50":
        model = resnet50(num_classes=len(LABELS))
    if model_version == "resnet18_gray":
        model = resnet18_gray(num_classes=len(LABELS))
 
    # Load model weights
    model_state_dict = torch.load(filename, map_location="cpu")
    model.load_state_dict(model_state_dict)

    model = model.to(device)

    return model




def classify(image, device, model, gray=False):
    image = tiff.imread(BytesIO(image))

    # Convert Image to tensor and resize it
    t = functional.to_tensor(image)
    t = functional.resize(t, (256, 256))
    t = t.unsqueeze(dim=0)

    if gray:
        t = 0.2125 * t[:, 0, :, :] + 0.7154 * t[:, 1, :, :] + 0.0721 * t[:, 2, :, :]
        t = torch.tile(t, (1, 3, 1, 1))  # Replicate grayscale across RGB channels

    # Model expects a batch of images so let's convert this image tensor to batch of 1 image
    t = t.to(device)

    with torch.set_grad_enabled(False):
        outputs = model(t)
        scores = torch.softmax(outputs, dim=1)  # Calculate softmax scores
        _, preds = torch.max(outputs, 1)

    return LABELS[preds[0]], scores[0]  # Return label and scores


def classify_batch(image_list, device, model, gray, batch_size=1):
    images = [tiff.imread(BytesIO(image)) for image in image_list]

    t_list = [functional.to_tensor(image) for image in images]
    t_resize = [functional.resize(t, (256, 256)) for t in t_list]

    if gray:
        t_resize = [
            0.2125 * t[0, :, :] + 0.7154 * t[1, :, :] + 0.0721 * t[2, :, :]
            for t in t_resize
        ]
        t_resize = [torch.tile(t, (1, 1, 1, 1)) for t in t_resize]
        t_resize = [t.squeeze(dim=0) for t in t_resize]

    t_batch = torch.stack(t_resize, dim=0)
    t_batch = t_batch.to(device)

    with torch.set_grad_enabled(False):
        outputs = model(t_batch)
        scores = torch.softmax(outputs, dim=1)  # Calculate softmax scores
        _, preds = torch.max(outputs, 1)

    labels = [LABELS[i] for i in preds]

    return labels, scores  # Return labels and scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plankton classifier")

    parser.add_argument(
        "-g",
        "--gray",
        action="store_true",
        help="load lighter weight(Resnet 18) with gray scale",
    )

    parser.add_argument(
        "-m",
        "--model_version",
        type=int,
        default=0,
        help="model version zero means Resnet50, and 1 means Resnet18, default=0",
    )

    parser.add_argument(
        "filename",
        type=str,
        help="Path to the tiff file or folder you want to classify",
    )

    parser.add_argument(
        "--folder",
        action="store_true",
        help="folder processing",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing images",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output csv file",
    )
    

    parser.add_argument(
        "-r",
        "--recursive",
        type=int,
        default=0,
        help="Operate on all subdirectories recursively. Off (0) by default",
    )
    
    device = get_device()

    args = parser.parse_args()
    # If you make any changes here, make the same changes in edge-ai.py!

    if args.model_version == 2:
        model = load_model("./models/model_18_21May.pth",
            device,
            "resnet18"
        )
    if args.model_version == 3:
        model = load_model("./models/model_18_3classes_RGB.pth",
            device,
            "resnet18"
        )        
    if args.model_version == 4:
        model = load_model("./models/model_18_18classes_RGB.pth",
            device,
            "resnet18"
        )    


    # Load image or load all images from a folder

    if args.folder:
        image_list = []
        filenames_list = []
        if args.recursive==1:
            for root, _, files in os.walk(args.filename):
            	for filename in files:
                    if "tif" in filename:
                        filenames_list.append(filename)
                        image_path = os.path.join(root, filename)
                        with open(image_path, "rb") as file:
                            image = file.read()
                            image_list.append(image)
        if args.recursive==0:
            for filename in os.listdir(args.filename):
                if "tif" in filename:
                    filenames_list.append(filename)
                    image_path = os.path.join(args.filename, filename)
                    with open(image_path, "rb") as file:
                        image = file.read()
                        image_list.append(image)
        # results = (classify_batch(image_list, device, model, args.gray))
        results = []
        scores = []
        for i in range(0, len(image_list), args.batch_size):
            batch_images = image_list[i : i + args.batch_size]
            batch_results, batch_scores = classify_batch(
                batch_images, device, model, args.gray, args.batch_size
            )
            results.extend(batch_results)
            scores.extend(batch_scores)
            

        with open(args.output, "a", newline="") as csvfile:
            fieldnames = ["Filename", "Predicted Class", "score"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(len(filenames_list)):
                writer.writerow(
                    {"Filename": filenames_list[i], "Predicted Class": results[i], "score": scores[i]}
                )

        print(f"Results saved to {args.output}")

    else:
        with open(args.filename, "rb") as file:
            image = file.read()
            result, score = classify(image, device, model, args.gray)

            # Save result to a CSV file
            with open(args.output, "rb", newline="") as csvfile:
                fieldnames = ["Filename", "Predicted Class", "score"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerow({"Filename": args.output, "Predicted Class": result,  "Score": score})

            print(f"Result saved to {args.output}")
