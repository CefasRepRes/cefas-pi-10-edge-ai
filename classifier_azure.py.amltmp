#!/usr/bin/env python3

import pandas as pd
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
import shutil
from torchvision.transforms import functional
import time
from torch.profiler import profile, record_function, ProfilerActivity
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

print(torch.version.cuda)
print(torch.cuda.is_available())


LABELS = [
    "copepod",
    "detritus",
    "noncopepod",
]


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


def resnet18_gray(num_classes):
    model = torchvision.models.resnet18()
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.eval()
    return model


def get_device():
    device = torch.device("cpu")

    # Check if GPU is available ..
    if torch.cuda.is_available():
        device = torch.device("cuda")

    logging.info(f"Device : {device}")
    print(f"Device : {device}")
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


def azurecomputeinstance(wsname,wssubscription_id,wsresource_group,dsname):
    from azureml.core import Dataset
    from azureml.core.compute import ComputeTarget
    from azureml.data.dataset_factory import DataType
    from azureml.core import Workspace, Datastore
    ws = Workspace.get(name=wsname,
                    subscription_id=wssubscription_id,
                    resource_group=wsresource_group)
    print(ws)
    datasets = Dataset.get_all(ws)
#    for dataset in datasets:
#        print("Your compute instance is able to see the dataset called:")
#        print(dataset)
    dataset = Dataset.get_by_name(ws, name=dsname)
    mount_context = dataset.download(target_path=dsname, overwrite=True)
#    print("downloading to "+dsname)



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

    parser.add_argument(
        "-s",
        "--wsname",
        type=str,
        default="0",
        help="work space name",
    )

    parser.add_argument(
        "-t",
        "--wssubscription_id",
        type=str,
        default="0",
        help="work space subscription id",
    )
    
    parser.add_argument(
        "-u",
        "--wsresource_group",
        type=str,
        default="0",
        help="work space resource group",
    )
    
    parser.add_argument(
        "-v",
        "--dsname",
        type=str,
        default="0",
        help="data store name",
    )
    
    parser.add_argument("-a", "--azure", type=int, default=0, help="Set to 1 if running on azure?")
    
    device = get_device()

    args = parser.parse_args()
    # If you make any changes here, make the same changes in edge-ai.py!
    if args.model_version == 1:
        model = load_model("./models/model_18_gray.pth",
            device,
            "resnet18_gray"
        )
    if args.model_version == 2:
        model = load_model("./models/model_18_21May.pth",
            device,
            "resnet18"
        )
    if args.model_version == 0:
        model = load_model("./models/model_18_pre.pth",
            device,
            "resnet18"
        )


    if args.azure == 1:
        azurecomputeinstance(wsname=args.wsname,wssubscription_id=args.wssubscription_id,wsresource_group=args.wsresource_group,dsname=args.dsname)
        downloaded_directory = args.dsname
#        print(downloaded_directory)
        image_list = []

        filenames_list = []
        
        def load_image(path):
#            print(f"Loading image from: {path}")
#            print("in load_image, os.listdir(/tmp):")
#            print(os.listdir("/tmp"))
#            print("in load_image, downloaded_directory:")
#            print(downloaded_directory)
#            print("in load_image, os.listdir(downloaded_directory):")
#            print(os.listdir(downloaded_directory))

            with open(path, "rb") as file:
                return file.read()
        
        import glob
#        print('Getting filenames list...')
        filenames_list = glob.glob(os.path.join(downloaded_directory+args.filename, '**', '*.tif'), recursive=True)
#        print(f"Found {len(filenames_list)} files.")
#        print("cwd before pool:")
#        print(os.getcwd())   


        def parallel_load_images(file_paths):
#            print(f"Loading {len(file_paths)} images in parallel...")
            #with ThreadPool() as pool:
            with Pool() as pool:
#                print("cwd in pool:")
#                print(os.getcwd())
#                print("os.listdir(/tmp):")
#                print(os.listdir("/tmp"))
                images = pool.map(load_image, file_paths)
#            print(f"Loaded {len(images)} images.")
            return images
        
        images = parallel_load_images(filenames_list)
        print(f"Total images loaded: {len(images)}")
        
        results = []
        scores = []
        
        print(f"Processing images in batches of size {args.batch_size}...")
        for i in range(0, len(images), args.batch_size):
            batch_images = images[i : i + args.batch_size]
#            print(f"Classifying batch {i // args.batch_size + 1}/{(len(images) // args.batch_size) + 1}...")
            batch_results, batch_scores = classify_batch(
                batch_images,
                device,
                model,
                args.gray,
                args.batch_size
            )
#            print(f"Batch results: {batch_results}")
#            print(f"Batch scores: {batch_scores}")
            results.extend(batch_results)
            scores.extend(batch_scores)
        
        print("Classification complete.")
        
#        print("Results:")
#        print(results)
        
#        print("Filenames list:")
#        print(filenames_list)
        
#        print(f"Writing results to {args.output}...")
        with open(args.output, "a", newline="") as csvfile:
            fieldnames = ["Filename", "Predicted Class", "score"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(len(filenames_list)):
                writer.writerow({"Filename": filenames_list[i], "Predicted Class": results[i], "score": scores[i]})
#                print(f"Written: {filenames_list[i]}, {results[i]}, {scores[i]}")
        
        print("Finished writing results.")
    
    
