#!/usr/bin/env python3

import argparse
import csv
import datetime
import glob
import logging
import os
import shutil
import sys
import time
from io import BytesIO
from itertools import combinations
import numpy as np
import pandas as pd
import torch
import torchvision
import tifffile as tiff
import cv2
import exifread
import imutils
from imutils import contours
from scipy.spatial import distance as dist
from torchvision.transforms import functional
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from torch.profiler import profile, record_function, ProfilerActivity
from extractor import extract
from gps import extract_gps
from classifier import classify, classify_batch, resnet50, resnet18, resnet18_gray, get_device, load_model
from azureml.core import Dataset
from azureml.core.compute import ComputeTarget
from azureml.data.dataset_factory import DataType
from azureml.core import Workspace, Datastore
from functools import partial


print(torch.version.cuda)
print(torch.cuda.is_available())


LABELS = [
    "copepod",
    "detritus",
    "noncopepod",
]




def movedatasettoazurecomputeinstance(wsname, wssubscription_id, wsresource_group, dsname):
    ws = Workspace.get(name=wsname, subscription_id=wssubscription_id, resource_group=wsresource_group)
    print(ws)
    dataset = Dataset.get_by_name(ws, name=dsname)
    mount_context = dataset.download(target_path=dsname, overwrite=True)
    return ws, dataset


def get_datastore_path(ws, ds_name):
    datastore = Datastore.get(ws, ds_name)
    return datastore

def upload_file_to_blob(datastore, local_file_path, blob_file_path):
    datastore.upload_files(
        files=[local_file_path],
        target_path=blob_file_path,
        overwrite=True,
        show_progress=False
    )

def load_images_in_batches(file_paths, image_batch_size):
    for i in range(0, len(file_paths), image_batch_size):
        batch_paths = file_paths[i:i + image_batch_size]
        images = parallel_load_images(batch_paths)
        yield images, batch_paths        

def load_image(path):
    with open(path, "rb") as file:
        return file.read()

def parallel_load_images(file_paths):
    with Pool() as pool:
        images = pool.map(load_image, file_paths)
    return images


def parallel_GPS_images(file_paths):
    with Pool() as pool:
        gps_info = pool.map(extract_gps, file_paths)
    return gps_info


def parallel_size_images(file_paths):
    with Pool() as pool:
        size_info = pool.map(partial(extract, second_arg=True), file_paths)
    return size_info





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
        "-b",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for processing images",
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
        help="data asset name",
    )    

    parser.add_argument(
        "-w",
        "--datastorename",
        type=str,
        default="0",
        help="data store name",
    )
    

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




    print(args.dsname)
    ws, dataset = movedatasettoazurecomputeinstance(
        wsname=args.wsname,
        wssubscription_id=args.wssubscription_id,
        wsresource_group=args.wsresource_group,
        dsname=args.dsname
    )
    
    downloaded_directory = args.dsname
    image_list = []
    filenames_list = []
    filenames_list = glob.glob(os.path.join(downloaded_directory, '**', '*.tif'), recursive=True)

    

    image_batch_size = 500000 # How many you should load in parallel depends on the RAM of your compute instance
    
    # Process images in batches
#    all_results = []
#    all_scores = []
    batchi=0
    for batch_images, batch_paths in load_images_in_batches(filenames_list, image_batch_size):
        batchi=batchi+1
        print(f"Processing batch of size: {len(batch_images)}")
        
        # Process size and GPS info
        sz_info = parallel_size_images(batch_paths)
        sz_df = pd.DataFrame(sz_info, columns=["esd_1", "esd_2", "threshold_area", "threshold_area_james_osho", "object_length", "max_points","contour_1","contour_2"], index=batch_paths)
        sz_df.index.name = 'Filename'

        gps_info = parallel_GPS_images(batch_paths)
        gps_df = pd.DataFrame(gps_info, columns=["Latitude", "Longitude", "Image DateTime"], index=batch_paths)
        gps_df.index.name = 'Filename'

        results = []
        scores = []
        
        for i in range(0, len(batch_images), args.batch_size):
            batch_images_subset = batch_images[i : i + args.batch_size]
            batch_results, batch_scores = classify_batch(
                batch_images_subset,
                device,
                model,
                args.gray,
                args.batch_size
            )
            results.extend(batch_results)
            scores.extend([score.cpu().numpy().tolist() for score in batch_scores])

        print("Classification complete.")
    
        print(len(batch_paths))
        print(len(results))
        print(len(scores))
        classifications_df = pd.DataFrame({
            'Filename': batch_paths,
            'Predicted Class': results,
            'score': scores
        })    

        combined_df = pd.merge(classifications_df, gps_df, on='Filename', how='outer')
        combined_df = pd.merge(combined_df, sz_df, on='Filename', how='outer')
        
        # Write combined DataFrame to CSV
        fname=f"{args.dsname}_batch_{batchi}_combined.csv".replace('/', '').replace('\\', '')
        if not os.path.exists("./temporary/"):
            os.makedirs("./temporary/")
        output_file = f"./temporary/{fname}"
        combined_df.to_csv(output_file, index=False)

        # Upload to Azure Blob Storage
        datastore = get_datastore_path(ws, args.datastorename)
        blob_file_path = f"{args.dsname}/{os.path.basename(output_file)}"
        upload_file_to_blob(datastore, output_file, blob_file_path)
        print(f"Uploaded file to blob: {blob_file_path}")    
