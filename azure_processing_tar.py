#!/usr/bin/env python3

import argparse
import logging
import os
import tarfile
import io
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import functional
from concurrent.futures import ThreadPoolExecutor, as_completed
from azureml.core import Dataset
from azureml.core import Workspace, Datastore
from classifier import classify_batch, load_model, get_device
from gps import extract_gps
from extractor import extract
from multiprocessing import Pool
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




def load_images_from_tar(tar_file_path):
    with tarfile.open(tar_file_path, 'r') as tar:
        images = []
        imagenames = []
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".tif"):
                file_obj = tar.extractfile(member)
                if file_obj:
                    image = file_obj.read()
                    images.append(image)
                    imagenames.append(member.name)
    return images, imagenames




def process_GPS_images(images):
    gps_info = [extract_gps(img) for img in images]
    return gps_info

def parallel_GPS_images(img):
    with Pool() as pool:
        gps_info = pool.map(extract_gps, img)
    return gps_info

def process_size_images(images):
    size_info = [extract(img) for img in images]
    return size_info

def parallel_size_images(img):
    with Pool() as pool:
        size_info = pool.map(partial(extract, return_contours=True), img)
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
    # Load the appropriate model
    model_path_map = {
        0: "./models/model_18_pre.pth",
        1: "./models/model_18_gray.pth",
        2: "./models/model_18_21May.pth",
    }
    model = load_model(model_path_map.get(args.model_version, model_path_map[0]), device, "resnet18")



    print(args.dsname)
    ws, dataset = movedatasettoazurecomputeinstance(
        wsname=args.wsname,
        wssubscription_id=args.wssubscription_id,
        wsresource_group=args.wsresource_group,
        dsname=args.dsname
    )

    tar_file_path = args.dsname+'/'+args.dsname
    images,imagenames = load_images_from_tar(tar_file_path)

    sz_info = parallel_size_images(images)
#    sz_info = process_size_images(images)
    sz_df = pd.DataFrame(sz_info, columns=["esd_1", "esd_2", "threshold_area", "threshold_area_james_osho", "object_length", "max_points","contour_1","contour_2"], index=imagenames)
    sz_df.index.name = 'Filename'

    gps_info = parallel_GPS_images(images)
#    gps_info = process_GPS_images(images)
    gps_df = pd.DataFrame(gps_info, columns=["Latitude", "Longitude", "Image DateTime"], index=imagenames)
    gps_df.index.name = 'Filename'

    results = []
    scores = []
    
    for i in range(0, len(images), args.batch_size):
        batch_images_subset = images[i : i + args.batch_size]
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

    print(len(imagenames))
    print(len(results))
    print(len(scores))
    classifications_df = pd.DataFrame({
        'Filename': imagenames,
        'Predicted Class': results,
        'score': scores
    })    

    combined_df = pd.merge(classifications_df, gps_df, on='Filename', how='outer')
    combined_df = pd.merge(combined_df, sz_df, on='Filename', how='outer')
    
    # Write combined DataFrame to CSV
    fname = f"{args.dsname}_combined.csv".replace('/', '').replace('\\', '')
    if not os.path.exists("./temp/"):
        os.makedirs("./temp/")
    output_file = f"./temp/{fname}"
    combined_df.to_csv(output_file, index=False)

    # Upload to Azure Blob Storage
    datastore = get_datastore_path(ws, args.datastorename)
    blob_file_path = f"{args.dsname}/{os.path.basename(output_file)}"
    upload_file_to_blob(datastore, output_file, blob_file_path)
    print(f"Uploaded file to blob: {blob_file_path}")
