import os
import subprocess
import argparse
from azure.storage.blob import BlobServiceClient
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from collections import defaultdict

# Function to connect to Azure Blob Storage
def connect_to_blob_storage(connection_string):
    return BlobServiceClient.from_connection_string(connection_string)

# Function to download blob from container using ThreadPoolExecutor for parallel downloading
def download_blob_parallel(container_client, blob_name, local_file_name, downloaded_files):
    with open(local_file_name, "wb") as my_blob:
        download_stream = container_client.download_blob(blob_name)
        my_blob.write(download_stream.readall())
    # Add the downloaded file to the downloaded_files dictionary
    downloaded_files[os.path.basename(os.path.dirname(blob_name))].append(blob_name)
    #print(f"Downloaded file: {blob_name}")

# Function to process blobs in source container
def process_blobs_parallel(source_container_client, destination, limit=None):
    blob_list = source_container_client.list_blobs()
    downloaded_files = defaultdict(list)
    with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust max_workers as needed
        future_to_blob = {executor.submit(download_blob_parallel, source_container_client, blob.name, os.path.join(destination, blob.name), downloaded_files): blob for blob in blob_list}
        for future in concurrent.futures.as_completed(future_to_blob):
            try:
                future.result()
            except Exception as exc:
                print(f"Blob {future_to_blob[future].name} generated an exception: {exc}")

        # Check for completion of each day
        for day, files in downloaded_files.items():
            destination_files = [os.path.basename(file) for file in files]
            if all(file in destination_files for file in os.listdir(os.path.join(destination, day))):
                print(f"All files for {day} have been downloaded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download blobs from Azure container.")
    parser.add_argument("--limit", type=int, help="Limit the number of downloaded files.")
    parser.add_argument("--pkey-file", type=str, required=True, help="Path to the file containing the Azure authentication key for the given AccountName. In azure storage explorer, right click on the storage account (citprodc8603asa) then copy (primary) key.")
    parser.add_argument("--source", type=str, required=True, help="Source container name, e.g. 'databox-2483495b-0221-43b8-584a-444434c92e9c' ")
    parser.add_argument("--destination", type=str, required=True, help="Destination directory for downloaded files.")
    parser.add_argument("--account", type=str, required=True, help="AccountName, e.g. 'citprodc8603asa'.")
    args = parser.parse_args()

    with open(args.pkey_file, "r") as file:
        pkey = file.read().strip()

    CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName="+args.account+";AccountKey="+pkey+";EndpointSuffix=core.windows.net"
    del pkey

    SOURCE_CONTAINER_NAME = args.source

    blob_service_client = connect_to_blob_storage(CONNECTION_STRING)
    source_container_client = blob_service_client.get_container_client(SOURCE_CONTAINER_NAME)

    process_blobs_parallel(source_container_client,  destination=args.destination, limit=args.limit)



# /\ don't move these comments up, it will give you a syntax error...
#
# use like:
#cd 'C:/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/rapid-plankton/edge-ai'
#python ./blobloadprocess.py --account 'citprodc8603asa' --source 'databox-2483495b-0221-43b8-584a-444434c92e9c' --destination '/home/joe/Downloads/sampleimages/' --pkey-file '/home/joe/pkey.txt'
# will return images in the format: C:\Users\JR13\Downloads\sampleimages\2023-09-29\0827\RawImages\pia1.2023-09-29.0827+N00000000.tif
#
# Windows:
#cd 'C:/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/rapid-plankton/edge-ai'
#python blobloadprocess.py --account 'citprodc8603asa' --source 'databox-2483495b-0221-43b8-584a-444434c92e9c' --destination 'C:/Users/JR13/Downloads/sampleimages/' --pkey-file 'C:/Users/JR13/Documents/authenticationkeys/20240326_asa_plankton_pkey.txt' --limit 100

# note to self - How should I implement this? The gps is saved inside the file details. Once running from azure we will need the ability to directly get that from each blob object.
# 
# Change to this approach:
# az storage blob download-batch --destination downloads --source databox-2483495b-0221-43b8-584a-444434c92e9c --pattern 2023-09-28/*

