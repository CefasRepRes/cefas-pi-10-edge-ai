import os
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient
import concurrent.futures
from gps import extract_gps
import pandas as pd
import csv
import matplotlib.pyplot as plt



sas_file_path = 'C:/Users/JR13/Downloads/tmp/sas.txt'
storage_account_url = 'https://citdsdp2000bdataboxsa.blob.core.windows.net'  # Replace with your storage account URL
container_name = 'databox-b6b2f4a0-8963-4ca6-b511-a2ab660d00f0'  # Replace with your container name
destination_folder = r'C:/Users/JR13/Downloads/tmp'

with open(sas_file_path, 'r') as file:
    sas_token = file.read().strip()

blob_service_client = BlobServiceClient(account_url=storage_account_url, credential=sas_token)

def download_blob(blob_service_client, container_name, blob_path, download_path):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
    print(f"Downloading {blob_path} to {download_path}...")
    try:
        with open(download_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
    except Exception as e:
        print(f"Failed to download {blob_path}: {e}")

def download_files_for_days(blob_service_client, container_name, start_date, end_date):
    tasks = []
    current_date = start_date
    # Create a ThreadPoolExecutor with up to 100 concurrent downloaders
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        while current_date <= end_date:
            day_str = current_date.strftime('%Y-%m-%d')
            for hour in range(24):
                for minute in range(60):
                    blob_path = f"{day_str}/{hour:02d}{minute:02d}/Background.tif"
                    download_path = os.path.join(destination_folder, f"Background_{day_str}_{hour:02d}{minute:02d}.tif")
                    if not os.path.exists(destination_folder):
                        os.makedirs(destination_folder)
                    # Submit the download task to the executor
                    future = executor.submit(download_blob, blob_service_client, container_name, blob_path, download_path)
                    tasks.append(future)
            current_date += timedelta(days=1)
        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(tasks):
            future.result()

def clean_empty_files(folder_path):
    """Delete 0-byte files in the specified folder."""
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and os.path.getsize(file_path) == 0:
            os.remove(file_path)
            print(f"Deleted empty file: {file_path}")

def process_files(folder_path):
    """Process each file in the folder using extract_gps and append GPS info to a DataFrame."""
    
    with open("C:/Users/JR13/Downloads/tmp/gps.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Latitude", "Longitude", "Image DateTime"])
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.tif'):
                print(f"Processing file: {file_path}")
                gps_info = extract_gps(file_path)
                if gps_info:
                    writer.writerow(gps_info)
    print("GPS information has been saved to gps.csv")

def main():
    start_date = datetime(2024, 5, 16)
    end_date = datetime(2024, 5, 23)
    download_files_for_days(blob_service_client, container_name, start_date, end_date)
    
    # Clean up 0-byte files
    clean_empty_files(destination_folder)
    
    process_files(destination_folder)

    df = pd.read_csv('C:/Users/JR13/Downloads/tmp/gps.csv')
    df = df[df['Latitude']!='0.0']
    latitudes = pd.to_numeric(df['Latitude'], errors='coerce')
    longitudes = pd.to_numeric(df['Longitude'], errors='coerce')
    plt.figure(figsize=(10, 6))
    plt.plot(longitudes, latitudes, marker='.', linestyle='-', color='b')
    plt.title('Ship Route')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
