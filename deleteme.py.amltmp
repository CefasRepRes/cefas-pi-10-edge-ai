from azureml.core import Workspace, Datastore
from azure.storage.blob import BlobServiceClient
import pandas as pd
from io import StringIO
import os

print('starting')

# Initialize Azure ML workspace and datastore
ws = Workspace.get(name="citdsdp4000nc6s-mlw",
                   subscription_id="25c7e1b1-ff04-418b-842b-29a58e065da7",
                   resource_group="CIT-DS-RG-DP4000-NC6S")

container_name = 'databox-b6b2f4a0-8963-4ca6-b511-a2ab660d00f0'
datastore_name = 'databoxb6b2f4a089634ca6b511a2ab660d00f0'

#container_name = 'databox-2483495b-0221-43b8-584a-444434c92e9c'
#datastore_name = 'databox2483495b022143b8584a444434c92e9c'


datastore = Datastore.get(ws, datastore_name)

# Azure Blob Storage details
blob_account_name = datastore.account_name
blob_account_token = datastore.sas_token

blob_service_client = BlobServiceClient(
    account_url=f"https://{blob_account_name}.blob.core.windows.net",
    credential=blob_account_token
)


container_client = blob_service_client.get_container_client(container_name)

# List blobs and process text files
blobs = container_client.list_blobs()
summary_data = []
#print(len(blobs))

i=0
for blob in blobs:
    if blob.name.endswith('HitsMisses.txt'):
        try:
            blob_client = container_client.get_blob_client(blob.name)
            download_stream = blob_client.download_blob()
            data = download_stream.readall().decode('utf-8')
            df = pd.read_csv(StringIO(data), header=None, names=['Hits', 'Misses'])
            total_hits = df['Hits'].tolist()
            total_misses = df['Misses'].tolist()
            parts = blob.name.split('/')
            if len(parts) >= 2:
                date = [parts[0]] * len(df)
                tenbin = [parts[1]] * len(df)
                minute = list(range(len(df)))
                summary_data.extend(zip(date, tenbin, minute, total_hits, total_misses))
                print(f"Added file to dataframe: {blob.name}")
            else:
                print(f"Blob name does not have expected format: {blob.name}")
        except Exception as e:
            print(f"Error processing blob {blob.name}: {e}")
summary_df = pd.DataFrame(summary_data, columns=['Date', 'Tenbin', 'Minute', 'Hits', 'Misses'])
summary_df.index.name = 'Index'

csv_buffer = StringIO()
summary_df.to_csv(csv_buffer)

output_blob_name = 'survey_all_hitsmisses.csv'
output_blob_client = container_client.get_blob_client(output_blob_name)
output_blob_client.upload_blob(csv_buffer.getvalue(), blob_type="BlockBlob", overwrite=True)

print(f"Summary CSV saved as {output_blob_name} in {container_name} container.")