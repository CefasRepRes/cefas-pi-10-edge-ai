#I run this by copying and pasting the whole thing into miniforge prompt. First, get into your python env with the following line, minus the leading #:
#cd C:\Users\JR13\Documents\LOCAL_NOT_ONEDRIVE\rapid-plankton\edge-ai\env\Scripts && activate && cd .. && cd .. && python utility_scripts\grab_tarred_hitsmisses.py
from azure.identity import InteractiveBrowserCredential
from azure.storage.blob import BlobServiceClient
import os

import re

def safe_filename(name):
    # Replace any character that is not alphanumeric, dot, dash, or underscore
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


def extract_hitsmisses_from_tail(tar_tail_bytes, wanted_files=("HitsMisses.txt")):
    BLOCK_SIZE = 512
    i = 0
    results = {}
    while i + BLOCK_SIZE <= len(tar_tail_bytes):
        header = tar_tail_bytes[i:i+BLOCK_SIZE]
        name = header[:100].split(b'\0', 1)[0].decode(errors='ignore')
        if not name:
            i += BLOCK_SIZE
            continue
        size_str = header[124:136].decode('ascii', errors='ignore').strip('\0').strip()
        try:
            size = int(size_str, 8)
        except Exception:
            size = 0
        if any(name.endswith(wf) for wf in wanted_files):
            start = i + BLOCK_SIZE
            end = start + size
            results[name] = tar_tail_bytes[start:end]
        data_blocks = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
        i += BLOCK_SIZE + data_blocks * BLOCK_SIZE
    return results

def process_container_grab_hitsmisses(container_name, tail_bytes=0.1*1024*1024, output_dir="C:/Users/JR13/Downloads/hitsmisses_results"):
    credential = InteractiveBrowserCredential()
    account_url = "https://citprodc8603uksa.blob.core.windows.net"
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)
    container_client = blob_service_client.get_container_client(container_name)
    os.makedirs(output_dir, exist_ok=True)
    for blob in container_client.list_blobs():
        if not blob.name.lower().endswith('.tar'):
            continue
        print(f"Processing {blob.name} ...")
        blob_client = container_client.get_blob_client(blob.name)
        blob_size = blob.size
        # Download only the last tail_bytes of the blob
        offset = max(0, blob_size - tail_bytes)
        stream = blob_client.download_blob(offset=offset, length=tail_bytes)
        tar_tail = stream.readall()
        # Custom extraction from tail
        files = extract_hitsmisses_from_tail(tar_tail)
        if not files:
            print(f"  No target files found in tail of {blob.name}. Try increasing tail_bytes if needed.")
        for fname, content in files.items():
            # Only save if the filename is exactly one of the wanted files
            if fname in ("HitsMisses.txt"):
                out_path = os.path.join(output_dir, f"{safe_filename(blob.name+fname)}")
                with open(out_path, "wb") as out_f:
                    out_f.write(content)
                print(f"  Extracted {fname} ({len(content)} bytes)")

process_container_grab_hitsmisses("cend09-25-mpa-echannel-earlysummer")
