#!/usr/bin/env python3
import argparse, io, json, logging, os, re, tarfile, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Iterator, Tuple
from azure.core.exceptions import ResourceExistsError, ResourceModifiedError, ResourceNotFoundError, HttpResponseError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings
ACCOUNT_URL=os.getenv("AZURE_STORAGE_ACCOUNT_URL","https://citprodc8603uksa.blob.core.windows.net")
SOURCE_CONTAINERS=[x.strip() for x in os.getenv("SOURCE_CONTAINERS","cend08-24-nephrops-summer-northsea").split(",") if x.strip()]
STATE_CONTAINER=os.getenv("AZURE_STATE_CONTAINER","github-actions-tenbin-tar-state")
SURVEY_START_DATE=os.getenv("SURVEY_START_DATE","2024-05-19")
SURVEY_END_DATE=os.getenv("SURVEY_END_DATE","2024-05-23")
RUN_KEY=os.getenv("TENBIN_RUN_KEY","demo-cend08-2024-05-19-to-2024-05-23")
PROBE_EVERY_MINUTE=os.getenv("PROBE_EVERY_MINUTE","false").lower() in ("1","true","yes")
TIF_DOWNLOAD_WORKERS=int(os.getenv("TIF_DOWNLOAD_WORKERS","1"))
DOWNLOAD_MAX_CONCURRENCY_PER_BLOB=int(os.getenv("DOWNLOAD_MAX_CONCURRENCY_PER_BLOB","1"))
UPLOAD_MAX_CONCURRENCY=int(os.getenv("UPLOAD_MAX_CONCURRENCY","1"))
IMAGE_STEM_OVERRIDE=os.getenv("IMAGE_STEM","").strip()
IMAGE_START_INDEX_OVERRIDE=os.getenv("IMAGE_START_INDEX","").strip()
SOURCE_PREFIX=os.getenv("SOURCE_PREFIX","").strip().strip("/")
HITSMISSES_BASENAME="HitsMisses.txt"
RAW_IMAGES_DIR="RawImages"
IMAGE_STEM_FALLBACK="pi2"
TAR_CONTENT_TYPE="application/x-tar"
def setup_logging():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("azure").setLevel(logging.WARNING)
def get_bsc():
    return BlobServiceClient(account_url=ACCOUNT_URL,credential=DefaultAzureCredential(exclude_interactive_browser_credential=True))
def destination_container_name(source_container:str)->str:
    name=f"{source_container.lower()[:4]}-{source_container.lower()[4:]}"
    name=re.sub(r"[^a-z0-9-]","-",name)
    name=re.sub(r"-+","-",name).strip("-")
    if len(name)<3 or len(name)>63: raise ValueError(f"Invalid destination container name: {name}")
    return name
def iter_candidate_bins(start_date:str,end_date:str,every_minute:bool)->Iterator[Tuple[str,str]]:
    current=datetime.strptime(start_date,"%Y-%m-%d")
    final=datetime.strptime(end_date,"%Y-%m-%d")+timedelta(days=1)
    step=timedelta(minutes=1 if every_minute else 10)
    while current<final:
        yield current.strftime("%Y-%m-%d"),current.strftime("%H%M")
        current+=step
def hitsmisses_blob_name(date_text,bin_text): return f"{date_text}/{bin_text}/{HITSMISSES_BASENAME}"
def raw_images_prefix(date_text,bin_text): return f"{date_text}/{bin_text}/{RAW_IMAGES_DIR}/"
def with_source_prefix(path):
    if not SOURCE_PREFIX: return path
    normalized=path.lstrip("/")
    if normalized==SOURCE_PREFIX or normalized.startswith(f"{SOURCE_PREFIX}/"): return normalized
    return f"{SOURCE_PREFIX}/{normalized}"
def tif_blob_name(date_text,bin_text,index,image_stem=None):
    stem=image_stem or IMAGE_STEM_OVERRIDE or IMAGE_STEM_FALLBACK
    return f"{date_text}/{bin_text}/{RAW_IMAGES_DIR}/{stem}.{date_text}.{bin_text}+N{index:08d}.tif"
def tif_member_name(date_text,bin_text,index,image_stem=None): return tif_blob_name(date_text,bin_text,index,image_stem)
def output_tar_name(date_text,bin_text): return f"{date_text}/{bin_text}.tar"
def read_blob_bytes(cc,blob_name): return cc.get_blob_client(blob_name).download_blob(max_concurrency=DOWNLOAD_MAX_CONCURRENCY_PER_BLOB).readall()
def blob_exists(cc,blob_name):
    try:
        cc.get_blob_client(blob_name).get_blob_properties(); return True
    except ResourceNotFoundError: return False
def parse_hitsmisses(data:bytes):
    counts=[]
    for raw in data.decode("utf-8-sig",errors="replace").splitlines():
        line=raw.strip()
        if line: counts.append(int(line.split(",",1)[0].strip()))
    return counts,sum(counts)
def ensure_container(bsc,name):
    try: bsc.create_container(name); logging.info("CONTAINER_CREATED name=%s",name)
    except ResourceExistsError: pass
    return bsc.get_container_client(name)
def manifest_blob(): return f"runs/{RUN_KEY}/manifest.jsonl"
def state_key(item): return f"{item['source_container']}__{item['date_text']}__{item['bin_text']}".replace("/","_")
def done_blob(key): return f"runs/{RUN_KEY}/done/{key}.json"
def failed_blob(key): return f"runs/{RUN_KEY}/failed/{key}.json"
def lock_blob(key): return f"runs/{RUN_KEY}/locks/{key}.lock"
def parse_image_name(blob_name,date_text,bin_text):
    base=os.path.basename(blob_name)
    pattern=rf"^(.+)\.{re.escape(date_text)}\.{re.escape(bin_text)}\+N(\d+)\.tiff?$"
    m=re.match(pattern,base,flags=re.IGNORECASE)
    if not m: return None
    return m.group(1),int(m.group(2))
def detect_image_naming(source_cc,date_text,bin_text,expected_tifs=None,max_samples=10000):
    if IMAGE_STEM_OVERRIDE:
        start=int(IMAGE_START_INDEX_OVERRIDE) if IMAGE_START_INDEX_OVERRIDE else 0
        logging.info("IMAGE_NAMING_ENV_OVERRIDE date=%s bin=%s stem=%s start_index=%s",date_text,bin_text,IMAGE_STEM_OVERRIDE,start)
        return IMAGE_STEM_OVERRIDE,start
    prefix=with_source_prefix(raw_images_prefix(date_text,bin_text))
    stems={}
    first_names=[]
    scanned=0
    logging.info("IMAGE_NAMING_DETECT_BEGIN date=%s bin=%s prefix=%s",date_text,bin_text,prefix)
    for b in source_cc.list_blobs(name_starts_with=prefix):
        name=getattr(b,"name","")
        if not name.lower().endswith((".tif",".tiff")): continue
        scanned+=1
        if len(first_names)<20: first_names.append(name)
        parsed=parse_image_name(name,date_text,bin_text)
        if parsed:
            stem,index=parsed
            stems.setdefault(stem,[]).append(index)
        if scanned>=max_samples: break
    logging.info("IMAGE_NAMING_DETECT_SAMPLES date=%s bin=%s scanned=%s samples=%s",date_text,bin_text,scanned,json.dumps(first_names))
    if not stems:
        if IMAGE_START_INDEX_OVERRIDE:
            start=int(IMAGE_START_INDEX_OVERRIDE)
        else:
            start=0
        logging.warning("IMAGE_NAMING_DETECT_NONE date=%s bin=%s prefix=%s fallback_stem=%s fallback_start_index=%s",date_text,bin_text,prefix,IMAGE_STEM_FALLBACK,start)
        return IMAGE_STEM_FALLBACK,start        
    best_stem=max(stems,key=lambda s:len(stems[s]))
    best_indices=sorted(stems[best_stem])
    start=int(IMAGE_START_INDEX_OVERRIDE) if IMAGE_START_INDEX_OVERRIDE else 0        
    if IMAGE_START_INDEX_OVERRIDE:
        start=int(IMAGE_START_INDEX_OVERRIDE)
    logging.info("IMAGE_NAMING_DETECT_OK date=%s bin=%s stem=%s start_index=%s matched=%s scanned=%s expected_tifs=%s",date_text,bin_text,best_stem,start,len(best_indices),scanned,expected_tifs)
    return best_stem,start
def discover():
    setup_logging(); bsc=get_bsc(); state_cc=ensure_container(bsc,STATE_CONTAINER); rows=[]
    for source_container in SOURCE_CONTAINERS:
        source_cc=bsc.get_container_client(source_container); dest_container=destination_container_name(source_container); ensure_container(bsc,dest_container)
        survey_image_stem=None; survey_image_start_index=None
        for date_text,bin_text in iter_candidate_bins(SURVEY_START_DATE,SURVEY_END_DATE,PROBE_EVERY_MINUTE):
            hm=hitsmisses_blob_name(date_text,bin_text)
            try: hm_bytes=read_blob_bytes(source_cc,with_source_prefix(hm))
            except ResourceNotFoundError: continue
            _,expected_tifs=parse_hitsmisses(hm_bytes)
            if survey_image_stem is None:
                survey_image_stem,survey_image_start_index=detect_image_naming(source_cc,date_text,bin_text,expected_tifs=expected_tifs)
                logging.info("SURVEY_IMAGE_NAMING source=%s stem=%s start_index=%s",source_container,survey_image_stem,survey_image_start_index)
            rows.append({"source_container":source_container,"destination_container":dest_container,"date_text":date_text,"bin_text":bin_text,"hitsmisses_blob":hm,"output_tar":output_tar_name(date_text,bin_text),"expected_tifs":expected_tifs,"image_stem":survey_image_stem,"image_start_index":survey_image_start_index})
            logging.info("DISCOVERED source=%s date=%s bin=%s expected_tifs=%s image_stem=%s image_start_index=%s",source_container,date_text,bin_text,expected_tifs,survey_image_stem,survey_image_start_index)
    state_cc.get_blob_client(manifest_blob()).upload_blob(("\n".join(json.dumps(r,sort_keys=True) for r in rows)+"\n").encode(),overwrite=True,content_settings=ContentSettings(content_type="application/jsonl"))
    logging.info("MANIFEST_UPLOADED blob=%s rows=%s",manifest_blob(),len(rows))
def load_manifest(state_cc): return [json.loads(x) for x in state_cc.get_blob_client(manifest_blob()).download_blob().readall().decode().splitlines() if x.strip()]
def try_acquire_lock(state_cc,key):
    bc=state_cc.get_blob_client(lock_blob(key))
    try:
        bc.get_blob_properties()
    except ResourceNotFoundError:
        try:
            bc.upload_blob(b"",overwrite=False)
        except ResourceExistsError:
            pass
        except HttpResponseError as e:
            error_code=getattr(e,"error_code","") or ""
            status_code=getattr(e,"status_code",None)
            if status_code not in (409,412) and error_code not in ("BlobAlreadyExists","LeaseIdMissing","LeaseAlreadyPresent","LeaseAlreadyAcquired"):
                raise
    try:
        return bc.acquire_lease(lease_duration=60)
    except ResourceModifiedError:
        return None
    except HttpResponseError as e:
        error_code=getattr(e,"error_code","") or ""
        status_code=getattr(e,"status_code",None)
        if status_code in (409,412) or error_code in ("LeaseAlreadyPresent","LeaseAlreadyAcquired","LeaseIdMissing"):
            return None
        raise
def start_lease_renewer(lease,stop_event):
    def run():
        while not stop_event.wait(30):
            try: lease.renew()
            except Exception: logging.exception("LEASE_RENEW_FAILED"); return
    t=threading.Thread(target=run,daemon=True); t.start(); return t
def add_bytes_to_tar(tar,member_name,data):
    info=tarfile.TarInfo(member_name); info.size=len(data); info.mtime=int(time.time()); tar.addfile(info,io.BytesIO(data))
def download_one_tif(source_cc,date_text,bin_text,index,image_stem):
    blob=tif_blob_name(date_text,bin_text,index,image_stem); return index,blob,read_blob_bytes(source_cc,with_source_prefix(blob))
def probe_tif_blob(source_cc,key,date_text,bin_text,index,image_stem):
    probe_blob=with_source_prefix(tif_blob_name(date_text,bin_text,index,image_stem))
    try:
        source_cc.get_blob_client(probe_blob).get_blob_properties()
        logging.info("IMAGE_PROBE_OK key=%s blob=%s",key,probe_blob)
        return True
    except ResourceNotFoundError:
        logging.error("IMAGE_PROBE_MISSING key=%s blob=%s stem=%s first_index=%s",key,probe_blob,image_stem,index)
        return False

def process_item(bsc,state_cc,item,worker_id):
    key=state_key(item); dest_cc=bsc.get_container_client(item["destination_container"]); source_cc=bsc.get_container_client(item["source_container"])
    if blob_exists(state_cc,done_blob(key)): return "done_exists"
    if blob_exists(dest_cc,item["output_tar"]):
        state_cc.get_blob_client(done_blob(key)).upload_blob(json.dumps({"status":"already_uploaded","item":item}).encode(),overwrite=True); return "tar_exists_marked_done"
    lease=try_acquire_lock(state_cc,key)
    if lease is None: return "locked_elsewhere"
    stop=threading.Event(); start_lease_renewer(lease,stop); started=time.time(); local_path=None
    try:
        logging.info("CLAIMED worker=%s key=%s",worker_id,key)
        expected=int(item["expected_tifs"]); hm_bytes=read_blob_bytes(source_cc,with_source_prefix(item["hitsmisses_blob"]))
        image_stem=item.get("image_stem") or None
        image_start_index=item.get("image_start_index")
        if image_stem is None or image_start_index is None:
            image_stem,image_start_index=detect_image_naming(source_cc,item["date_text"],item["bin_text"],expected_tifs=expected)
        image_start_index=int(image_start_index)
        first_index=image_start_index
        last_index=image_start_index+expected-1
        logging.info("IMAGE_NAMING_USED key=%s stem=%s first_index=%s last_index=%s expected_tifs=%s",key,image_stem,first_index,last_index,expected)
        if not probe_tif_blob(source_cc,key,item["date_text"],item["bin_text"],first_index,image_stem):
            try:
                state_cc.get_blob_client(failed_blob(key)).upload_blob(json.dumps({"status":"failed","reason":"image_probe_missing","expected_tifs":expected,"image_stem":image_stem,"image_start_index":image_start_index,"item":item},indent=2).encode(),overwrite=True)
            except Exception:
                logging.exception("FAILED_BLOB_WRITE_FAILED key=%s",key)
            return "failed"
        fd,local_path=tempfile.mkstemp(prefix="tenbin_",suffix=".tar"); os.close(fd)
        tif_count=0; failures=[]; payload=len(hm_bytes)
        with tarfile.open(local_path,"w") as tar:
            with ThreadPoolExecutor(max_workers=TIF_DOWNLOAD_WORKERS) as pool:
                futures=[pool.submit(download_one_tif,source_cc,item["date_text"],item["bin_text"],i,image_stem) for i in range(first_index,last_index+1)]
                for fut in as_completed(futures):
                    try: idx,_,data=fut.result()
                    except Exception as e: failures.append(repr(e)); continue
                    add_bytes_to_tar(tar,tif_member_name(item["date_text"],item["bin_text"],idx,image_stem),data); tif_count+=1; payload+=len(data)
                    if tif_count%1000==0: logging.info("TIF_PROGRESS key=%s downloaded=%s/%s",key,tif_count,expected)
            add_bytes_to_tar(tar,HITSMISSES_BASENAME,hm_bytes)
        if failures or tif_count!=expected:
            state_cc.get_blob_client(failed_blob(key)).upload_blob(json.dumps({"status":"failed","tif_count":tif_count,"expected_tifs":expected,"image_stem":image_stem,"image_start_index":image_start_index,"failures":failures[:20],"item":item},indent=2).encode(),overwrite=True); return "failed"
        with open(local_path,"rb") as fh:
            dest_cc.get_blob_client(item["output_tar"]).upload_blob(fh,overwrite=False,max_concurrency=UPLOAD_MAX_CONCURRENCY,content_settings=ContentSettings(content_type=TAR_CONTENT_TYPE))
        state_cc.get_blob_client(done_blob(key)).upload_blob(json.dumps({"status":"uploaded","tif_count":tif_count,"payload_bytes":payload,"tar_bytes":os.path.getsize(local_path),"elapsed_s":time.time()-started,"worker_id":worker_id,"image_stem":image_stem,"image_start_index":image_start_index,"hitsmisses_member_last":True,"item":item},indent=2).encode(),overwrite=True)
        return "uploaded"
    finally:
        stop.set()
        try: lease.release()
        except Exception: pass
        if local_path and os.path.exists(local_path): os.remove(local_path)
def worker():
    setup_logging(); worker_id=os.getenv("WORKER_ID","0"); bsc=get_bsc(); state_cc=bsc.get_container_client(STATE_CONTAINER); processed=0
    for item in load_manifest(state_cc):
        result=process_item(bsc,state_cc,item,worker_id); logging.info("WORKER_RESULT worker=%s date=%s bin=%s result=%s",worker_id,item.get("date_text"),item.get("bin_text"),result)
        if result in ("uploaded","failed","tar_exists_marked_done"): processed+=1
    logging.info("WORKER_END worker=%s processed=%s",worker_id,processed)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["discover","worker"],required=True); args=p.parse_args(); discover() if args.mode=="discover" else worker()
if __name__=="__main__": main()
