#!/usr/bin/env python3
"""I/O and dataframe utilities for ML prediction-result time series."""
from __future__ import annotations
import csv, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen
import pandas as pd

SUMMARY_RE = re.compile(r"^runs/(?P<run_name>[^/]+)/per_tar/(?P<date>\d{4}-\d{2}-\d{2})/(?P<hhmm>\d{4})/summary\.json$")
SUMMARY_PATH_RE = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})/(?P<hhmm>\d{4})/summary\.json$")
META_COLUMNS = {"timestamp", "run_name", "blob_name"}

def safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "value"

def find_class_counts(obj: Any) -> Optional[Dict[str, int]]:
    if isinstance(obj, dict):
        value=obj.get("class_counts")
        if isinstance(value, dict):
            cleaned={str(k): int(v) for k,v in value.items() if isinstance(k,str) and isinstance(v,(int,float))}
            if cleaned: return cleaned
        for child in obj.values():
            found=find_class_counts(child)
            if found: return found
    elif isinstance(obj,list):
        for child in obj:
            found=find_class_counts(child)
            if found: return found
    return None

def normalise_timeseries_rows_from_payload(payload: Any, source_name: str, timestamp: Optional[pd.Timestamp]=None) -> List[Dict[str,Any]]:
    row_timestamp=timestamp or pd.Timestamp.utcnow().normalize()
    base={"timestamp":row_timestamp,"run_name":source_name,"blob_name":source_name}
    if isinstance(payload,dict):
        counts=find_class_counts(payload)
        if counts: return [{**base,**counts}]
        for key in ("predictions","items","rows","data","results"):
            if payload.get(key) is not None:
                return normalise_timeseries_rows_from_payload(payload[key],source_name,row_timestamp)
        if any(payload.get(k) is not None for k in ("predicted_label","label","class","prediction")):
            payload=[payload]
    if isinstance(payload,list) and all(isinstance(x,dict) for x in payload):
        counts={}
        for record in payload:
            label=record.get("predicted_label") or record.get("label") or record.get("class") or record.get("prediction")
            if label not in (None,""):
                counts[str(label)]=counts.get(str(label),0)+1
        if counts: return [{**base,**counts}]
    return []

def parse_azure_blob_url(source: str) -> Optional[Tuple[str,str,str]]:
    parsed=urlparse(source)
    if parsed.scheme!="https" or not parsed.netloc.endswith(".blob.core.windows.net"): return None
    parts=[p for p in parsed.path.split("/") if p]
    if len(parts)<2: return None
    return parsed.netloc,parts[0],"/".join(parts[1:])

def get_blob_service_client(account_url: str, *, log_cb=None):
    try:
        from gui.application_validation.azure_utils import get_blob_service_client as factory
    except Exception as exc:
        raise RuntimeError("Azure Blob SDK dependencies are not available") from exc
    return factory(account_url,log_cb=log_cb)

def _canonical_summary_path(path: str) -> str:
    return (path or "").replace("\\", "/").strip().lstrip("/")


def parse_summary_timestamp(path: str) -> Optional[datetime]:
    normalized_path = _canonical_summary_path(path)
    match=SUMMARY_PATH_RE.search(normalized_path)    
    if not match: return None
    try:
        stamp=datetime.strptime(f"{match.group('date')} {match.group('hhmm')}","%Y-%m-%d %H%M")
        return stamp.replace(tzinfo=timezone.utc)
    except ValueError: return None

def parse_summary_run_name(path: str) -> Optional[str]:
    normalized_path = _canonical_summary_path(path)
    match=SUMMARY_RE.match(normalized_path)
    return match.group("run_name") if match else None

def load_json_payload(source: Any, *, blob_service_client=None) -> Any:
    if source is None: raise ValueError("No JSON source provided")
    if isinstance(source,(str,Path)):
        text=str(source)
        if text.startswith("http://"): raise ValueError("Plain HTTP URLs are not supported; use HTTPS")
        if text.startswith("https://"):
            blob=parse_azure_blob_url(text)
            if blob and blob_service_client is not None:
                _,container,name=blob
                raw=blob_service_client.get_container_client(container).download_blob(name).readall()
                return json.loads(raw.decode("utf-8"))
            with urlopen(text,timeout=30) as handle: return json.load(handle)
        path=Path(text)
        if path.exists(): return json.loads(path.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"JSON source does not exist: {text}")
    if hasattr(source,"read"): return json.load(source)
    raise TypeError(f"Unsupported JSON source type: {type(source)!r}")

def load_summary_rows_from_blob_container(client,container: str,prefix: str="",log_cb=None) -> List[Dict[str,Any]]:
    prefix=(prefix or "").strip().lstrip("/")
    cc=client.get_container_client(container)
    names=[]
    for blob in cc.list_blobs(name_starts_with=prefix):
        name=getattr(blob,"name",None) or (blob.get("name") if isinstance(blob,dict) else None)
        if name and str(name).lower().endswith("summary.json"): names.append(str(name).lstrip("/"))
    rows=[]
    for name in sorted(set(names)):
        try: payload=json.loads(cc.get_blob_client(name).download_blob().readall().decode("utf-8"))
        except Exception as exc:
            if log_cb: log_cb(f"Failed to read blob {container}/{name}: {exc}")
            continue
        rows += normalise_timeseries_rows_from_payload(payload,parse_summary_run_name(name) or name,parse_summary_timestamp(name))
    return rows

def load_summary_rows_from_local_dir(path: Path|str,log_cb=None) -> List[Dict[str,Any]]:
    root=Path(path)
    if not root.is_dir(): return []
    rows=[]
    for item in sorted(root.rglob("summary.json")):
        try: payload=json.loads(item.read_text(encoding="utf-8"))
        except Exception as exc:
            if log_cb: log_cb(f"Failed to read {item}: {exc}")
            continue
        rows += normalise_timeseries_rows_from_payload(payload,str(item),parse_summary_timestamp(str(item)))
    return rows

def load_inference_rows_from_source(source: Any, *, blob_service_client=None,log_cb=None) -> List[Dict[str,Any]]:
    if source is None: raise ValueError("No inference source provided")
    if isinstance(source,(str,Path)):
        text=str(source)
        if text.startswith("http://"): raise ValueError("Plain HTTP URLs are not supported; use HTTPS")
        if text.startswith("https://"):
            blob=parse_azure_blob_url(text)
            if blob:
                account,container,prefix=blob
                client=blob_service_client or get_blob_service_client(f"https://{account}",log_cb=log_cb)
                return load_summary_rows_from_blob_container(client,container,prefix,log_cb)
            return normalise_timeseries_rows_from_payload(load_json_payload(text),text)
        path=Path(text)
        if path.is_dir(): return load_summary_rows_from_local_dir(path,log_cb)
        if path.exists(): return normalise_timeseries_rows_from_payload(load_json_payload(path),text)
        raise FileNotFoundError(f"JSON source does not exist: {text}")
    if hasattr(source,"read"): return normalise_timeseries_rows_from_payload(load_json_payload(source),str(source))
    raise TypeError(f"Unsupported JSON source type: {type(source)!r}")

def build_dataframe(rows: List[Dict[str,Any]]) -> Tuple[pd.DataFrame,pd.DataFrame]:
    if not rows: raise RuntimeError("No usable class count rows found")
    df=pd.DataFrame(rows)
    classes=sorted(c for c in df.columns if c not in META_COLUMNS)
    for col in classes: df[col]=pd.to_numeric(df[col],errors="coerce").fillna(0).astype(int)
    df=df.sort_values(["timestamp","run_name","blob_name"]).reset_index(drop=True)
    result=df[["timestamp","run_name","blob_name"]+classes]
    return result,df

def write_long_counts_csv(df: pd.DataFrame,output_csv: Path) -> None:
    classes=[c for c in df.columns if c not in META_COLUMNS]
    df.melt(id_vars=["timestamp","run_name","blob_name"],value_vars=classes,var_name="class",value_name="count").to_csv(output_csv,index=False)

def write_long_uncertainty_csv(df: pd.DataFrame,output_csv: Path) -> None:
    suffixes=("_corrected_mean","_corrected_median")
    names=sorted({c.removesuffix(s) for c in df.columns for s in suffixes if c.endswith(s)})
    rows=[]
    for _,row in df.iterrows():
        for name in names:
            rows.append({"timestamp":row["timestamp"],"run_name":row["run_name"],"blob_name":row["blob_name"],"class":name,
              **{m:row.get(f"{name}_{m}") for m in ("corrected_mean","corrected_median","corrected_lower","corrected_upper")}})
    if rows: pd.DataFrame(rows).to_csv(output_csv,index=False)
