import os
import re
from datetime import datetime, timezone

try:
    from azure.identity import InteractiveBrowserCredential, TokenCachePersistenceOptions
except ImportError:  # pragma: no cover - supports older azure-identity versions
    from azure.identity import InteractiveBrowserCredential
    TokenCachePersistenceOptions = None

from azure.storage.blob import BlobServiceClient

from .constants import STORAGE_SCOPE

_CACHED_CREDENTIAL = None
_CACHED_BLOB_SERVICE_CLIENTS = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def interactive_credential(log_cb=None):
    """Return one cached InteractiveBrowserCredential for the current process.

    The first call opens the browser and forces token acquisition. Later calls reuse
    the same credential object, so application steps/workers do not repeatedly ask
    the user to authenticate.

    TokenCachePersistenceOptions also lets Azure Identity reuse a cached token where
    supported by the installed azure-identity/keyring setup.
    """
    global _CACHED_CREDENTIAL

    if _CACHED_CREDENTIAL is not None:
        if log_cb:
            log_cb("Using cached Azure credential.")
        return _CACHED_CREDENTIAL

    if log_cb:
        log_cb("Opening browser sign-in...")

    kwargs = {}
    if TokenCachePersistenceOptions is not None:
        kwargs["cache_persistence_options"] = TokenCachePersistenceOptions(
            name="rapid_plankton_cache",
            allow_unencrypted_storage=True,
        )

    cred = InteractiveBrowserCredential(**kwargs)
    cred.get_token(STORAGE_SCOPE)  # force browser once, at Step 1 when used there

    if log_cb:
        log_cb("Sign-in succeeded.")

    _CACHED_CREDENTIAL = cred
    return _CACHED_CREDENTIAL

import requests

from azure.core.pipeline.transport import RequestsTransport
from azure.storage.blob import BlobServiceClient
from requests.adapters import HTTPAdapter

# Should comfortably cover:
# - HITSMISSES_PROBE_WORKERS (4)
# - TIF_DOWNLOAD_WORKERS (8)
# - UPLOAD_MAX_CONCURRENCY (8)
#
# Increase if you substantially increase worker counts.
_CONNECTION_POOL_SIZE = 256


def get_blob_service_client(account_url: str, log_cb=None):
    """
    Return a cached BlobServiceClient for an account URL.

    This keeps all tab and worker operations on the same cached credential,
    session and urllib3 connection pool after Step 1 has authenticated.
    """

    account_url = (account_url or "").strip().rstrip("/")
    if not account_url:
        raise RuntimeError("Missing Azure Blob Storage account URL")

    bsc = _CACHED_BLOB_SERVICE_CLIENTS.get(account_url)
    if bsc is not None:
        return bsc

    cred = interactive_credential(log_cb=log_cb)

    # Workaround for Azure SDK issue #38054:
    # download_blob(max_concurrency=...) does NOT automatically increase
    # the underlying urllib3 connection pool size.
    session = requests.Session()

    adapter = HTTPAdapter(
        pool_connections=_CONNECTION_POOL_SIZE,
        pool_maxsize=_CONNECTION_POOL_SIZE,
        pool_block=True,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    transport = RequestsTransport(
        session=session,
        session_owner=True,
    )

    bsc = BlobServiceClient(
        account_url=account_url,
        credential=cred,
        transport=transport,
    )

    _CACHED_BLOB_SERVICE_CLIENTS[account_url] = bsc
    return bsc

def parse_blob_url(url: str):
    """Parse https://<acct>.blob.core.windows.net/<container>/<optional/path>."""
    url = (url or "").strip()
    if not url:
        return None
    url_no_q = url.split("?", 1)[0]
    m = re.match(r"^(https?://[^/]+)(?:/([^/]+)(?:/(.*))?)?$", url_no_q)
    if not m:
        return None
    account_url = m.group(1)
    container = m.group(2)
    path = (m.group(3) or "").lstrip("/")
    return account_url, container, path


def parse_model_input(txt: str):
    """Accept full model URL OR bare blob name like model<runid>.pt."""
    txt = (txt or "").strip()
    if not txt:
        return None
    if not txt.lower().startswith("http"):
        return txt if txt.lower().endswith(".pt") else None
    parsed = parse_blob_url(txt)
    if not parsed:
        return None
    _acct, container, path = parsed
    if container != "trainedmodels":
        return None
    return path


def extract_runid_from_model_blob(model_blob_name: str):
    base = os.path.basename(model_blob_name or "")
    m = re.match(r"^model(.+)\.pt$", base)
    return m.group(1) if m else None
