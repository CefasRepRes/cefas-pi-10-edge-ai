import os
import sys
import json
import hashlib
import subprocess
from io import BytesIO
from datetime import datetime, timezone
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QListWidget, QListWidgetItem, QAbstractItemView, QProgressBar, QGroupBox,
    QLineEdit, QFileDialog, QComboBox
)
from azure.storage.blob import BlobServiceClient
from azure.storage.blob import ContentSettings
import yaml
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
EDGE_AI_DIR = os.path.dirname(GUI_DIR)
if EDGE_AI_DIR not in sys.path:
    sys.path.insert(0, EDGE_AI_DIR)
try:
    from application_validation.azure_utils import get_blob_service_client
except ImportError:
    from azure_utils import get_blob_service_client
import gps
import exif
SUPPORTED_MODEL_ARCHITECTURES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
]
DEFAULT_MODEL_ARCHITECTURE = "mobilenet_v3_small"
def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
def get_git_info(repo_dir):
    """Return (sha, dirty) or (None, None) if not a git repo."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_dir, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_dir, text=True).strip())
        return sha, dirty
    except Exception:
        return None, None
def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()
def sha256_stream(chunks_iter) -> str:
    h = hashlib.sha256()
    for ch in chunks_iter:
        h.update(ch)
    return h.hexdigest()
def normalise_model_architecture(arch: str) -> str:
    arch = str(arch or DEFAULT_MODEL_ARCHITECTURE).strip().lower().replace("-", "_")
    aliases = {
        "resnet": "resnet18",
        "resnet18": "resnet18",
        "resnet34": "resnet34",
        "resnet50": "resnet50",
        "mobilenet": "mobilenet_v3_small",
        "mobilenetv2": "mobilenet_v2",
        "mobilenet_v2": "mobilenet_v2",
        "mobilenetv3": "mobilenet_v3_small",
        "mobilenetv3_small": "mobilenet_v3_small",
        "mobilenet_v3_small": "mobilenet_v3_small",
        "mobilenetv3_large": "mobilenet_v3_large",
        "mobilenet_v3_large": "mobilenet_v3_large",
    }
    arch = aliases.get(arch, arch)
    if arch not in SUPPORTED_MODEL_ARCHITECTURES:
        raise RuntimeError(f"Unsupported model architecture: {arch}. Supported: {', '.join(SUPPORTED_MODEL_ARCHITECTURES)}")
    return arch
def set_architecture_in_training_defaults(defaults: dict, architecture: str) -> dict:
    """Set architecture in common config locations without removing existing defaults."""
    architecture = normalise_model_architecture(architecture)
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise RuntimeError("training_defaults.yaml must contain a dictionary at the top level")
    defaults["model_architecture"] = architecture
    defaults["arch"] = architecture
    model_cfg = defaults.setdefault("model", {})
    if isinstance(model_cfg, dict):
        model_cfg["architecture"] = architecture
        model_cfg["arch"] = architecture
        model_cfg["name"] = architecture
    training_cfg = defaults.setdefault("training", {})
    if isinstance(training_cfg, dict):
        training_cfg["model_architecture"] = architecture
        training_cfg["arch"] = architecture
    return defaults
def extract_instrument_serial_from_bytes(image_bytes: bytes):
    """
    Attempt to extract an instrument serial from EXIF tags.
    TIFF EXIF is not always standardised; we scan keys that look like serial number fields.
    We return (serial_value_or_None, serial_candidates_dict).
    """
    try:
        import exifread
        tags = exifread.process_file(BytesIO(image_bytes), details=False)
        candidates = {}
        for k, v in tags.items():
            lk = k.lower()
            if "serial" in lk or ("body" in lk and "serial" in lk) or ("camera" in lk and "serial" in lk):
                candidates[k] = str(v)
        serial = None
        if candidates:
            preferred = [k for k in candidates if "body" in k.lower() and "serial" in k.lower()]
            if not preferred:
                preferred = [k for k in candidates if "serial" in k.lower()]
            serial = candidates[preferred[0]]
        return serial, candidates
    except Exception:
        return None, {}
SDK_MAX_CONCURRENCY = 1
MAX_SINGLE_GET_SIZE = 256 * 1024 * 1024
MAX_CHUNK_GET_SIZE = 16 * 1024 * 1024
THREADS = 16
def safe_local_name(blob_name: str) -> str:
    base = os.path.basename(blob_name)
    pathhash = hashlib.sha256(blob_name.encode("utf-8")).hexdigest()[:12]
    return f"{pathhash}_{base}"
def download_and_parse_one(container_client, blob_name, cache_dir, log_cb=None):
    blob_client = container_client.get_blob_client(blob_name)
    downloader = blob_client.download_blob(max_concurrency=SDK_MAX_CONCURRENCY)
    local_name = safe_local_name(blob_name)
    local_path = os.path.join(cache_dir, local_name)
    h = hashlib.sha256()
    with open(local_path, "wb") as f:
        for ch in downloader.chunks():
            h.update(ch)
            f.write(ch)
    file_sha = h.hexdigest()
    parts = blob_name.split("/")
    root = parts[0]
    class_label = parts[1] if len(parts) > 2 else "UNKNOWN"
    data = b""
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        _, _, dt = exif.getexif(data)
    except Exception:
        dt = None
    try:
        lat, lon, _img_dt = gps.extract_gps(local_path)
        if lat == "error" or lon == "error":
            lat, lon = None, None
    except Exception:
        lat, lon = None, None
    serial, candidates = extract_instrument_serial_from_bytes(data)
    return {
        "blob_path": blob_name,
        "root": root,
        "class_label": class_label,
        "sha256": file_sha,
        "exif_datetime": dt,
        "gps": {"lat": lat, "lon": lon},
        "instrument_serial": serial,
        "serial_candidates": candidates,
        "local_path": local_path,
    }
class BlobScanWorker(QThread):
    log = Signal(str)
    progress = Signal(int)
    roots_ready = Signal(list)
    result_ready = Signal(dict)
    def __init__(self, account_url, container_name, selected_roots, output_dir, repo_dir, model_architecture=DEFAULT_MODEL_ARCHITECTURE, parent=None):
        super().__init__(parent)
        self.account_url = account_url
        self.container_name = container_name
        self.selected_roots = selected_roots or []
        self.output_dir = output_dir
        self.repo_dir = repo_dir
        self.model_architecture = normalise_model_architecture(model_architecture)
        self._stop = False
    def stop(self):
        self._stop = True
    def _make_blob_service_client(self):
        """Create/reuse a BlobServiceClient for the training workflow."""
        conn = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        if conn:
            self.log.emit("Using AZURE_STORAGE_CONNECTION_STRING for BlobServiceClient.")
            return BlobServiceClient.from_connection_string(conn)
        return get_blob_service_client(self.account_url, log_cb=self.log.emit)
    def list_top_level_roots(self):
        bsc = self._make_blob_service_client()
        container = bsc.get_container_client(self.container_name)
        roots = []
        for item in container.walk_blobs(name_starts_with="", delimiter="/"):
            if hasattr(item, "name"):
                root = item.name.rstrip("/")
                if root:
                    roots.append(root)
        return sorted(set(roots))
    def run(self):
        try:
            bsc = self._make_blob_service_client()
            container = bsc.get_container_client(self.container_name)
            if not self.selected_roots:
                self.log.emit("No selected roots provided; only listing roots.")
                roots = self.list_top_level_roots()
                self.roots_ready.emit(roots)
                return
            run_id = f"{utc_now_iso().replace(':','-')}"
            git_sha, git_dirty = get_git_info(self.repo_dir)
            settings = {
                "schema_version": "1.0",
                "training_run_id": run_id,
                "generated_utc": utc_now_iso(),
                "code": {"git_sha": git_sha, "dirty": git_dirty},
                "dataset_selection": {
                    "account_url": self.account_url,
                    "container": self.container_name,
                    "selected_roots": list(self.selected_roots),
                    "selection_timestamp_utc": utc_now_iso(),
                },
                "dataset_fingerprints": {},
                "instrument": {
                    "serial_number": None,
                    "source": "EXIF",
                    "consistency_check": None,
                    "serial_candidates_seen": {},
                },
                "files": [],
                "dataset_summary": {"total_images": 0, "class_totals": {}},
                "training_parameters": {
                    "model_architecture": self.model_architecture,
                    "arch": self.model_architecture,
                    "image_size": [256, 256],
                    "batch_size": 64,
                    "epochs": 50,
                    "learning_rate": 3e-4,
                },
                "outputs": {},
            }
            run_dir = os.path.join(self.output_dir, f"run_{run_id}")
            cache_dir = os.path.join(run_dir, "cache_images")
            os.makedirs(cache_dir, exist_ok=True)
            settings["outputs"]["run_dir"] = run_dir
            settings["outputs"]["cache_dir"] = cache_dir
            settings["model_architecture"] = self.model_architecture
            serials = []
            class_counter = Counter()
            per_root_lines = defaultdict(list)
            tif_blobs = []
            for root in self.selected_roots:
                prefix = root.rstrip("/") + "/"
                for blob in container.list_blobs(name_starts_with=prefix):
                    if self._stop:
                        self.log.emit("Stopped.")
                        return
                    if blob.name.lower().endswith((".tif", ".tiff")):
                        tif_blobs.append(blob.name)
                dataset_json_blob = f"{root.rstrip('/')}/dataset.json"
                try:
                    blob_client = container.get_blob_client(dataset_json_blob)
                    data = blob_client.download_blob().readall()
                    dataset_meta = json.loads(data)
                    self.log.emit(f"Loaded dataset.json from {dataset_json_blob}")
                    settings.setdefault("dataset_metadata_by_root", {})[root] = {
                        "blob_path": dataset_json_blob,
                        "content": dataset_meta,
                    }
                    settings["dataset_metadata"] = {
                        "blob_path": dataset_json_blob,
                        "content": dataset_meta,
                    }
                except Exception as err:
                    self.log.emit(f"No valid dataset.json found at {dataset_json_blob}: {err}")
            total = max(len(tif_blobs), 1)
            self.log.emit(f"Found {len(tif_blobs)} TIFFs across selected roots.")
            results = []
            with ThreadPoolExecutor(max_workers=THREADS) as pool:
                future_map = {pool.submit(download_and_parse_one, container, bn, cache_dir): bn for bn in tif_blobs}
                done = 0
                for fut in as_completed(future_map):
                    bn = future_map[fut]
                    try:
                        r = fut.result()
                        results.append(r)
                    except Exception as e:
                        self.log.emit(f"ERROR processing {bn}: {e}")
                        continue
                    done += 1
                    self.progress.emit(int((done / total) * 100))
                    if done % 10 == 0 or done == total:
                        self.log.emit(f"Please wait for download and fingerprint of training dataset .tiff files {done}/{total}")
            for r in results:
                settings["files"].append({
                    "blob_path": r["blob_path"],
                    "class_label": r["class_label"],
                    "sha256": r["sha256"],
                    "exif_datetime": r["exif_datetime"],
                    "gps": r["gps"],
                    "local_path": r.get("local_path"),
                })
                class_counter[r["class_label"]] += 1
                if r.get("instrument_serial"):
                    serials.append(r["instrument_serial"])
                per_root_lines[r["root"]].append(f"{r['sha256']}  {r['blob_path']}")
                if r.get("serial_candidates"):
                    settings["instrument"]["serial_candidates_seen"][r["blob_path"]] = r["serial_candidates"]
            if serials:
                serial_counts = Counter(serials)
                settings["instrument"]["serial_number"] = serial_counts.most_common(1)[0][0]
                settings["instrument"]["consistency_check"] = "single_serial" if len(serial_counts) == 1 else "multiple_serials_seen"
                settings["instrument"]["serial_counts"] = dict(serial_counts)
            for root, lines in per_root_lines.items():
                joined = "\n".join(sorted(lines)).encode("utf-8")
                settings["dataset_fingerprints"][root] = hashlib.sha256(joined).hexdigest()
            settings["dataset_summary"]["total_images"] = len(settings["files"])
            settings["dataset_summary"]["class_totals"] = dict(class_counter)
            settings["runid"] = run_id
            os.makedirs(self.output_dir, exist_ok=True)
            out_path = os.path.join(self.output_dir, f"modeltrainsettings_{run_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2)
            self.log.emit(f"Selected model architecture: {self.model_architecture}")
            self.log.emit("This code currently uses training_defaults.yaml, with the GUI-selected architecture injected before training.")
            self.log.emit(f"Wrote modeltrainsettings JSON: {out_path}")
            self.log.emit("You are now ready for step 3: train model (see button below).")
            self.result_ready.emit(settings)
        except Exception as e:
            self.log.emit(f"ERROR: {e}")
class ModelTrainWorker(QThread):
    log = Signal(str)
    finished_ok = Signal(dict)
    failed = Signal(str)
    def __init__(self, settings: dict, run_dir: str, model_architecture=DEFAULT_MODEL_ARCHITECTURE, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.run_dir = run_dir
        self.model_architecture = normalise_model_architecture(model_architecture)
    def run(self):
        try:
            self.log.emit("Loading training defaults...")
            defaults_path = os.path.join(os.path.dirname(__file__), "training_defaults.yaml")
            with open(defaults_path, "r", encoding="utf-8") as f:
                defaults = yaml.safe_load(f)
            selected_arch = normalise_model_architecture(
                self.model_architecture
                or self.settings.get("model_architecture")
                or (self.settings.get("training_parameters") or {}).get("model_architecture")
                or DEFAULT_MODEL_ARCHITECTURE
            )
            defaults = set_architecture_in_training_defaults(defaults, selected_arch)
            self.settings["training_parameters"] = defaults
            self.settings["model_architecture"] = selected_arch
            self.settings.setdefault("outputs", {})["run_dir"] = self.run_dir
            os.makedirs(self.run_dir, exist_ok=True)
            settings_path = os.path.join(self.run_dir, "modeltrainsettings.json")
            with open(settings_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            self.log.emit(f"Launching training as a subprocess with architecture: {selected_arch}")
            train_script = os.path.join(os.path.dirname(__file__), "..", "train.py")
            model_path = os.path.join(self.run_dir, defaults["output"]["model_filename"])
            cmd = [sys.executable, train_script, "--settings", settings_path, "--output", model_path]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in proc.stdout:
                self.log.emit(line.rstrip())
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"Training failed with exit code {rc}")
            self.log.emit("Training complete.")
            self.log.emit("Uploading artifacts...")
            self._upload_outputs(model_path, settings_path, self.settings["runid"])
            self.finished_ok.emit(self.settings)
        except Exception as e:
            self.failed.emit(str(e))
    def _upload_outputs(self, model_path, settings_path, runid):
        account_url = ((self.settings.get("dataset_selection") or {}).get("account_url") or "https://citprodc8603uksa.blob.core.windows.net")
        container = "trainedmodels"
        self.log.emit("Reusing cached Azure authentication for upload...")
        bsc = get_blob_service_client(account_url, log_cb=self.log.emit)
        container_client = bsc.get_container_client(container)
        artifacts = [model_path, settings_path]
        report_path = os.path.join(os.path.dirname(model_path), "training_report.json")
        if os.path.exists(report_path):
            artifacts.append(report_path)
        try:
            tp = self.settings.get("training_parameters") or {}
            trn = tp.get("training") or {}
            lg = trn.get("logging") or {}
        except Exception:
            lg = {}
        for key, fallback in (
            ("curve_csv", "training_curve.csv"),
            ("curve_png", "training_curve.png"),
            ("confusion_matrix_csv", "confusion_matrix.csv"),
            ("confusion_matrix_png", "confusion_matrix.png"),
        ):
            fn = lg.get(key, fallback)
            if not fn or str(fn).lower() in ("none", "null", "false", "off"):
                continue
            p = fn if os.path.isabs(str(fn)) else os.path.join(os.path.dirname(model_path), str(fn))
            if os.path.exists(p):
                artifacts.append(p)
        seen = set()
        artifacts = [a for a in artifacts if (a not in seen and not seen.add(a))]
        for local_path in artifacts:
            base_name = os.path.basename(local_path)
            extension = os.path.splitext(base_name)[1]
            name_no_ext = os.path.splitext(base_name)[0]
            blob_name = f"{name_no_ext}{runid}{extension}"
            self.log.emit(f"Uploading {blob_name}...")
            with open(local_path, "rb") as f:
                container_client.upload_blob(
                    name=blob_name,
                    data=f,
                    overwrite=True,
                    content_settings=ContentSettings(
                        content_type="application/json" if blob_name.endswith(".json") else "application/octet-stream"
                    ),
                )
class ModelTrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.account_url_default = "https://citprodc8603uksa.blob.core.windows.net"
        self.container_default = "training-libs"
        self.repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.train_btn = QPushButton("3: Train model")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self.start_training)
        layout = QVBoxLayout(self)
        storage_box = QGroupBox("Azure Blob Storage")
        storage_layout = QHBoxLayout(storage_box)
        self.account_url = QLineEdit(self.account_url_default)
        self.container_name = QLineEdit(self.container_default)
        storage_layout.addWidget(QLabel("Account URL:"))
        storage_layout.addWidget(self.account_url)
        storage_layout.addWidget(QLabel("Container:"))
        storage_layout.addWidget(self.container_name)
        layout.addWidget(storage_box)
        model_box = QGroupBox("Model architecture")
        model_layout = QHBoxLayout(model_box)
        self.model_architecture = QComboBox()
        self.model_architecture.addItems(SUPPORTED_MODEL_ARCHITECTURES)
        self.model_architecture.setCurrentText(DEFAULT_MODEL_ARCHITECTURE)
        model_layout.addWidget(QLabel("Architecture:"))
        model_layout.addWidget(self.model_architecture)
        model_layout.addWidget(QLabel("MobileNet v3 small is a lightweight mobile device compatible option (less accurate)."))
        layout.addWidget(model_box)
        roots_box = QGroupBox("Selectable dataset roots (top-level folders)")
        roots_layout = QVBoxLayout(roots_box)
        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("1: Authenticate + Get blob datasets")
        self.build_btn = QPushButton("2: Build training session manifest + fingerprint files")
        self.build_btn.setEnabled(False)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.build_btn)
        self.roots_list = QListWidget()
        self.roots_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        roots_layout.addLayout(btn_row)
        roots_layout.addWidget(self.roots_list)
        layout.addWidget(roots_box)
        out_box = QGroupBox("Output")
        out_layout = QHBoxLayout(out_box)
        self.output_dir = QLineEdit(os.path.join(self.repo_dir, "edge-ai", "training_runs"))
        self.browse_out = QPushButton("Browse...")
        out_layout.addWidget(QLabel("Run output dir:"))
        out_layout.addWidget(self.output_dir)
        out_layout.addWidget(self.browse_out)
        layout.addWidget(out_box)
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("Dataset summary will appear here.")
        layout.addWidget(self.summary)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Logs...")
        layout.addWidget(self.log)
        self.worker = None
        self.train_worker = None
        self._last_settings = None
        self.refresh_btn.clicked.connect(self.refresh_roots)
        self.build_btn.clicked.connect(self.build_manifest)
        self.browse_out.clicked.connect(self.pick_output_dir)
        self.roots_list.itemSelectionChanged.connect(self._update_build_enabled)
        self.model_architecture.currentTextChanged.connect(self._architecture_changed)
        layout.addWidget(self.train_btn)
    def _selected_architecture(self):
        return normalise_model_architecture(self.model_architecture.currentText())
    def _architecture_changed(self, arch):
        try:
            arch = normalise_model_architecture(arch)
            self.log.append(f"Selected model architecture: {arch}")
            if self._last_settings is not None:
                self._last_settings["model_architecture"] = arch
                self._last_settings.setdefault("training_parameters", {})["model_architecture"] = arch
                self._last_settings.setdefault("training_parameters", {})["arch"] = arch
                self._show_summary(self._last_settings)
        except Exception as e:
            self.log.append(f"Architecture selection error: {e}")
    def start_training(self):
        if not self._last_settings:
            self.log.append("Build a training session manifest before starting training.")
            return
        run_dir = (self._last_settings.get("outputs") or {}).get("run_dir") or self.output_dir.text().strip()
        arch = self._selected_architecture()
        self._last_settings["model_architecture"] = arch
        self._last_settings.setdefault("training_parameters", {})["model_architecture"] = arch
        self._last_settings.setdefault("training_parameters", {})["arch"] = arch
        self.log.append(f"Starting training with architecture: {arch}")
        self.train_btn.setEnabled(False)
        self.progress.setValue(0)
        self.train_worker = ModelTrainWorker(settings=self._last_settings, run_dir=run_dir, model_architecture=arch)
        self.train_worker.log.connect(self.log.append)
        self.train_worker.finished_ok.connect(self._training_done)
        self.train_worker.failed.connect(self._training_failed)
        self.train_worker.start()
    def _training_done(self, settings):
        self.log.append("Training + upload complete.")
        self.train_btn.setEnabled(True)
    def _training_failed(self, msg):
        self.log.append(f"Training failed: {msg}")
        self.train_btn.setEnabled(True)
    def pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_dir.text())
        if d:
            self.output_dir.setText(d)
    def _update_build_enabled(self):
        self.build_btn.setEnabled(len(self.roots_list.selectedItems()) > 0)
    def refresh_roots(self):
        self.log.append("Refreshing dataset roots...")
        self.progress.setValue(0)
        self.roots_list.clear()
        self.build_btn.setEnabled(False)
        self.worker = BlobScanWorker(
            account_url=self.account_url.text().strip(),
            container_name=self.container_name.text().strip(),
            selected_roots=[],
            output_dir=self.output_dir.text().strip(),
            repo_dir=self.repo_dir,
            model_architecture=self._selected_architecture(),
        )
        self.worker.log.connect(self.log.append)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.roots_ready.connect(self._populate_roots)
        self.worker.start()
    def _populate_roots(self, roots):
        self.roots_list.clear()
        for r in roots:
            item = QListWidgetItem(r)
            self.roots_list.addItem(item)
        self.log.append(f"Loaded {len(roots)} roots.")
        self._update_build_enabled()
    def build_manifest(self):
        selected = [i.text() for i in self.roots_list.selectedItems()]
        arch = self._selected_architecture()
        self.log.append(f"Building manifest for: {selected}")
        self.log.append(f"Manifest will use architecture: {arch}")
        self.progress.setValue(0)
        self.summary.clear()
        self.worker = BlobScanWorker(
            account_url=self.account_url.text().strip(),
            container_name=self.container_name.text().strip(),
            selected_roots=selected,
            output_dir=self.output_dir.text().strip(),
            repo_dir=self.repo_dir,
            model_architecture=arch,
        )
        self.worker.log.connect(self.log.append)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.result_ready.connect(self._show_summary)
        self.worker.start()
    def _show_summary(self, settings: dict):
        if settings is None:
            return
        self._last_settings = settings
        lines = []
        lines.append(f"Run: {settings.get('training_run_id')}")
        lines.append(f"Generated (UTC): {settings.get('generated_utc')}")
        lines.append(f"Git SHA: {settings.get('code', {}).get('git_sha')} (dirty={settings.get('code', {}).get('dirty')})")
        lines.append(f"Model architecture: {settings.get('model_architecture') or settings.get('training_parameters', {}).get('model_architecture')}")
        lines.append("")
        lines.append("Instrument:")
        lines.append(f"  serial_number: {settings.get('instrument', {}).get('serial_number')}")
        lines.append(f"  check: {settings.get('instrument', {}).get('consistency_check')}")
        lines.append("")
        lines.append("Dataset fingerprints (per root):")
        for root, fp in settings.get("dataset_fingerprints", {}).items():
            lines.append(f"  {root}: {fp}")
        lines.append("")
        lines.append("Class totals:")
        for k, v in sorted(settings.get("dataset_summary", {}).get("class_totals", {}).items()):
            lines.append(f"  {k}: {v}")
        lines.append("")
        lines.append(f"Total images: {settings.get('dataset_summary', {}).get('total_images')}")
        self.summary.setText("\n".join(lines))
        self.train_btn.setEnabled(True)
