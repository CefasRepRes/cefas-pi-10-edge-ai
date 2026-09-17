import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import pyvips
from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QProgressBar,
    QGroupBox,
    QLineEdit,
    QFileDialog,
    QRadioButton,
    QButtonGroup,
    QInputDialog,
    QMessageBox,
    QDialog,
    QComboBox,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
)

UTILITY_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "utility_scripts"
if str(UTILITY_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILITY_SCRIPTS_DIR))

from plot_class_counts_timeseries import run_bayesian_timeseries_analysis

from .azure_utils import get_blob_service_client, parse_blob_url, parse_model_input, utc_now_iso
from .constants import DEFAULT_ACCOUNT_URL
from .csv_utils import generate_tar_selection_csv, load_tar_selection_csv
from .prediction_sources import load_predictions_from_source
from .tar_streaming import iter_tar_members_from_blob
from .validation_chart_widget import ValidationErrorBarChart
from .validation_export import export_validation_session_to_training_libs, prepare_validation_summary_for_export
from .validation_stopping import ValidationConfig, check_stopping_criteria, wilson_interval
from .workers import ApplicationWorker, ContainerListWorker, ModelListWorker, MoveByClassWorker

# Legacy alias kept for any external code that may reference this constant.
# The validation session no longer uses a fixed image cap; see ValidationConfig.
VALIDATION_SAMPLE_SIZE = 50


def _wilson_interval(successes: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion.

    Delegates to :func:`validation_stopping.wilson_interval`.
    Kept for internal use; result includes ``half_width`` in addition to the
    keys that were present in earlier versions.
    """
    return wilson_interval(successes, n, z)


def _prediction_key(pred: dict):
    return (pred.get("container"), pred.get("source_tar_blob"), pred.get("tar_member"))


def _stratified_validation_sample(
    predictions: list,
    n: int = VALIDATION_SAMPLE_SIZE,
    important_classes=None,
):
    """Build a TAR-balanced and globally class-balanced validation pool.

    The pool is balanced across predicted labels first, while each label queue is
    itself balanced across source TARs. Important classes are prioritised, then
    smaller/minority classes. The ordering is deterministic so the same
    predictions produce the same validation candidate order.
    """
    if not predictions or n <= 0:
        return []

    usable = [
        p for p in predictions
        if p.get("predicted_label") not in (None, "ERROR")
        and os.path.basename(
            str(p.get("tar_member") or "")
        ).lower() != "background.tif"
    ]
    if not usable:
        return []

    important = {str(x) for x in (important_classes or []) if x not in (None, "")}
    by_label_tar = defaultdict(lambda: defaultdict(list))
    for pred in usable:
        label_key = str(pred.get("predicted_label"))
        tar_key = (
            str(pred.get("container") or ""),
            str(pred.get("source_tar_blob") or "UNKNOWN_TAR"),
        )
        by_label_tar[label_key][tar_key].append(pred)

    label_totals = {
        label: sum(len(rows) for rows in by_tar.values())
        for label, by_tar in by_label_tar.items()
    }

    def prediction_sort_key(pred):
        return (
            str(pred.get("tar_member") or ""),
            str(pred.get("confidence") or ""),
            str(pred.get("container") or ""),
            str(pred.get("source_tar_blob") or ""),
        )

    def label_priority(label):
        return (0 if label in important else 1, label_totals.get(label, 0), label)

    label_queues = {}
    for label, by_tar in by_label_tar.items():
        tar_queues = []
        for tar_key in sorted(by_tar):
            rows = sorted(by_tar[tar_key], key=prediction_sort_key)
            if rows:
                tar_queues.append(deque(rows))
        label_rows = []
        while tar_queues:
            next_round = []
            for queue in tar_queues:
                if queue:
                    label_rows.append(queue.popleft())
                if queue:
                    next_round.append(queue)
            tar_queues = next_round
        if label_rows:
            label_queues[label] = deque(label_rows)

    selected = []
    labels = sorted(label_queues, key=label_priority)
    while labels and len(selected) < n:
        next_labels = []
        for label in labels:
            if len(selected) >= n:
                break
            queue = label_queues[label]
            if queue:
                selected.append(queue.popleft())
            if queue:
                next_labels.append(label)
        labels = sorted(next_labels, key=label_priority)

    return selected

def _image_bytes_to_png(image_bytes: bytes, max_dim: int = 700):
    """Render arbitrary supported image bytes, including TIFF, to PNG bytes for Qt display."""
    im = pyvips.Image.new_from_buffer(image_bytes, "", access="sequential")
    try:
        if im.hasalpha():
            im = im.flatten(background=[255, 255, 255])
    except Exception:
        pass
    if im.bands != 3:
        try:
            im = im.colourspace("srgb")
        except Exception:
            if im.bands == 1:
                im = im.bandjoin([im, im])
            elif im.bands > 3:
                im = im.extract_band(0, n=3)
            else:
                raise
    if im.bands > 3:
        im = im.extract_band(0, n=3)
    scale = min(max_dim / max(im.width, im.height), 1.0)
    if scale < 1.0:
        im = im.resize(scale)
    if im.format != "uchar":
        im = im.cast("uchar")
    return im.write_to_buffer(".png")


class ValidationImagePreloadWorker(QThread):
    """Preload validation images away from the GUI thread and top up missing candidates."""

    log = Signal(str)
    image_ready = Signal(object, object)
    image_error = Signal(object, str)
    progress = Signal(int)
    final_pool_ready = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        *,
        account_url: str,
        initial_predictions: list,
        all_candidates: list,
        max_sample_size: int,
        important_classes=None,
        parent=None,
    ):
        super().__init__(parent)
        self.account_url = account_url
        self.initial_predictions = list(initial_predictions or [])
        self.all_candidates = list(all_candidates or [])
        self.max_sample_size = int(max_sample_size or 0)
        self.important_classes = {str(x) for x in (important_classes or []) if x not in (None, "")}
        self._cached_keys = set()
        self._bad_keys = set()
        self._used_keys = {_prediction_key(pred) for pred in self.initial_predictions}

    def _preload_prediction_images(self, bsc, predictions, *, reason="validation image"):
        by_tar = defaultdict(dict)
        for pred in predictions:
            key = _prediction_key(pred)
            container, tar_name, member_name = key
            if not (container and tar_name and member_name):
                self._bad_keys.add(key)
                self.image_error.emit(key, "Prediction is missing container, TAR, or member name")
                continue
            if key in self._cached_keys:
                continue
            by_tar[(container, tar_name)][member_name] = pred

        requested = sum(len(members) for members in by_tar.values())
        if not requested:
            return 0

        loaded = 0
        completed = 0
        for tar_i, ((container, tar_name), wanted) in enumerate(by_tar.items(), start=1):
            remaining = dict(wanted)
            self.log.emit(
                f"[{tar_i}/{len(by_tar)}] Streaming TAR once for {len(remaining)} {reason}(s): "
                f"{container}/{tar_name}"
            )
            try:
                cc = bsc.get_container_client(container)
                bc = cc.get_blob_client(tar_name)
                for member, fh in iter_tar_members_from_blob(bc, tar_name, log_cb=self.log.emit):
                    member_name = member.name
                    pred = remaining.get(member_name)
                    if pred is None:
                        continue
                    key = _prediction_key(pred)
                    try:
                        png = _image_bytes_to_png(fh.read())
                        self._cached_keys.add(key)
                        self.image_ready.emit(key, png)
                        loaded += 1
                    except Exception as exc:
                        self._bad_keys.add(key)
                        self.image_error.emit(key, str(exc))
                        self.log.emit(
                            f"WARNING: failed to decode validation image "
                            f"{container}/{tar_name}!/{member_name}: {exc}"
                        )
                    remaining.pop(member_name, None)
                    completed += 1
                    self.progress.emit(min(99, int((completed / max(1, requested)) * 100)))
                    if not remaining:
                        break
                if remaining:
                    for member_name, pred in remaining.items():
                        key = _prediction_key(pred)
                        self._bad_keys.add(key)
                        self.image_error.emit(key, f"Member not found in TAR: {container}/{tar_name}!/{member_name}")
                    self.log.emit(f"WARNING: {len(remaining)} validation image(s) were not found in {container}/{tar_name}")
            except Exception as exc:
                self.log.emit(f"ERROR streaming validation TAR {container}/{tar_name}: {exc}")
                for pred in remaining.values():
                    key = _prediction_key(pred)
                    self._bad_keys.add(key)
                    self.image_error.emit(key, str(exc))
        self.log.emit(f"    Cached {loaded}/{requested} requested {reason}(s)")
        return loaded

    def _candidate_priority(self, deficits):
        def priority(pred):
            label = str(pred.get("predicted_label"))
            return (
                0 if deficits.get(label, 0) > 0 else 1,
                0 if label in self.important_classes else 1,
                -deficits.get(label, 0),
                label,
                str(pred.get("source_tar_blob") or ""),
                str(pred.get("tar_member") or ""),
            )
        return priority

    def run(self):
        try:
            target_size = min(self.max_sample_size, len(self.all_candidates))
            if target_size <= 0:
                self.final_pool_ready.emit([])
                self.progress.emit(100)
                return
            bsc = get_blob_service_client(self.account_url, log_cb=self.log.emit)
            self.log.emit(
                f"Preloading {len(self.initial_predictions)} validation image(s); "
                "metadata fields remain editable while this runs..."
            )
            self._preload_prediction_images(bsc, self.initial_predictions, reason="validation image")

            good_predictions = []
            deficits = defaultdict(int)
            for pred in self.initial_predictions:
                key = _prediction_key(pred)
                if key in self._cached_keys:
                    good_predictions.append(pred)
                else:
                    deficits[str(pred.get("predicted_label"))] += 1

            reserve = [
                pred for pred in self.all_candidates
                if _prediction_key(pred) not in self._used_keys
                and _prediction_key(pred) not in self._bad_keys
            ]
            added = []
            tried = 0
            if len(good_predictions) < target_size and reserve:
                self.log.emit(
                    f"Validation preload left {sum(deficits.values())} uncached/broken image(s); "
                    "topping up from the balanced reserve pool..."
                )
            while len(good_predictions) + len(added) < target_size and reserve:
                reserve.sort(key=self._candidate_priority(deficits))
                candidate = reserve.pop(0)
                key = _prediction_key(candidate)
                self._used_keys.add(key)
                tried += 1
                self._preload_prediction_images(bsc, [candidate], reason="top-up validation image")
                if key in self._cached_keys:
                    added.append(candidate)
                    label = str(candidate.get("predicted_label"))
                    if deficits.get(label, 0) > 0:
                        deficits[label] -= 1

            final_pool = good_predictions + added
            self.log.emit(
                f"Validation preload complete: usable pool has {len(final_pool)}/{target_size} image(s) "
                f"after trying {tried} reserve candidate(s)."
            )
            self.progress.emit(100)
            self.final_pool_ready.emit(final_pool)
        except Exception as exc:
            self.failed.emit(str(exc))



class ValidationConfigDialog(QDialog):
    """Small dialog for configuring adaptive validation stopping criteria.

    Parameters configured here are forwarded to :class:`ValidationConfig` and
    subsequently to :class:`ValidationSessionDialog`.
    """

    def __init__(self, known_labels: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Validation stopping criteria")
        self.resize(480, 320)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # CI width target
        self._ci_spin = QDoubleSpinBox()
        self._ci_spin.setRange(0.01, 0.50)
        self._ci_spin.setSingleStep(0.01)
        self._ci_spin.setDecimals(2)
        self._ci_spin.setValue(0.05)
        self._ci_spin.setToolTip(
            "Validation stops when the overall-accuracy Wilson CI half-width "
            "falls below this value.  Default ±5 % (0.05)."
        )
        form.addRow("Accuracy CI half-width target (±):", self._ci_spin)

        # Min class precision
        self._prec_spin = QDoubleSpinBox()
        self._prec_spin.setRange(0.0, 1.0)
        self._prec_spin.setSingleStep(0.05)
        self._prec_spin.setDecimals(2)
        self._prec_spin.setValue(0.80)
        self._prec_spin.setToolTip(
            "Minimum precision required for every important class.  Default 0.80 (80 %)."
        )
        form.addRow("Min precision for important classes:", self._prec_spin)

        # Max sample size
        self._max_spin = QSpinBox()
        self._max_spin.setRange(10, 10000)
        self._max_spin.setSingleStep(50)
        self._max_spin.setValue(500)
        self._max_spin.setToolTip(
            "Maximum images drawn into the validation pool.  "
            "Validation stops as soon as the adaptive criteria are met, "
            "which may be before this limit is reached."
        )
        form.addRow("Maximum sample pool size:", self._max_spin)

        layout.addLayout(form)

        # Important classes checkboxes
        imp_box = QGroupBox("Important classes (must meet precision threshold)")
        imp_layout = QVBoxLayout(imp_box)
        self._class_checkboxes: list = []
        for label in sorted(known_labels):
            cb = QCheckBox(label)
            imp_layout.addWidget(cb)
            self._class_checkboxes.append(cb)
        if not known_labels:
            imp_layout.addWidget(QLabel("(no known labels — run application first)"))
        imp_box.setLayout(imp_layout)
        layout.addWidget(imp_box)

        btns = QHBoxLayout()
        ok_btn = QPushButton("Start validation")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        btns.addStretch(1)
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def get_config(self) -> ValidationConfig:
        """Return a :class:`ValidationConfig` built from the current dialog state."""
        important = [cb.text() for cb in self._class_checkboxes if cb.isChecked()]
        return ValidationConfig(
            ci_width_target=round(self._ci_spin.value(), 4),
            important_classes=important,
            min_class_precision=round(self._prec_spin.value(), 4),
            max_sample_size=self._max_spin.value(),
        )


class ValidationSessionDialog(QDialog):
    """Post-application validation session with adaptive CI-based stopping.

    The session continues until *both* stopping criteria configured in
    *config* are satisfied, or until the user explicitly clicks
    "Finish session":

    1. **Accuracy CI criterion** – the Wilson score interval half-width for
       overall accuracy drops below ``config.ci_width_target`` (default ±5 %).
    2. **Class precision gate** – every class in ``config.important_classes``
       reaches a point-estimate precision ≥ ``config.min_class_precision``
       (default 80 %).  Classes with zero predictions are treated
       conservatively (criterion not met).

    Images are drawn from a stratified pool of up to ``config.max_sample_size``
    items selected round-robin across predicted classes.  The selection is
    deterministic given the same predictions list.
    """

    def __init__(
        self,
        *,
        account_url: str,
        predictions: list,
        output_run_dir: str,
        labels: list,
        config: ValidationConfig = None,
        log_cb=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Validation session")
        self.resize(900, 540)
        self.account_url = account_url
        self._config = config if config is not None else ValidationConfig()
        all_predictions = [
            p for p in (predictions or [])
            if p.get("predicted_label") not in (None, "ERROR")
        ]
        self._all_validation_candidates = _stratified_validation_sample(
            all_predictions,
            len(all_predictions),
            important_classes=self._config.important_classes,
        )
        self.predictions = self._all_validation_candidates[:self._config.max_sample_size]
        self.output_run_dir = output_run_dir
        self.labels = sorted({str(x) for x in labels if x not in (None, "")})
        self.labels = sorted(set(self.labels))
        self.log_cb = log_cb
        self.current_idx = 0
        self.annotations = []
        self._image_cache = {}
        self._image_errors = {}
        self._bsc = None
        self._loading_current = False
        self._preload_worker = None
        self._preload_complete = False
        self._submitted_metadata = None
        self._rebuild_annotations_from_predictions()

        layout = QVBoxLayout(self)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.metadata_box = QGroupBox("Validation session metadata (fill while images load)")
        metadata_layout = QFormLayout(self.metadata_box)
        self.session_name_edit = QLineEdit(utc_now_iso().replace(":", "-").replace("Z", ""))
        self.validator_name_edit = QLineEdit("")
        self.validator_email_edit = QLineEdit("")
        self.organisation_edit = QLineEdit("")
        self.project_name_edit = QLineEdit("")
        self.dataset_name_edit = QLineEdit("")
        self.instrument_edit = QLineEdit("")
        self.taxonomy_name_edit = QLineEdit("")
        self.model_label_edit = QLineEdit("")
        self.validation_purpose_combo = QComboBox()
        self.validation_purpose_combo.addItems([
            "inference_quality_assurance",
            "model_acceptance_test",
            "model_improvement",
            "publication_dataset",
            "other",
        ])
        self.validation_notes_edit = QTextEdit()
        self.validation_notes_edit.setMaximumHeight(70)
        self.submit_metadata_btn = QPushButton("Submit validation metadata")
        metadata_layout.addRow("Session name *:", self.session_name_edit)
        metadata_layout.addRow("Validator name *:", self.validator_name_edit)
        metadata_layout.addRow("Validator email *:", self.validator_email_edit)
        metadata_layout.addRow("Organisation *:", self.organisation_edit)
        metadata_layout.addRow("Project / survey:", self.project_name_edit)
        metadata_layout.addRow("Dataset name:", self.dataset_name_edit)
        metadata_layout.addRow("Instrument:", self.instrument_edit)
        metadata_layout.addRow("Taxonomy name:", self.taxonomy_name_edit)
        metadata_layout.addRow("Model/session label:", self.model_label_edit)
        metadata_layout.addRow("Validation purpose *:", self.validation_purpose_combo)
        metadata_layout.addRow("Notes:", self.validation_notes_edit)
        metadata_layout.addRow("", self.submit_metadata_btn)
        layout.addWidget(self.metadata_box)
        self.metadata_submitted_label = QLabel("")
        self.metadata_submitted_label.setWordWrap(True)
        self.metadata_submitted_label.hide()
        layout.addWidget(self.metadata_submitted_label)

        self.preload_progress = QProgressBar()
        self.preload_progress.setRange(0, 100)
        self.preload_progress.setValue(0)
        layout.addWidget(self.preload_progress)

        # Stopping-criteria status banner
        self._criteria_label = QLabel("")
        self._criteria_label.setWordWrap(True)
        self._criteria_label.setStyleSheet(
            "QLabel { font-weight: bold; padding: 4px; border-radius: 3px; }"
        )
        layout.addWidget(self._criteria_label)

        self.image_label = QLabel("Image preview")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(280)
        self.image_label.setStyleSheet("QLabel { background: #111; color: #ddd; border: 1px solid #555; }")
        layout.addWidget(self.image_label, stretch=1)

        self.validation_chart = ValidationErrorBarChart(self)
        layout.addWidget(self.validation_chart)

        form = QFormLayout()
        self.predicted_label = QLabel("")
        self.true_label_combo = QComboBox()
        self.true_label_combo.setEditable(False)
        self.true_label_combo.addItems(self.labels)
        form.addRow("Predicted class:", self.predicted_label)
        form.addRow("True/correct class:", self.true_label_combo)
        layout.addLayout(form)

        nav = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.next_btn = QPushButton("Save + Next")
        self.finish_btn = QPushButton("Finish session and write validation.json")
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.next_btn)
        nav.addWidget(self.finish_btn)
        layout.addLayout(nav)

        self.prev_btn.clicked.connect(self.previous_image)
        self.next_btn.clicked.connect(self.next_image)
        self.finish_btn.clicked.connect(self.finish_session)
        self.submit_metadata_btn.clicked.connect(self._submit_validation_metadata)
        self.true_label_combo.currentTextChanged.connect(self._current_label_changed)

        if not self.predictions:
            self.status_label.setText("No usable predictions available for validation.")
            self.next_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.finish_btn.setEnabled(False)
        else:
            self._start_preload_worker()

    def _log(self, msg: str):
        if self.log_cb:
            self.log_cb(msg)

    def _ensure_client(self):
        if self._bsc is None:
            self._bsc = get_blob_service_client(self.account_url, log_cb=self._log)

    def _current_metadata_from_form(self):
        return {
            "session_name": self.session_name_edit.text().strip(),
            "validator_name": self.validator_name_edit.text().strip(),
            "validator_email": self.validator_email_edit.text().strip(),
            "organisation": self.organisation_edit.text().strip(),
            "project_name": self.project_name_edit.text().strip(),
            "dataset_name": self.dataset_name_edit.text().strip(),
            "instrument": self.instrument_edit.text().strip(),
            "taxonomy_name": self.taxonomy_name_edit.text().strip(),
            "model_label": self.model_label_edit.text().strip(),
            "validation_purpose": self.validation_purpose_combo.currentText().strip(),
            "validation_notes": self.validation_notes_edit.toPlainText().strip(),
            "submitted_utc": utc_now_iso(),
        }

    def _validation_metadata(self):
        return dict(self._submitted_metadata or self._current_metadata_from_form())

    def _metadata_missing_required(self, metadata=None):
        md = metadata or self._validation_metadata()
        required = {
            "session_name": "Session name",
            "validator_name": "Validator name",
            "validator_email": "Validator email",
            "organisation": "Organisation",
            "validation_purpose": "Validation purpose",
        }
        return [label for key, label in required.items() if not md.get(key)]

    def _submit_validation_metadata(self):
        md = self._current_metadata_from_form()
        missing = self._metadata_missing_required(md)
        if missing:
            QMessageBox.warning(
                self,
                "Validation metadata incomplete",
                "Please complete the required validation metadata fields before submitting:\n"
                + "\n".join(f"- {item}" for item in missing),
            )
            return
        self._submitted_metadata = md
        self.metadata_box.hide()

        self.layout().invalidate()
        self.layout().activate()
        self.adjustSize()        
        
        self.metadata_submitted_label.setText(
            f"Validation metadata submitted: {md.get('session_name')} "
            f"({md.get('validator_name')}, {md.get('organisation')})"
        )
        self.metadata_submitted_label.setStyleSheet(
            "QLabel { font-weight: bold; padding: 4px; border-radius: 3px; "
            "background: #e3f2fd; color: #0d47a1; }"
        )
        self.metadata_submitted_label.show()
        self._update_validation_chart()

    def _rebuild_annotations_from_predictions(self):
        self.annotations = []
        for pred in self.predictions:
            predicted = str(pred.get("predicted_label"))
            if predicted not in self.labels:
                self.labels.append(predicted)
            self.annotations.append({
                "container": pred.get("container"),
                "source_tar_blob": pred.get("source_tar_blob"),
                "tar_member": pred.get("tar_member"),
                "blob_member_uri": pred.get("blob_member_uri"),
                "selected_prefix": pred.get("selected_prefix"),
                "original_label": predicted,
                "final_label": predicted,
                "confidence": pred.get("confidence"),
                "model_blob": pred.get("model_blob"),
                "model_runid": pred.get("model_runid"),
                "validated": False,
            })
        self.labels = sorted(set(self.labels))

    def _refresh_label_combo_items(self):
        current = self.true_label_combo.currentText().strip() if hasattr(self, "true_label_combo") else ""
        self.true_label_combo.blockSignals(True)
        self.true_label_combo.clear()
        self.true_label_combo.addItems(self.labels)
        if current:
            idx = self.true_label_combo.findText(current)
            if idx >= 0:
                self.true_label_combo.setCurrentIndex(idx)
        self.true_label_combo.blockSignals(False)

    def _on_preload_image_ready(self, key, png):
        self._image_cache[key] = png
        self._image_errors.pop(key, None)

    def _on_preload_image_error(self, key, message):
        self._image_errors[key] = message

    def _on_preload_finished(self, final_pool):
        self.predictions = final_pool or []
        self._rebuild_annotations_from_predictions()
        self._refresh_label_combo_items()
        self._preload_complete = True
        self.preload_progress.setValue(100)
        if not self.predictions:
            self.status_label.setText("No validation images could be loaded. Check the log for preload errors.")
            self.next_btn.setEnabled(False)
            self.prev_btn.setEnabled(False)
            self.finish_btn.setEnabled(False)
            return
        self.next_btn.setEnabled(len(self.predictions) > 1)
        self.prev_btn.setEnabled(False)
        self.finish_btn.setEnabled(True)
        self._show_current()
        self._update_validation_chart()

    def _on_preload_failed(self, message: str):
        self._preload_complete = True
        self.status_label.setText(f"Validation image preload failed:\n{message}")
        self.finish_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.prev_btn.setEnabled(False)
        self._log(f"Validation image preload failed: {message}")

    def _start_preload_worker(self):
        self.status_label.setText("Validation images are loading. Fill and submit the metadata fields while this runs...")
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)
        self.finish_btn.setEnabled(False)
        self._preload_worker = ValidationImagePreloadWorker(
            account_url=self.account_url,
            initial_predictions=self.predictions,
            all_candidates=self._all_validation_candidates,
            max_sample_size=self._config.max_sample_size,
            important_classes=self._config.important_classes,
            parent=self,
        )
        self._preload_worker.log.connect(self._log)
        self._preload_worker.image_ready.connect(self._on_preload_image_ready)
        self._preload_worker.image_error.connect(self._on_preload_image_error)
        self._preload_worker.progress.connect(self.preload_progress.setValue)
        self._preload_worker.final_pool_ready.connect(self._on_preload_finished)
        self._preload_worker.failed.connect(self._on_preload_failed)
        self._preload_worker.start()

    def _preload_all_images_single_pass(self):
        """Group validation images by TAR and stream each TAR only once."""
        by_tar = defaultdict(dict)
        for pred in self.predictions:
            container, tar_name, member_name = _prediction_key(pred)
            if not (container and tar_name and member_name):
                self._image_errors[_prediction_key(pred)] = "Prediction is missing container, TAR, or member name"
                continue
            by_tar[(container, tar_name)][member_name] = pred

        total_images = sum(len(members) for members in by_tar.values())
        self._log(
            f"Preloading {total_images} validation image(s) from {len(by_tar)} TAR archive(s) "
            "using one streaming pass per TAR..."
        )
        loaded = 0
        for tar_i, ((container, tar_name), wanted) in enumerate(by_tar.items(), start=1):
            remaining = dict(wanted)
            self._log(
                f"[{tar_i}/{len(by_tar)}] Streaming TAR once for {len(remaining)} validation image(s): "
                f"{container}/{tar_name}"
            )
            try:
                cc = self._bsc.get_container_client(container)
                bc = cc.get_blob_client(tar_name)
                for member, fh in iter_tar_members_from_blob(bc, tar_name, log_cb=self._log):
                    member_name = member.name
                    pred = remaining.get(member_name)
                    if pred is None:
                        continue
                    key = _prediction_key(pred)
                    try:
                        self._image_cache[key] = _image_bytes_to_png(fh.read())
                        loaded += 1
                    except Exception as exc:
                        self._image_errors[key] = str(exc)
                        self._log(f"WARNING: failed to decode validation image {container}/{tar_name}!/{member_name}: {exc}")
                    remaining.pop(member_name, None)
                    if not remaining:
                        break
                if remaining:
                    for member_name, pred in remaining.items():
                        key = _prediction_key(pred)
                        self._image_errors[key] = f"Member not found in TAR: {container}/{tar_name}!/{member_name}"
                    self._log(f"WARNING: {len(remaining)} validation image(s) were not found in {container}/{tar_name}")
            except Exception as exc:
                self._log(f"ERROR streaming validation TAR {container}/{tar_name}: {exc}")
                for pred in remaining.values():
                    self._image_errors[_prediction_key(pred)] = str(exc)
            self._log(f"    Cached {loaded}/{total_images} validation image(s)")
        self._log(
            "Validation image preload complete. Each TAR was streamed at most once; "
        )

    def _fetch_image_png(self, pred: dict):
        """Fallback reader. Normal validation navigation should use _image_cache."""
        key = _prediction_key(pred)
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached
        container, tar_name, member_name = key
        if not (container and tar_name and member_name):
            raise RuntimeError("Prediction is missing container, TAR, or member name")
        cc = self._bsc.get_container_client(container)
        bc = cc.get_blob_client(tar_name)
        for member, fh in iter_tar_members_from_blob(bc, tar_name, log_cb=self._log):
            if member.name == member_name:
                return _image_bytes_to_png(fh.read())
        raise RuntimeError(f"Could not find member in TAR: {container}/{tar_name}!/{member_name}")

    def _save_current_choice(self):
        if self.annotations:
            self.annotations[self.current_idx]["final_label"] = self.true_label_combo.currentText().strip()
            self.annotations[self.current_idx]["validated"] = True

    def _current_label_changed(self):
        if self._loading_current or not self.annotations:
            return
        self._save_current_choice()
        self._update_validation_chart()

    def _update_validation_chart(self):
        rows = [r for r in self.annotations if r.get("validated")]
        summary = self._build_validation_summary(rows=rows)
        self.validation_chart.set_summary(summary, sample_count=len(self.annotations))
        self._refresh_criteria_banner(summary.get("stopping_criteria") or {})

    def _refresh_criteria_banner(self, criteria: dict):
        """Update the stopping-criteria status banner with the latest check result."""
        if not criteria:
            self._criteria_label.setText("")
            self._criteria_label.setStyleSheet(
                "QLabel { font-weight: bold; padding: 4px; border-radius: 3px; }"
            )
            return
        summary_text = criteria.get("summary", "")
        all_met = bool(criteria.get("all_criteria_met"))
        if all_met:
            self._criteria_label.setStyleSheet(
                "QLabel { font-weight: bold; padding: 4px; border-radius: 3px; "
                "background: #c8e6c9; color: #1b5e20; }"
            )
            self._criteria_label.setText(
                f"{summary_text}\n"
                "You may finish the session now, or continue validating more images."
            )
        else:
            self._criteria_label.setStyleSheet(
                "QLabel { font-weight: bold; padding: 4px; border-radius: 3px; "
                "background: #fff3e0; color: #bf360c; }"
            )
            self._criteria_label.setText(summary_text)

    def _show_current(self):
        pred = self.predictions[self.current_idx]
        ann = self.annotations[self.current_idx]
        key = _prediction_key(pred)
        self.status_label.setText(
            f"Image {self.current_idx + 1} of {len(self.predictions)} | "
            f"{pred.get('container')}/{pred.get('source_tar_blob')}!/{pred.get('tar_member')}"
        )
        self.predicted_label.setText(f"{ann.get('original_label')} (confidence={ann.get('confidence')})")
        self._loading_current = True
        idx = self.true_label_combo.findText(str(ann.get("final_label")))
        if idx >= 0:
            self.true_label_combo.setCurrentIndex(idx)
        self._loading_current = False
        try:
            if key in self._image_errors:
                raise RuntimeError(self._image_errors[key])
            png = self._image_cache.get(key)
            if png is None:
                if not self._preload_complete:
                    raise RuntimeError("Image is still loading; please continue filling/submitting the metadata fields.")
                png = self._fetch_image_png(pred)
                self._image_cache[key] = png
            pix = QPixmap()
            pix.loadFromData(png, "PNG")
            self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as exc:
            self.image_label.setText(f"Could not load image preview:\n{exc}")
        self.prev_btn.setEnabled(self.current_idx > 0)
        self.next_btn.setEnabled(self.current_idx < len(self.predictions) - 1)

    def previous_image(self):
        self._save_current_choice()
        if self.current_idx > 0:
            self.current_idx -= 1
            self._show_current()
        self._update_validation_chart()

    def next_image(self):
        self._save_current_choice()
        if self.current_idx < len(self.predictions) - 1:
            self.current_idx += 1
            self._show_current()
        self._update_validation_chart()

    def _build_validation_summary(self, rows=None):
        rows = self.annotations if rows is None else rows
        total = len(rows)
        correct_total = sum(1 for row in rows if row.get("original_label") == row.get("final_label"))
        stopping = check_stopping_criteria(rows, config=self._config)
        error_model = stopping.get("confusion_error_model") or {}
        per_class = {}
        for cls in error_model.get("class_order") or []:
            metrics = error_model.get("per_class_metrics", {}).get(cls, {})
            per_class[cls] = {
                "true_support": metrics.get("true_support"),
                "predicted_support": metrics.get("predicted_support"),
                "correct": metrics.get("correct"),
                "false_negative": metrics.get("false_negative"),
                "false_positive": metrics.get("false_positive"),
                "recall": metrics.get("recall"),
                "precision": metrics.get("precision"),
            }
        metadata = self._validation_metadata()
        summary = {
            "generated_utc": utc_now_iso(),
            "validation_session": metadata,
            "validation_sample_size_requested": self._config.max_sample_size,
            "validation_sample_size_deprecated_fixed_cap": VALIDATION_SAMPLE_SIZE,
            "validated_image_count": total,
            "overall_accuracy": _wilson_interval(correct_total, total),
            "per_class": per_class,
            "classifier_error_model": error_model,
            "stopping_criteria": stopping,
            "images": rows,
        }
        return prepare_validation_summary_for_export(
            summary,
            rows=rows,
            labels=self.labels,
            metadata=metadata,
            error_model=error_model,
        )

    def finish_session(self):
        if self._submitted_metadata is None:
            QMessageBox.warning(
                self,
                "Validation metadata not submitted",
                "Please complete and submit the validation metadata before finishing the session.",
            )
            return
        self._save_current_choice()
        if not self.output_run_dir:
            QMessageBox.warning(self, "Missing run directory", "No application run folder is available.")
            return
        validated_rows = [row for row in self.annotations if row.get("validated")]
        if not validated_rows:
            QMessageBox.warning(self, "No validated images", "Validate at least one image before finishing the session.")
            return
        validation_summary = self._build_validation_summary(rows=validated_rows)
        os.makedirs(self.output_run_dir, exist_ok=True)
        out_path = os.path.join(self.output_run_dir, "validation.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(validation_summary, f, indent=2)
        self._log(f"Wrote validation summary: {out_path}")
        try:
            self._ensure_client()
            export_info = export_validation_session_to_training_libs(
                blob_service_client=self._bsc,
                account_url=self.account_url,
                rows=validated_rows,
                validation_summary=validation_summary,
                labels=self.labels,
                config=self._config,
                metadata=self._validation_metadata(),
                log_cb=self._log,
            )
            dataset_path = os.path.join(self.output_run_dir, "dataset.json")
            with open(dataset_path, "w", encoding="utf-8") as f:
                json.dump(export_info["dataset_json"], f, indent=2)
            self._log(f"Wrote validation dataset metadata: {dataset_path}")
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Validation export failed",
                f"validation.json was written locally, but export to training-libs failed:\n{exc}",
            )
            return
        QMessageBox.information(
            self,
            "Validation complete",
            "Wrote validation outputs:\n"
            f"{out_path}\n"
            f"{dataset_path}\n\n"
            "Uploaded validation dataset to:\n"
            f"training-libs/{export_info['export_folder']}",
        )
        self.accept()


class ApplicationValidationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.account_url_default = DEFAULT_ACCOUNT_URL

        layout = QVBoxLayout(self)

        storage_box = QGroupBox("Azure Blob Storage")
        storage_layout = QHBoxLayout(storage_box)
        self.account_url = QLineEdit(self.account_url_default)
        self.refresh_containers_btn = QPushButton("1: Authenticate + List containers")
        storage_layout.addWidget(QLabel("Account URL:"))
        storage_layout.addWidget(self.account_url)
        storage_layout.addWidget(self.refresh_containers_btn)
        layout.addWidget(storage_box)

        targets_box = QGroupBox("Targets: containers / folders")
        targets_layout = QVBoxLayout(targets_box)
        add_row = QHBoxLayout()
        self.prefix_edit = QLineEdit("")
        self.prefix_edit.setPlaceholderText("Optional prefix (e.g. 2025-09-19/1200.tar/RawImages/) - blank = whole container")
        self.add_selected_container_btn = QPushButton("2: Add selected container + prefix")
        add_row.addWidget(QLabel("Prefix:"))
        add_row.addWidget(self.prefix_edit)
        add_row.addWidget(self.add_selected_container_btn)
        targets_layout.addLayout(add_row)

        lists_row = QHBoxLayout()
        self.containers_list = QListWidget()
        self.containers_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.containers_list.setMinimumWidth(300)
        self.selected_targets_list = QListWidget()
        self.selected_targets_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        lists_row.addWidget(self.containers_list)
        lists_row.addWidget(self.selected_targets_list)
        targets_layout.addLayout(lists_row)

        url_row = QHBoxLayout()
        self.url_text = QTextEdit()
        self.url_text.setPlaceholderText(
            "Paste one URL per line (container or folder), e.g.\n"
            "https://citprodc8603uksa.blob.core.windows.net/cend-10-25-megsneps-earlysummer-northsea\n"
            "https://citprodc8603uksa.blob.core.windows.net/cend16-25-abts-irishsea-bchannel-autumn/2025-09-19/1200.tar/RawImages/"
        )
        self.url_text.setFixedHeight(90)
        self.add_urls_btn = QPushButton("Add URLs")
        self.remove_targets_btn = QPushButton("Remove selected targets")
        url_row.addWidget(self.url_text)
        btn_col = QVBoxLayout()
        btn_col.addWidget(self.add_urls_btn)
        btn_col.addWidget(self.remove_targets_btn)
        btn_col.addStretch(1)
        url_row.addLayout(btn_col)
        targets_layout.addLayout(url_row)
        layout.addWidget(targets_box)

        resume_row = QHBoxLayout()
        self.resume_previous_job_edit = QLineEdit("")
        self.resume_previous_job_edit.setPlaceholderText(
            "Optional previous job URL (e.g. https://.../runs/guilocalinference-.../)")
        resume_row.addWidget(QLabel("Resume previous job:"))
        resume_row.addWidget(self.resume_previous_job_edit)
        layout.addLayout(resume_row)

        self.gen_csv_btn = QPushButton("3: Generate TAR CSV")
        self.load_csv_btn = QPushButton("(optional 3b) Load TAR CSV")
        self.gen_csv_btn.clicked.connect(self.generate_tar_csv)
        self.load_csv_btn.clicked.connect(self.load_tar_csv)
        layout.addWidget(self.gen_csv_btn)
        layout.addWidget(self.load_csv_btn)

        model_box = QGroupBox("Model selection (container: trainedmodels)")
        model_layout = QVBoxLayout(model_box)
        model_btn_row = QHBoxLayout()
        self.refresh_models_btn = QPushButton("4: List models")
        self.model_url_edit = QLineEdit("")
        self.model_url_edit.setPlaceholderText("Paste model URL OR type blob name (e.g. model2026-...Z.pt)")
        self.use_model_btn = QPushButton("Use model input")
        model_btn_row.addWidget(self.refresh_models_btn)
        model_btn_row.addWidget(self.model_url_edit)
        #model_btn_row.addWidget(self.use_model_btn)
        model_layout.addLayout(model_btn_row)
        self.models_list = QListWidget()
        self.models_list.setSelectionMode(QAbstractItemView.SingleSelection)
        model_layout.addWidget(self.models_list)
        layout.addWidget(model_box)

        mode_box = QGroupBox("Mode")
        mode_layout = QHBoxLayout(mode_box)
        self.rb_application = QRadioButton("Application")
        self.rb_validation = QRadioButton("Validation session after predictions")
        self.rb_application.setChecked(True)
        self.rb_validation.setEnabled(False)
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_application)
        self.mode_group.addButton(self.rb_validation)
        mode_layout.addWidget(self.rb_application)
        mode_layout.addWidget(self.rb_validation)
        self.predictions_source_edit = QLineEdit("")
        self.predictions_source_edit.setPlaceholderText("Local per_tar dir or blob container/prefix")
        self.load_predictions_btn = QPushButton("Load predictions")
        self.load_predictions_btn.clicked.connect(self.load_predictions_from_source)
        mode_layout.addWidget(QLabel("Predictions source:"))
        mode_layout.addWidget(self.predictions_source_edit)
        mode_layout.addWidget(self.load_predictions_btn)
        mode_layout.addStretch(1)
        layout.addWidget(mode_box)

        out_box = QGroupBox("Output")
        out_layout = QHBoxLayout(out_box)
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        default_out = os.path.join(repo_dir, "edge-ai", "application_runs")
        self.output_dir = QLineEdit(default_out)
        self.browse_out = QPushButton("Browse...")
        out_layout.addWidget(QLabel("Results folder:"))
        out_layout.addWidget(self.output_dir)
        out_layout.addWidget(self.browse_out)
        layout.addWidget(out_box)

        action_row = QHBoxLayout()
        self.run_btn = QPushButton("5: Run application (stream TARs directly from blob storage)")
        self.move_btn = QPushButton("6: Re-stream and download TIFFs into class subfolders")
        self.validation_btn = QPushButton("7: Start validation session (adaptive CI-based stopping)")
        self.bayesian_btn = QPushButton("8: Run Bayesian statistics")
        self.move_btn.setEnabled(False)
        self.validation_btn.setEnabled(False)
        self.bayesian_btn.setEnabled(False)
        action_row.addWidget(self.run_btn)
        action_row.addWidget(self.move_btn)
        action_row.addWidget(self.validation_btn)
        action_row.addWidget(self.bayesian_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Logs...")
        layout.addWidget(self.log)

        self._container_worker = None
        self._model_worker = None
        self._app_worker = None
        self._move_worker = None
        self._last_predictions = None
        self._last_account_url = None
        self._last_run_dir = None
        self._last_predictions_source = None
        self.selected_tars = None

        self.refresh_containers_btn.clicked.connect(self.refresh_containers)
        self.add_selected_container_btn.clicked.connect(self.add_selected_containers)
        self.add_urls_btn.clicked.connect(self.add_urls)
        self.remove_targets_btn.clicked.connect(self.remove_selected_targets)
        self.refresh_models_btn.clicked.connect(self.refresh_models)
        self.use_model_btn.clicked.connect(self.use_model_input)
        self.models_list.itemSelectionChanged.connect(self._sync_model_edit_from_selection)
        self.browse_out.clicked.connect(self.pick_output_dir)
        self.run_btn.clicked.connect(self.run_application)
        self.move_btn.clicked.connect(self.move_by_class)
        self.validation_btn.clicked.connect(self.start_validation_session)
        self.bayesian_btn.clicked.connect(self.run_bayesian_statistics)

    def _storage_client(self):
        return get_blob_service_client(self.account_url.text().strip(), log_cb=self._append_log)

    def _selected_tars_sampling_summary(self):
        if not self.selected_tars:
            return "within-TAR sample 100.00%"
        ratios = sorted({
            float(row[3]) if isinstance(row, (list, tuple)) and len(row) >= 4 else 1.0
            for row in self.selected_tars
        })
        if len(ratios) == 1:
            return f"within-TAR sample {ratios[0]:.2%}"
        return f"mixed within-TAR samples {ratios[0]:.2%} to {ratios[-1]:.2%}"

    def _tar_csv_targets_from_ui(self):
        targets = self._targets_from_list()
        if targets:
            return targets

        current = self.containers_list.currentItem()
        if current is None:
            return []

        prefix = self.prefix_edit.text().strip()
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return [{"container": current.text().strip(), "prefix": prefix}]

    def generate_tar_csv(self):
        csv_path, _ = QFileDialog.getSaveFileName(self, "Save TAR selection CSV", "tar_selection.csv", "CSV Files (*.csv)")
        if not csv_path:
            return
        targets = self._tar_csv_targets_from_ui()
        if not targets:
            QMessageBox.warning(self, "No targets selected", "Please select at least one inference target first.")
            return
        ratio, ok = QInputDialog.getDouble(self, "Preselect ratio", "Fraction of TARs to preselect in your csv:", 0.10, 0.0, 1.0, 2)
        if not ok:
            return
        sample_ratio, ok = QInputDialog.getDouble(
            self,
            "Within-TAR sampling",
            "Fraction of images to process within each selected TAR:",
            1.0,
            0.0,
            1.0,
            4,
        )
        if not ok:
            return
        bsc = self._storage_client()
        resume_previous_job_url = self.resume_previous_job_edit.text().strip()
        generate_tar_selection_csv(
            bsc=bsc,
            targets=targets,
            csv_path=csv_path,
            preselect_ratio=ratio,
            sample_ratio=sample_ratio,
            resume_previous_job_url=resume_previous_job_url,
            log_cb=self._append_log,
        )
        self.selected_tars = load_tar_selection_csv(csv_path)
        self._append_log(
            f"Loaded {len(self.selected_tars)} TARs from newly generated CSV "
            f"({self._selected_tars_sampling_summary()}). "
            "You are welcome to modify the selected containers (Y/N) and re-import the csv."
        )

    def load_tar_csv(self):
        csv_path, _ = QFileDialog.getOpenFileName(self, "Load TAR selection CSV", "", "CSV Files (*.csv)")
        if not csv_path:
            return
        try:
            self.selected_tars = load_tar_selection_csv(csv_path)
        except Exception as e:
            QMessageBox.critical(self, "CSV error", str(e))
            self.selected_tars = None
            return
        self._append_log(
            f"Loaded {len(self.selected_tars)} TARs marked for processing "
            f"({self._selected_tars_sampling_summary()})"
        )

    def _append_log(self, msg: str):
        self.log.append(msg)

    def pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory", self.output_dir.text())
        if d:
            self.output_dir.setText(d)

    def _sync_model_edit_from_selection(self):
        items = self.models_list.selectedItems()
        if not items:
            return
        blob = items[0].text().strip()
        full_url = f"{self.account_url.text().strip().rstrip('/')}/trainedmodels/{blob}"
        self.model_url_edit.setText(full_url)
        self._append_log(f"Selected model: {blob}")

    def _get_selected_model_blob(self):
        blob = parse_model_input(self.model_url_edit.text().strip())
        if blob:
            return blob
        items = self.models_list.selectedItems()
        if items:
            return items[0].text().strip()
        return None

    def _targets_from_list(self):
        targets = []
        for i in range(self.selected_targets_list.count()):
            txt = self.selected_targets_list.item(i).text()
            if " | " in txt:
                container, prefix = txt.split(" | ", 1)
            else:
                container, prefix = txt, ""
            prefix = prefix.strip()
            if prefix == "(whole container)":
                prefix = ""
            targets.append({"container": container.strip(), "prefix": prefix})
        uniq = []
        seen = set()
        for t in targets:
            key = (t["container"], t.get("prefix") or "")
            if key in seen:
                continue
            seen.add(key)
            uniq.append(t)
        return uniq

    def refresh_containers(self):
        self._append_log("Refreshing containers...")
        self.progress.setValue(0)
        self.containers_list.clear()
        self._container_worker = ContainerListWorker(self.account_url.text().strip())
        self._container_worker.log.connect(self._append_log)
        self._container_worker.containers_ready.connect(self._populate_containers)
        self._container_worker.start()

    def _populate_containers(self, containers):
        self.containers_list.clear()
        for c in containers:
            self.containers_list.addItem(QListWidgetItem(c))

    def add_selected_containers(self):
        prefix = self.prefix_edit.text().strip()
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        for it in self.containers_list.selectedItems():
            c = it.text().strip()
            disp_prefix = prefix if prefix else "(whole container)"
            self.selected_targets_list.addItem(QListWidgetItem(f"{c} | {disp_prefix}"))

    def add_urls(self):
        lines = [ln.strip() for ln in self.url_text.toPlainText().splitlines() if ln.strip()]
        for ln in lines:
            parsed = parse_blob_url(ln)
            if not parsed:
                self._append_log(f"Could not parse URL: {ln}")
                continue
            acct, container, path = parsed
            if acct.rstrip("/") != self.account_url.text().strip().rstrip("/"):
                self._append_log(f"WARNING: URL account {acct} differs from Account URL field. Using Account URL field.")
            prefix = path
            if prefix and not prefix.endswith("/"):
                prefix += "/"
            disp_prefix = prefix if prefix else "(whole container)"
            self.selected_targets_list.addItem(QListWidgetItem(f"{container} | {disp_prefix}"))

    def remove_selected_targets(self):
        for it in self.selected_targets_list.selectedItems():
            row = self.selected_targets_list.row(it)
            self.selected_targets_list.takeItem(row)

    def refresh_models(self):
        self._append_log("Refreshing models...")
        self.progress.setValue(0)
        self.models_list.clear()
        self._model_worker = ModelListWorker(self.account_url.text().strip(), models_container="trainedmodels")
        self._model_worker.log.connect(self._append_log)
        self._model_worker.models_ready.connect(self._populate_models)
        self._model_worker.start()

    def _populate_models(self, models):
        self.models_list.clear()
        for m in models:
            self.models_list.addItem(QListWidgetItem(m))

    def use_model_input(self):
        txt = self.model_url_edit.text().strip()
        self._append_log(f"Model input: {txt}")
        blob = parse_model_input(txt)
        if not blob:
            if txt.lower().startswith("http"):
                self._append_log(f"parse_blob_url() -> {parse_blob_url(txt)}")
            else:
                self._append_log("Input is not a URL and does not look like a .pt blob name")
            self._append_log("Could not parse model url")
            return
        for i in range(self.models_list.count()):
            if self.models_list.item(i).text().strip() == blob:
                self.models_list.setCurrentRow(i)
                break
        self._append_log(f"Using model blob: {blob}")

    def run_application(self):
        if self.rb_validation.isChecked():
            self.start_validation_session()
            return
        targets = self._targets_from_list()
        model_blob = self._get_selected_model_blob()
        out_root = self.output_dir.text().strip()
        os.makedirs(out_root, exist_ok=True)
        self._append_log(f"Running application for {len(targets)} target(s) with model {model_blob}")
        self.progress.setValue(0)
        self.move_btn.setEnabled(False)
        self.validation_btn.setEnabled(False)
        self.rb_validation.setEnabled(False)
        self._last_predictions = None
        self._last_account_url = self.account_url.text().strip()
        self._last_run_dir = None
        if not self.selected_tars:
            QMessageBox.warning(self, "No TAR selection", "Please generate and load a TAR CSV first.")
            return
        sampling_mode = (
            "full TAR skip/sample"
        )
        self._append_log(
            f"TAR selection ready: {len(self.selected_tars)} TAR(s), {self._selected_tars_sampling_summary()} "
            f"(sampling mode: {sampling_mode})."
        )
        self._app_worker = ApplicationWorker(
            account_url=self.account_url.text().strip(),
            targets=targets,
            model_blob=model_blob,
            output_root=out_root,
            selected_tars=self.selected_tars
        )
        self._app_worker.log.connect(self._append_log)
        self._app_worker.progress.connect(self.progress.setValue)
        self._app_worker.finished_ok.connect(self._app_done)
        self._app_worker.failed.connect(self._app_failed)
        self._app_worker.start()

    def _load_predictions_from_per_tar_dir(self, per_tar_dir: str):
        if not per_tar_dir:
            return None
        return load_predictions_from_source(
            per_tar_dir,
            account_url=self._last_account_url or self.account_url.text().strip(),
            log_cb=self._append_log,
        )

    def _set_predictions_ready_state(self, ready: bool):
        self.move_btn.setEnabled(ready)
        self.validation_btn.setEnabled(ready)
        self.rb_validation.setEnabled(ready)

    def _set_bayesian_ready_state(self, ready: bool):
        self.bayesian_btn.setEnabled(ready)

    def run_bayesian_statistics(self):
        if not self._last_run_dir:
            QMessageBox.warning(
                self,
                "No run directory",
                "Run an application workflow first so there is a local run directory to analyse.",
            )
            return
        validation_json = os.path.join(self._last_run_dir, "validation.json")
        if not os.path.exists(validation_json):
            QMessageBox.warning(
                self,
                "Validation session not finished",
                "Complete a validation session first so this run has a validation.json file.",
            )
            return
        output_dir = Path(self._last_run_dir) / "bayesian_statistics"
        self._append_log(
            f"Running Bayesian statistics with timeseries source {self._last_run_dir} and validation JSON {validation_json}"
        )
        try:
            outputs = run_bayesian_timeseries_analysis(
                inference_source=self._last_run_dir,
                validation_source=validation_json,
                output_dir=output_dir,
                mc_samples=250,
                mc_seed=42,
                target_class="bayesian_statistics",
                diagnostics=True,
                diagnostic_prior_draws=500,
                stacked=False,
                log_y=False,
            )
        except Exception as exc:
            self._append_log(f"Bayesian statistics failed: {exc}")
            QMessageBox.critical(self, "Bayesian statistics failed", str(exc))
            return
        self._append_log("Bayesian statistics completed")
        for key, path in outputs.items():
            if path and path.exists():
                self._append_log(f"{key}: {path}")
        timeseries_html = outputs.get("timeseries_html")
        diagnostics_dir = outputs.get("diagnostics_dir")
        if timeseries_html and timeseries_html.exists():
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(timeseries_html)))
            except Exception as exc:
                self._append_log(f"Could not open timeseries HTML: {exc}")
        if diagnostics_dir and diagnostics_dir.exists():
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(diagnostics_dir)))
            except Exception as exc:
                self._append_log(f"Could not open diagnostics folder: {exc}")
        QMessageBox.information(
            self,
            "Bayesian statistics complete",
            "Wrote Bayesian statistics outputs to:\n"
            f"{output_dir}\n\n"
            f"Timeseries HTML: {timeseries_html}\n"
            f"Diagnostics directory: {diagnostics_dir}",
        )

    def _resolve_validation_output_dir(self):
        out_root = self.output_dir.text().strip()
        if not out_root:
            repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            out_root = os.path.join(repo_dir, "edge-ai", "application_runs")
        os.makedirs(out_root, exist_ok=True)
        stamp = utc_now_iso().replace(":", "-")
        out_dir = os.path.join(out_root, f"validation_{stamp}")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def load_predictions_from_source(self):
        source = self.predictions_source_edit.text().strip()
        if not source:
            QMessageBox.warning(self, "No predictions source", "Please enter a local folder or blob container/prefix.")
            return
        self._append_log(f"Loading predictions from source: {source}")
        self._last_predictions = None
        self._last_account_url = self.account_url.text().strip()
        self._last_predictions_source = source
        try:
            preds = load_predictions_from_source(
                source,
                account_url=self._last_account_url,
                log_cb=self._append_log,
            )
        except Exception as exc:
            self._append_log(f"Failed to load predictions from source: {exc}")
            self._last_predictions = None
            self._last_predictions_source = None
            QMessageBox.critical(self, "Predictions load failed", str(exc))
            self._set_predictions_ready_state(False)
            return
        if not preds:
            self._append_log(f"No predictions found in source: {source}")
            QMessageBox.warning(self, "No predictions loaded", "No predictions were found in the supplied source.")
            self._set_predictions_ready_state(False)
            return
        self._last_predictions = preds
        self._append_log(f"Loaded {len(self._last_predictions)} predictions from {source}")
        self._set_predictions_ready_state(True)
        self._append_log(
            "Validation is now available. The session uses adaptive CI-based stopping: "
            "validation continues until overall accuracy CI half-width < target and all "
            "important classes meet their precision threshold."
        )

    def _app_done(self, info: dict):
        self._append_log("Application complete.")
        self._append_log(f"Run folder: {info.get('run_dir')}")
        self._last_run_dir = info.get("run_dir")
        self._last_predictions = None
        try:
            if "predictions_dir" in info:
                self._append_log("Loading per-TAR predictions...")
                preds = self._load_predictions_from_per_tar_dir(info.get("predictions_dir"))
                if not preds:
                    raise RuntimeError("No per-TAR predictions found")
                self._last_predictions = preds
                self._append_log(f"Loaded {len(self._last_predictions)} predictions from per-TAR outputs.")
            else:
                raise RuntimeError("No predictions provided by application worker")
        except Exception as e:
            self._append_log(f"Failed to load predictions: {e}")
            self._last_predictions = None
        ready = self._last_predictions is not None
        self._set_predictions_ready_state(ready)
        if ready:
            self._append_log(
                "Validation is now available. The session uses adaptive CI-based stopping: "
                "validation continues until overall accuracy CI half-width < target and all "
                "important classes meet their precision threshold."
            )

    def _app_failed(self, msg: str):
        self._append_log(f"Application failed: {msg}")
        self.move_btn.setEnabled(False)
        self.validation_btn.setEnabled(False)
        self.rb_validation.setEnabled(False)

    def start_validation_session(self):
        if not self._last_predictions:
            QMessageBox.warning(self, "No predictions loaded", "Run application first so there are predictions to validate.")
            return
        labels = sorted({str(p.get("predicted_label")) for p in self._last_predictions if p.get("predicted_label") not in (None, "ERROR")})
        cfg_dlg = ValidationConfigDialog(known_labels=labels, parent=self)
        if cfg_dlg.exec() != QDialog.Accepted:
            return
        config = cfg_dlg.get_config()
        self._append_log(
            f"Validation config: CI target=±{config.ci_width_target:.0%}, "
            f"important classes={config.important_classes or '(none)'}, "
            f"min precision={config.min_class_precision:.0%}, "
            f"max pool={config.max_sample_size}"
        )
        output_run_dir = self._last_run_dir or self._resolve_validation_output_dir()
        self._append_log(f"Validation outputs will be written to: {output_run_dir}")
        dlg = ValidationSessionDialog(
            account_url=self._last_account_url or self.account_url.text().strip(),
            predictions=self._last_predictions,
            output_run_dir=output_run_dir,
            labels=labels,
            config=config,
            log_cb=self._append_log,
            parent=self,
        )
        dlg.exec()
        if dlg.result() == QDialog.Accepted:
            validation_output_dir = getattr(dlg, "output_run_dir", None) or output_run_dir
            if validation_output_dir:
                self._last_run_dir = validation_output_dir
                validation_json = os.path.join(validation_output_dir, "validation.json")
                if os.path.exists(validation_json):
                    self._set_bayesian_ready_state(True)
                    self._append_log(
                        f"Validation completed. Bayesian statistics are ready for {validation_output_dir}"
                    )
                    self.run_bayesian_statistics()

    def move_by_class(self):
        if not self._last_predictions:
            self._append_log("No predictions loaded - run application first")
            return
        self._append_log("Re-streaming blob TARs and downloading TIFFs into class subfolders...")
        self.progress.setValue(0)
        self._move_worker = MoveByClassWorker(
            account_url=self._last_account_url or self.account_url.text().strip(),
            predictions=self._last_predictions,
            output_run_dir=self._last_run_dir,
        )
        self._move_worker.log.connect(self._append_log)
        self._move_worker.progress.connect(self.progress.setValue)
        self._move_worker.finished_ok.connect(self._move_done)
        self._move_worker.failed.connect(self._move_failed)
        self._move_worker.start()

    def _move_done(self, stats: dict):
        self._append_log(
            f"Download complete: downloaded={stats.get('downloaded', stats.get('moved'))} "
            f"skipped={stats.get('skipped')} failed={stats.get('failed')} total={stats.get('total')} mode={stats.get('mode')}"
        )

    def _move_failed(self, msg: str):
        self._append_log(f"Re-download failed: {msg}")
