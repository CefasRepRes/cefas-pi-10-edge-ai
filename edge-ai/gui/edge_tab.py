from shlex import split as shlex_split

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QLabel,
    QLineEdit,
)

from worker import EdgeAIWorker


DEFAULT_EDGE_AI_COMMAND = (
    "python edge-ai.py model2026-07-21T09-12-07Z -b 50 -t --subsampling_rate 10 --sendoneimage"
)

class EdgeAITab(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)

        self.args_label = QLabel("Edge AI command / args:")
        self.args_input = QLineEdit()
        self.args_input.setText(DEFAULT_EDGE_AI_COMMAND)
        self.args_input.setPlaceholderText(
            "e.g. python edge-ai.py model2026-07-21T09-12-07Z -b 50 -t --subsampling_rate 10 --sendoneimage"
        )

        args_row = QHBoxLayout()
        args_row.addWidget(self.args_label)
        args_row.addWidget(self.args_input)

        self.sas_label = QLabel("SAS token (paste this BEFORE running if you are downloading a new model):")
        self.sas_input = QLineEdit()
        self.sas_input.setPlaceholderText("Paste SAS token here (e.g. ?sv=2025-...)")
        self.sas_input.setEchoMode(QLineEdit.EchoMode.Password)

        sas_row = QHBoxLayout()
        sas_row.addWidget(self.sas_label)
        sas_row.addWidget(self.sas_input)

        self.start_btn = QPushButton("Start Edge AI")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.layout.addLayout(args_row)
        self.layout.addLayout(sas_row)
        self.layout.addWidget(self.start_btn)
        self.layout.addWidget(self.stop_btn)
        self.layout.addWidget(self.log)

        self.worker = None

        self.start_btn.clicked.connect(self.start_edge_ai)
        self.stop_btn.clicked.connect(self.stop_edge_ai)

    def start_edge_ai(self):
        raw = self.args_input.text().strip()

        if raw.startswith("python "):
            parts = shlex_split(raw)
            argv = parts[2:] if len(parts) >= 3 else []
        else:
            argv = shlex_split(raw) if raw else []

        sas_token = self.sas_input.text().strip()
        if sas_token and "--sas-token" not in argv:
            argv += ["--sas-token", sas_token]

        self.log.append(f"Starting Edge AI with args: {' '.join(argv)}")

        self.worker = EdgeAIWorker(argv=argv)
        self.worker.output.connect(self.log.append)
        self.worker.finished.connect(self.edge_ai_stopped)

        self.worker.start()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_edge_ai(self):
        if self.worker:
            self.worker.stop()

    def edge_ai_stopped(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log.append("Edge AI stopped.")