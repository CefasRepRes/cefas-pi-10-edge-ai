from PySide6.QtCore import QThread, Signal
import subprocess
import sys
import os


class EdgeAIWorker(QThread):
    output = Signal(str)

    def __init__(self, argv=None):
        super().__init__()
        self.process = None
        self._stop_requested = False
        self.argv = argv or ["--classify", "--send"]

    def run(self):
        edge_ai_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "edge-ai.py")
        )

        cmd = [sys.executable, edge_ai_path, *self.argv]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )

        for line in self.process.stdout:
            if self._stop_requested:
                break
            self.output.emit(line.rstrip())

        self.process.terminate()
        self.process.wait()

    def stop(self):
        self._stop_requested = True
        if self.process:
            self.process.terminate()