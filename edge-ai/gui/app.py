import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget
)

from edge_tab import EdgeAITab
from train_tab import ModelTrainingTab
from application_validation_tab import ApplicationValidationTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plankton Imager – Edge AI")
        self.resize(1000, 700)

        tabs = QTabWidget()
        tabs.addTab(EdgeAITab(), "Edge AI")
        tabs.addTab(ModelTrainingTab(), "Model Training")
        tabs.addTab(ApplicationValidationTab(), "Apply trained model and validate")

        self.setCentralWidget(tabs)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()