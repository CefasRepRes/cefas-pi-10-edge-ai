import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from train import save_confusion_matrix_artifacts
except ModuleNotFoundError as exc:
    if exc.name in {"torch", "torchvision", "matplotlib", "sklearn"}:
        raise unittest.SkipTest(f"skipping because dependency missing: {exc.name}") from exc
    raise


class ConfusionMatrixArtifactTests(unittest.TestCase):
    def test_save_confusion_matrix_artifacts(self):
        matrix = [[3, 1], [0, 4]]
        class_labels = ["copepod", "other"]

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "confusion_matrix.csv")
            png_path = os.path.join(tmpdir, "confusion_matrix.png")

            info = save_confusion_matrix_artifacts(matrix, class_labels, csv_path, png_path)

            self.assertEqual(info["csv"], csv_path)
            self.assertEqual(info["png"], png_path)
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(png_path))
            with open(csv_path, "r", encoding="utf-8") as handle:
                self.assertEqual(handle.readline().rstrip(), "true_label,copepod,other")


if __name__ == "__main__":
    unittest.main()
