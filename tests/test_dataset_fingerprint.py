from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class DatasetFingerprintTest(unittest.TestCase):
    def test_fingerprint_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_dir = root / "data"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "train.json").write_text("[1,2,3,4,5,6]")
            out1 = root / "fp1.json"
            out2 = root / "fp2.json"

            cmd = [
                "python",
                "scripts/13_dataset_fingerprint.py",
                "--dataset-id",
                "test",
                "--path",
                str(dataset_dir),
                "--split",
                "train",
                "--out",
                str(out1),
            ]
            rc1 = subprocess.run(cmd, check=False)
            self.assertEqual(rc1.returncode, 0)

            cmd[-1] = str(out2)
            rc2 = subprocess.run(cmd, check=False)
            self.assertEqual(rc2.returncode, 0)

            p1 = json.loads(out1.read_text())
            p2 = json.loads(out2.read_text())
            self.assertEqual(p1["dataset"]["sha256"], p2["dataset"]["sha256"])
            self.assertEqual(p1["dataset"]["num_tokens"], 6)


if __name__ == "__main__":
    unittest.main()
