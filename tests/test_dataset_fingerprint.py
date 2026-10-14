from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_SCRIPT = REPO_ROOT / "scripts" / "13_dataset_fingerprint.py"
TOKEN_DATA_SCRIPT = REPO_ROOT / "scripts" / "04_make_token_data.py"


def _run_script(args: list[str]) -> subprocess.CompletedProcess:
    # Use this interpreter (not whatever `python` is on PATH) and import `ttt`
    # from this checkout.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), *[p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]]
    )
    return subprocess.run(
        [sys.executable, *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=env,
    )


class DatasetFingerprintTest(unittest.TestCase):
    def test_fingerprint_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dataset_dir = root / "data"
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "train.json").write_text("[1,2,3,4,5,6]")
            out1 = root / "fp1.json"
            out2 = root / "fp2.json"

            args = [
                str(FINGERPRINT_SCRIPT),
                "--dataset-id",
                "test",
                "--path",
                str(dataset_dir),
                "--split",
                "train",
                "--out",
                str(out1),
            ]
            rc1 = _run_script(args)
            self.assertEqual(rc1.returncode, 0, rc1.stderr)

            args[-1] = str(out2)
            rc2 = _run_script(args)
            self.assertEqual(rc2.returncode, 0, rc2.stderr)

            p1 = json.loads(out1.read_text())
            p2 = json.loads(out2.read_text())
            self.assertEqual(p1["dataset"]["sha256"], p2["dataset"]["sha256"])
            self.assertEqual(p1["dataset"]["num_tokens"], 6)

    def test_toy_generator_writes_sidecars_matching_fingerprint_script(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td) / "tokens"
            gen = _run_script(
                [str(TOKEN_DATA_SCRIPT), "--out", str(data_dir), "--train-tokens", "300", "--val-tokens", "200"]
            )
            self.assertEqual(gen.returncode, 0, gen.stderr)

            for split, num_tokens in (("train", 300), ("val", 200)):
                sidecar = json.loads((data_dir / f"{split}.fingerprint.json").read_text())
                self.assertEqual(sidecar["dataset"]["num_tokens"], num_tokens)
                self.assertEqual(sidecar["dataset"]["split"], split)

                out = Path(td) / f"{split}.check.json"
                fp = _run_script(
                    [
                        str(FINGERPRINT_SCRIPT),
                        "--dataset-id",
                        "check",
                        "--path",
                        str(data_dir),
                        "--split",
                        split,
                        "--out",
                        str(out),
                    ]
                )
                self.assertEqual(fp.returncode, 0, fp.stderr)
                self.assertEqual(
                    json.loads(out.read_text())["dataset"]["sha256"],
                    sidecar["dataset"]["sha256"],
                )


if __name__ == "__main__":
    unittest.main()
