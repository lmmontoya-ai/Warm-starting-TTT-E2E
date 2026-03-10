from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class B1B2RunnerTest(unittest.TestCase):
    def test_dry_run_command_chain(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "25_run_b1_b2_real.py"

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            dclm = tmp / "dclm"
            books = tmp / "books"
            dclm.mkdir(parents=True, exist_ok=True)
            books.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(script),
                "--dclm-root",
                str(dclm),
                "--books-root",
                str(books),
                "--dry-run",
                "--skip-figures",
                "--skip-bundle",
            ]
            result = subprocess.run(
                cmd,
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            stdout = result.stdout
            self.assertIn("scripts/23_warmstart_registry.py", stdout)
            self.assertIn("scripts/18_eval_matrix.py", stdout)
            self.assertIn("scripts/24_eval_ruler.py", stdout)
            self.assertIn("scripts/20_make_paper_tables.py", stdout)


if __name__ == "__main__":
    unittest.main()
