from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from ttt.research.registry import load_registry, select_stages


class RegistryTest(unittest.TestCase):
    def test_load_and_select(self) -> None:
        content = textwrap.dedent(
            """
            schema_version: "1.0"
            paper_id: test
            datasets:
              d1:
                dataset_id: d1
                path: /tmp/d1
                split: train
                tokenizer_id: tok
                tokenizer_revision: rev
                num_tokens: 10
                sha256: abc
            eval_profiles:
              e1:
                eval_id: e1
                contexts: [8192]
                datasets: [d1]
                eval_split: val
            stages:
              - stage_id: S0
                kind: pretrain
                model_key: m1
                path_group: scratch
                experiment: foo/bar
                dataset_ids: [d1]
                required_parent_checkpoint_ids: []
                required_profile_keys: []
              - stage_id: S1
                kind: ext
                model_key: m1
                path_group: adapter
                experiment: foo/baz
                dataset_ids: [d1]
                required_parent_checkpoint_ids: [S0]
                required_profile_keys: []
            """
        ).strip()

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "registry.yaml"
            path.write_text(content)
            registry = load_registry(path)

            self.assertEqual(registry.paper_id, "test")
            self.assertEqual(len(registry.stages), 2)

            selected = select_stages(registry, model_key="m1", path_group="adapter")
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].stage_id, "S1")


if __name__ == "__main__":
    unittest.main()
