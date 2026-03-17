from __future__ import annotations

import unittest

from ttt.research.protocol_760m import (
    CONTROL_760M_STAGE_IDS,
    CORE_760M_STAGE_IDS,
    FAITHFUL_ADAPT_GLOBAL_BATCH_SIZE,
    FAITHFUL_ADAPT_STEPS,
    FAITHFUL_EXT_GLOBAL_BATCH_SIZE,
    FAITHFUL_EXT_STEPS,
    REVISED_8X_H200_GLOBAL_BATCH_SIZE,
    build_protocol_r_760m_manifest,
    build_protocol_r_760m_stage_map,
    scaled_steps_for_token_budget,
)


class Protocol760MTest(unittest.TestCase):
    def test_scaled_steps_preserve_token_budget_exactly(self) -> None:
        self.assertEqual(
            scaled_steps_for_token_budget(
                original_steps=FAITHFUL_EXT_STEPS,
                original_global_batch_size=FAITHFUL_EXT_GLOBAL_BATCH_SIZE,
                revised_global_batch_size=REVISED_8X_H200_GLOBAL_BATCH_SIZE,
            ),
            2900,
        )
        self.assertEqual(
            scaled_steps_for_token_budget(
                original_steps=FAITHFUL_ADAPT_STEPS,
                original_global_batch_size=FAITHFUL_ADAPT_GLOBAL_BATCH_SIZE,
                revised_global_batch_size=REVISED_8X_H200_GLOBAL_BATCH_SIZE,
            ),
            23200,
        )

    def test_stage_map_uses_revised_batch_for_all_runtime_stages(self) -> None:
        stage_map = build_protocol_r_760m_stage_map()
        self.assertEqual(stage_map["S0"].revised_global_batch_size, 8)
        self.assertEqual(stage_map["S1"].revised_global_batch_size, 8)
        self.assertEqual(stage_map["S2_ADAPT"].revised_global_batch_size, 8)
        self.assertEqual(stage_map["S2"].revised_global_batch_size, 8)
        self.assertEqual(stage_map["S3"].revised_global_batch_size, 8)
        self.assertEqual(stage_map["S0"].revised_total_steps, 2900)
        self.assertEqual(stage_map["S2_ADAPT"].revised_total_steps, 23200)

    def test_protocol_manifest_records_effective_step_scaling(self) -> None:
        manifest = build_protocol_r_760m_manifest(
            paper_run_id="protocol_r_760m_author_seed_v1",
            exp_folder="protocol_r_760m_author_seed_v1",
        )
        self.assertEqual(manifest["revised_global_batch_size"], 8)
        self.assertEqual(manifest["effective_ext_steps"], 2900)
        self.assertEqual(manifest["effective_adapt_steps"], 23200)
        self.assertEqual(manifest["stages"]["S3"]["author_seed_key"], "760m_e2e")
        self.assertEqual(manifest["execution_tranches"]["core"], ["S2_ADAPT", "S2", "S3"])

    def test_stage_groups_match_staged_760m_execution_plan(self) -> None:
        self.assertEqual(CORE_760M_STAGE_IDS, ("S2_ADAPT", "S2", "S3"))
        self.assertEqual(CONTROL_760M_STAGE_IDS, ("S0", "S1"))


if __name__ == "__main__":
    unittest.main()
