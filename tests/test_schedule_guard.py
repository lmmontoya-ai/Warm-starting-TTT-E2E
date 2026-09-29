from __future__ import annotations

import unittest

from ttt.jax_runtime.schedule_guard import (
    ScheduleValidationError,
    enforce_lr_schedule,
    lr_schedule_problems,
    warmup_cosine_lr,
)


def _problems(**kwargs):
    base = dict(total_steps=120, peak_lr=4e-4, init_lr=0.0, end_lr=1e-5, warmup_steps=12, decay_steps=120)
    base.update(kwargs)
    return lr_schedule_problems(**base)


class ScheduleGuardTest(unittest.TestCase):
    def test_published_extension_recipe_passes(self) -> None:
        # Tandon et al.: 120 extension steps, 10% warmup, cosine to the last step.
        self.assertEqual(_problems(), [])

    def test_revision_v2_extension_is_rejected(self) -> None:
        # 480 steps with the 120-step schedule: the LR sits at its floor for 360 steps.
        problems = _problems(total_steps=480)
        self.assertEqual(len(problems), 1)
        self.assertIn("stays there for the last 360 steps (75% of the run)", problems[0])

    def test_revision_v2_bridge_is_rejected(self) -> None:
        # 480-step bridge with the 4800-step pretraining schedule: warmup covers the whole run.
        problems = _problems(total_steps=480, peak_lr=3e-3, warmup_steps=480, decay_steps=4800)
        self.assertEqual(len(problems), 1)
        self.assertIn("ends inside the warmup", problems[0])

    def test_truncated_cosine_is_rejected(self) -> None:
        # E3-style bridge: warmup finishes but the cosine is cut off at a high LR.
        problems = _problems(total_steps=1920, peak_lr=3e-3, warmup_steps=480, decay_steps=4800)
        self.assertEqual(len(problems), 1)
        self.assertIn("stops before the cosine finishes", problems[0])

    def test_inner_lr_warmup_longer_than_run_is_rejected(self) -> None:
        problems = _problems(total_steps=480, peak_lr=3e-3, warmup_steps=48, decay_steps=480, ilr_warmup_steps=480, ilr_init=0.1, ilr_target=1.0)
        self.assertEqual(len(problems), 1)
        self.assertIn("inner learning rate is still warming up", problems[0])

    def test_scaled_bridge_passes(self) -> None:
        self.assertEqual(
            _problems(total_steps=480, peak_lr=1e-3, warmup_steps=48, decay_steps=480, ilr_warmup_steps=48, ilr_init=0.1, ilr_target=1.0),
            [],
        )

    def test_zero_lr_is_ignored(self) -> None:
        self.assertEqual(_problems(total_steps=480, peak_lr=0.0), [])

    def test_lr_formula_matches_optax(self) -> None:
        import optax

        schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=4e-4, warmup_steps=12, decay_steps=120, end_value=1e-5)
        for step in (0, 5, 12, 30, 60, 119, 200):
            self.assertAlmostEqual(
                warmup_cosine_lr(step, init_lr=0.0, peak_lr=4e-4, end_lr=1e-5, warmup_steps=12, decay_steps=120),
                float(schedule(step)),
                places=9,
            )

    def test_enforce_modes(self) -> None:
        problems = _problems(total_steps=480)
        with self.assertRaises(ScheduleValidationError):
            enforce_lr_schedule(problems, mode="error", label="ext")
        enforce_lr_schedule(problems, mode="warn", label="ext")
        enforce_lr_schedule(problems, mode="off", label="ext")
        enforce_lr_schedule([], mode="error", label="ext")
        with self.assertRaises(ValueError):
            enforce_lr_schedule(problems, mode="loud", label="ext")


if __name__ == "__main__":
    unittest.main()
