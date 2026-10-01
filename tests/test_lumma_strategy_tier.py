"""Local Lumma strategy tier: async cache freshness, confidence gate, slim payload, error paths."""
import os
import time
import unittest
from unittest.mock import patch

from adapters.clash_adapter import (
    SLIM_STRATEGY_INSTRUCTIONS,
    ClashBattleAdapter,
    _slim_strategy_criteria,
)
from clash_jev.strategies import STRATEGIES


class _FakeModel:
    def decide(self, state, questions):
        assert "strategy" in questions
        q = questions["strategy"]
        assert q["type"] == "choice"
        assert set(q["criteria"]) == {s.name for s in STRATEGIES}
        return {
            "strategy": {
                "type": "choice",
                "choice": "push_left",
                "confidence": 0.4,
                "probabilities": {
                    "push_left": 0.42,
                    "push_right": 0.21,
                    "save_elixir": 0.1,
                },
            }
        }


class LummaTierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("LOCAL_STRATEGY_DIR", None)
        cls.ad = ClashBattleAdapter()

    def setUp(self):
        self.ad._lumma_cache = None
        self.ad._lumma_error = None

    def test_no_dir_means_no_thread(self):
        self.assertIsNone(self.ad._lumma_thread)

    def test_slim_criteria_cover_all_strategies(self):
        crit = _slim_strategy_criteria()
        self.assertEqual(set(crit), {s.name for s in STRATEGIES})
        for name, text in crit.items():
            self.assertTrue(text, name)
            self.assertLessEqual(len(text), 90, name)
        self.assertIn("elixir", SLIM_STRATEGY_INSTRUCTIONS)

    def test_refresh_populates_cache(self):
        self.ad._last_state = object()
        with patch("clash_jev.policy.build_state", return_value="state-json"):
            ok = self.ad._lumma_refresh_once(_FakeModel())
        self.assertTrue(ok)
        snap = self.ad._lumma_snapshot()
        self.assertIsNotNone(snap)
        self.assertEqual(snap["strategy"], "push_left")
        self.assertAlmostEqual(snap["conf"], 0.42)

    def test_refresh_skips_without_state(self):
        self.ad._last_state = None
        self.assertFalse(self.ad._lumma_refresh_once(_FakeModel()))
        self.assertIsNone(self.ad._lumma_snapshot())

    def test_snapshot_goes_stale(self):
        self.ad._lumma_cache = {
            "strategy": "cycle",
            "conf": 0.9,
            "ts": time.time() - 99,
            "ms": 1,
        }
        self.assertIsNone(self.ad._lumma_snapshot())

    def test_pick_gate_matrix(self):
        now = time.time()
        self.ad._lumma_cache = {"strategy": "cycle", "conf": 0.9, "ts": now, "ms": 1}
        self.assertEqual(self.ad._lumma_pick(immediate_contact=False), ("cycle", 0.9))
        self.assertIsNone(self.ad._lumma_pick(immediate_contact=True))

        self.ad._lumma_cache = {"strategy": "cycle", "conf": 0.1, "ts": now, "ms": 1}
        self.assertIsNone(self.ad._lumma_pick(immediate_contact=False))

        self.ad._lumma_cache = {
            "strategy": "yeet_the_tower",
            "conf": 0.9,
            "ts": now,
            "ms": 1,
        }
        self.assertIsNone(self.ad._lumma_pick(immediate_contact=False))

    def test_loop_load_failure_sets_error_and_returns(self):
        with patch.object(self.ad, "_lumma_load", side_effect=RuntimeError("no model")):
            self.ad._lumma_loop()
        self.assertIn("load", self.ad._lumma_error or "")

    def test_loop_records_refresh_errors_and_keeps_going(self):
        calls = {"n": 0}

        def fake_refresh(model):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")
            raise SystemExit  # BaseException escapes the loop — used as a stop token

        with patch.object(self.ad, "_lumma_load", return_value=_FakeModel()), patch.object(
            self.ad, "_lumma_refresh_once", side_effect=fake_refresh
        ), patch("time.sleep", return_value=None):
            with self.assertRaises(SystemExit):
                self.ad._lumma_loop()
        self.assertGreaterEqual(calls["n"], 2)
        self.assertIn("refresh", self.ad._lumma_error or "")


if __name__ == "__main__":
    unittest.main()
