"""Fusion wiring: adapter Tier-1 reads the brain's escalation ladder via cache thread."""
import os
import time
import unittest
from unittest.mock import patch

import numpy

os.environ["LOCAL_STRATEGY_DIR"] = ""  # never boot the model inside unit tests

from adapters.clash_adapter import ClashBattleAdapter
from clash_jev.state import BattleState, HandCard, LaneView, Towers
from profile_manager import DEFAULT_PROFILES

CLASH_PROFILE = next(p for p in DEFAULT_PROFILES if p.id == "mobile_clash_royale")


def make_state(elixir=5, elapsed_s=60.0, left=None, right=None):
    return BattleState(
        elapsed_s=elapsed_s,
        elixir=elixir,
        hand=(HandCard(0, "knight", True),),
        left=left or LaneView(),
        right=right or LaneView(),
        towers=Towers(),
        seconds_at_full_elixir=0.0,
        units=(),
        enemy_elixir_estimate=None,
    )


def make_adapter():
    adapter = ClashBattleAdapter()
    adapter.jev_client = None  # cloud tier stays offline in unit tests
    return adapter


class FusionMapTests(unittest.TestCase):
    def test_deploy_actions_map_to_lane_strategies(self):
        map_fn = ClashBattleAdapter._fusion_map_strategy
        self.assertEqual(map_fn("deploy_defense_center", 0.0), "defend_centre")
        self.assertEqual(map_fn("wait", 0.0), "save_elixir")

    def test_lane_deploy_is_intent_resolved_by_threat(self):
        map_fn = ClashBattleAdapter._fusion_map_strategy
        self.assertEqual(map_fn("deploy_card_left", 0.1), "push_left")
        self.assertEqual(map_fn("deploy_card_right", 0.1), "push_right")
        self.assertEqual(map_fn("deploy_card_left", 0.6), "defend_left")
        self.assertEqual(map_fn("deploy_card_right", 0.6), "defend_right")

    def test_unmappable_actions_skip_the_cache(self):
        map_fn = ClashBattleAdapter._fusion_map_strategy
        for action in ("deploy_clash_card", "deploy_spell_center", "start_battle", "confirm_ok", "nonsense"):
            self.assertIsNone(map_fn(action, 0.0), action)

    def test_lane_threat_matches_decide_derivation(self):
        state = make_state()
        self.assertEqual(ClashBattleAdapter._fusion_lane_threat(state), 0.0)
        state = make_state(left=LaneView(enemy_on_my_side=2))
        self.assertAlmostEqual(ClashBattleAdapter._fusion_lane_threat(state), 0.71, places=5)


class FusionRefreshTests(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter()
        self.captured = {}

    def _wire(self, result):
        def fake_query(profile, scene):
            self.captured["profile"] = profile
            self.captured["scene"] = scene
            return result

        self.adapter._fusion_query = fake_query
        self.adapter._fusion_profile = CLASH_PROFILE

    def test_refresh_builds_scene_and_caches_mapped_strategy(self):
        self._wire({
            "action": "deploy_defense_center",
            "confidence": 0.88,
            "threat_score": 0.6,
            "source": "laya_jev_consensus",
            "latency_ms": 5.0,
        })
        self.adapter._last_state = make_state(elixir=7)
        self.assertTrue(self.adapter._fusion_refresh_once())
        snap = self.adapter._fusion_snapshot()
        self.assertEqual(snap["strategy"], "defend_centre")
        self.assertEqual(snap["conf"], 0.88)
        self.assertEqual(snap["threat"], 0.6)
        self.assertEqual(self.captured["profile"].id, "mobile_clash_royale")
        self.assertEqual(self.captured["scene"].elixir, 7)
        self.assertEqual(self.captured["scene"].game_phase, "in_battle")

    def test_refresh_without_state_is_a_noop(self):
        self._wire({"action": "wait", "confidence": 0.9})
        self.adapter._last_state = None
        self.assertFalse(self.adapter._fusion_refresh_once())
        self.assertIsNone(self.adapter._fusion_snapshot())

    def test_query_exception_records_error_not_cache(self):
        def boom(profile, scene):
            raise RuntimeError("ladder down")

        self.adapter._fusion_query = boom
        self.adapter._fusion_profile = CLASH_PROFILE
        self.adapter._last_state = make_state()
        self.assertFalse(self.adapter._fusion_refresh_once())
        self.assertIn("ladder down", self.adapter._fusion_error)
        self.assertIsNone(self.adapter._fusion_snapshot())

    def test_unmappable_action_caches_none_strategy(self):
        self._wire({"action": "deploy_spell_center", "confidence": 0.9})
        self.adapter._last_state = make_state()
        self.assertTrue(self.adapter._fusion_refresh_once())
        self.assertIsNone(self.adapter._fusion_pick(False))


class FusionPickGateTests(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter()

    def _prime(self, strategy="defend_centre", conf=0.9, threat=0.1, age_s=0.0):
        with self.adapter._fusion_lock:
            self.adapter._fusion_cache = {
                "strategy": strategy,
                "conf": conf,
                "threat": threat,
                "source": "laya_jev_consensus",
                "ts": time.time() - age_s,
            }

    def test_fresh_confident_cache_picks(self):
        self._prime()
        strategy, conf, threat = self.adapter._fusion_pick(False)
        self.assertEqual(strategy, "defend_centre")
        self.assertEqual(conf, 0.9)
        self.assertEqual(threat, 0.1)

    def test_stale_cache_picks_nothing(self):
        self._prime(age_s=9.0)
        self.assertIsNone(self.adapter._fusion_pick(False))

    def test_low_confidence_picks_nothing(self):
        self._prime(conf=0.4)
        self.assertIsNone(self.adapter._fusion_pick(False))

    def test_immediate_contact_blocks_fusion(self):
        self._prime()
        self.assertIsNone(self.adapter._fusion_pick(True))

    def test_empty_cache_picks_nothing(self):
        self.assertIsNone(self.adapter._fusion_pick(False))


class FusionDecideTests(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter()
        self.state = make_state(elixir=5, elapsed_s=60.0)

    def _decide(self):
        frame = numpy.zeros((32, 32, 3), dtype=numpy.uint8)
        with patch.object(self.adapter, "extract_state_from_frame", return_value=self.state), \
                patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}):
            return self.adapter.decide(frame, threat_urgency=0.1)

    def _prime(self, strategy, conf=0.9, threat=0.0):
        with self.adapter._fusion_lock:
            self.adapter._fusion_cache = {
                "strategy": strategy,
                "conf": conf,
                "threat": threat,
                "source": "laya_jev_consensus",
                "ts": time.time(),
            }

    def test_fusion_pick_wins_tier1(self):
        self._prime("save_elixir", threat=0.15)
        res = self._decide()
        self.assertEqual(res["source"], "fusion_local")
        self.assertEqual(res["strategy"], "save_elixir")
        self.assertEqual(res["action"], "wait")
        # Ladder threat mirrors up when it exceeds the vision urgency.
        self.assertAlmostEqual(res["threat_score"], 0.15, places=5)

    def test_ghost_defend_post_validates_to_reflex(self):
        self._prime("defend_centre")
        res = self._decide()
        self.assertEqual(res["source"], "tactical_reflex")
        self.assertNotIn("defend", res["strategy"])

    def test_low_conf_fusion_falls_through_to_reflex(self):
        self._prime("push_left", conf=0.30)
        res = self._decide()
        self.assertEqual(res["source"], "tactical_reflex")

    def test_immediate_contact_prefers_reflex_over_fusion(self):
        self._prime("save_elixir")
        self.state = make_state(elixir=5, left=LaneView(enemy_on_my_side=1))
        res = self._decide()
        self.assertEqual(res["source"], "tactical_reflex")


class BrainAttachTests(unittest.TestCase):
    def test_brain_wires_fusion_into_clash_adapter(self):
        import universal_brain as brain_module

        with patch.object(brain_module.threading.Thread, "start"), \
                patch.object(brain_module.UniversalBrain, "_start_laya_loader"), \
                patch.object(brain_module.UniversalBrain, "_start_llm2jev_loader"), \
                patch.object(brain_module.UniversalBrain, "_start_openjev_loader"):
            brain = brain_module.UniversalBrain()
        adapter = brain.clash_adapter
        if adapter is None:
            self.skipTest("clash adapter unavailable in this build")
        self.assertIsNotNone(adapter._fusion_query)
        self.assertIs(adapter._fusion_query.__self__, brain)
        self.assertEqual(adapter._fusion_profile.id, "mobile_clash_royale")
        self.assertIsNotNone(adapter._fusion_thread)
        self.assertEqual(adapter._fusion_thread.name, "clash-fusion")


if __name__ == "__main__":
    unittest.main()
