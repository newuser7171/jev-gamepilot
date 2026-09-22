"""Regression: stale-cache fallback, coerce_action_name, path-gated duck reflex."""
import unittest
from unittest.mock import patch

import universal_brain as brain_module
from profile_manager import GameProfile, GameAction, DEFAULT_PROFILES
from universal_vision import UniversalSceneState, UniversalEntity

RUNNER = next(p for p in DEFAULT_PROFILES if p.id == "runner_3lane")


def make_scene(
    threat_urgency=0.0,
    recommended_action="wait",
    player=None,
    threats=None,
    targets=None,
):
    return UniversalSceneState(
        player=player,
        threats=threats or [],
        targets=targets or [],
        nearest_threat=(threats[0] if threats else None),
        threat_urgency=threat_urgency,
        recommended_action=recommended_action,
    )


class CoerceActionNameTests(unittest.TestCase):
    def test_valid_name_passthrough(self):
        self.assertEqual(
            brain_module.UniversalBrain.coerce_action_name(RUNNER, "dodge_left"),
            "dodge_left",
        )

    def test_empty_returns_wait(self):
        self.assertEqual(
            brain_module.UniversalBrain.coerce_action_name(RUNNER, ""),
            "wait",
        )

    def test_unknown_name_mapped_by_substring(self):
        self.assertEqual(
            brain_module.UniversalBrain.coerce_action_name(RUNNER, "quick_dodge_left_now"),
            "dodge_left",
        )

    def test_unrelated_garbage_falls_back_to_scene_recommendation(self):
        scene = make_scene(recommended_action="jump")
        self.assertEqual(
            brain_module.UniversalBrain.coerce_action_name(RUNNER, "zzz_nonsense", scene),
            "jump",
        )

    def test_unrelated_garbage_without_scene_returns_wait(self):
        self.assertEqual(
            brain_module.UniversalBrain.coerce_action_name(RUNNER, "zzz_nonsense"),
            "wait",
        )


class StaleCacheFallbackTests(unittest.TestCase):
    def setUp(self):
        with patch.object(brain_module.threading.Thread, "start"), \
                patch.object(brain_module.UniversalBrain, "_start_laya_loader"):
            self.brain = brain_module.UniversalBrain()

    def test_stale_cache_with_live_threat_returns_local_heuristic(self):
        # Bug: last_decision stuck at wait/init while a real threat is on screen.
        self.brain.last_decision = {"action": "wait", "confidence": 0.0, "source": "init"}
        player = UniversalEntity(x=100, y=200, w=40, h=60, entity_type="player")
        threat = UniversalEntity(x=140, y=240, w=30, h=50, entity_type="threat")
        scene = make_scene(
            threat_urgency=0.8,
            recommended_action="wait",
            player=player,
            threats=[threat],
        )
        with patch.object(brain_module.UniversalBrain, "_evaluate_intelligent_reflex", return_value=None):
            decision = self.brain.get_action(RUNNER, scene)
        self.assertNotEqual(decision.get("action"), "wait")
        self.assertEqual(decision.get("source"), "local_scene_heuristic")
        self.assertNotEqual(decision.get("source"), "init")

    def test_stale_cache_clear_scene_with_recommended_action_uses_it(self):
        self.brain.last_decision = {"action": "wait", "confidence": 0.0, "source": "init"}
        scene = make_scene(threat_urgency=0.0, recommended_action="dodge_right")
        with patch.object(brain_module.UniversalBrain, "_evaluate_intelligent_reflex", return_value=None):
            decision = self.brain.get_action(RUNNER, scene)
        self.assertEqual(decision.get("action"), "dodge_right")
        self.assertEqual(decision.get("source"), "local_scene_heuristic")

    def test_fresh_cloud_cache_is_reused_when_not_blind_to_threat(self):
        self.brain.last_decision = {
            "action": "dodge_left",
            "confidence": 0.95,
            "source": "laya_jev_consensus",
            "threat_score": 0.7,
            "target_coords": None,
        }
        player = UniversalEntity(x=100, y=200, w=40, h=60, entity_type="player")
        threat = UniversalEntity(x=140, y=240, w=30, h=50, entity_type="threat")
        scene = make_scene(threat_urgency=0.8, player=player, threats=[threat])
        with patch.object(brain_module.UniversalBrain, "_evaluate_intelligent_reflex", return_value=None):
            decision = self.brain.get_action(RUNNER, scene)
        self.assertEqual(decision.get("action"), "dodge_left")
        self.assertEqual(decision.get("source"), "laya_jev_consensus")

    def test_cloud_decision_blind_to_urgent_threat_falls_back_local(self):
        self.brain.last_decision = {
            "action": "maintain_course",
            "confidence": 0.95,
            "source": "cloud",
            "threat_score": 0.1,
            "target_coords": None,
        }
        player = UniversalEntity(x=100, y=200, w=40, h=60, entity_type="player")
        threat = UniversalEntity(x=130, y=240, w=30, h=50, entity_type="threat")
        scene = make_scene(threat_urgency=0.9, player=player, threats=[threat])
        with patch.object(brain_module.UniversalBrain, "_evaluate_intelligent_reflex", return_value=None):
            decision = self.brain.get_action(RUNNER, scene)
        self.assertEqual(decision.get("source"), "local_scene_heuristic")
        self.assertNotEqual(decision.get("action"), "maintain_course")


class PathGatedReflexTests(unittest.TestCase):
    """smart_duck/smart_jump must not fire for threats outside the player's path."""

    def setUp(self):
        with patch.object(brain_module.threading.Thread, "start"), \
                patch.object(brain_module.UniversalBrain, "_start_laya_loader"):
            self.brain = brain_module.UniversalBrain()
        self.runner_dino = next(p for p in DEFAULT_PROFILES if p.id == "runner_dino")

    def test_midheight_threat_outside_horizontal_path_does_not_duck(self):
        player = UniversalEntity(x=100, y=240, w=40, h=60, entity_type="player")
        # Far above the player and horizontally distant: old height-only gate fired here.
        far_threat = UniversalEntity(x=600, y=100, w=60, h=40, entity_type="threat")
        scene = make_scene(threat_urgency=0.9, player=player, threats=[far_threat])
        reflex = self.brain._evaluate_intelligent_reflex(self.runner_dino, scene)
        if reflex is not None:
            self.assertNotIn(reflex.get("action"), {"duck", "slide"})

    def test_ground_threat_far_off_path_does_not_jump(self):
        player = UniversalEntity(x=100, y=240, w=40, h=60, entity_type="player")
        far_threat = UniversalEntity(x=900, y=270, w=40, h=60, entity_type="threat")
        scene = make_scene(threat_urgency=0.9, player=player, threats=[far_threat])
        reflex = self.brain._evaluate_intelligent_reflex(self.runner_dino, scene)
        if reflex is not None:
            self.assertNotEqual(reflex.get("action"), "jump")

    def test_ground_threat_in_path_still_jumps(self):
        player = UniversalEntity(x=100, y=240, w=40, h=60, entity_type="player")
        threat = UniversalEntity(x=130, y=270, w=40, h=60, entity_type="threat")
        scene = make_scene(threat_urgency=0.9, player=player, threats=[threat])
        reflex = self.brain._evaluate_intelligent_reflex(self.runner_dino, scene)
        self.assertIsNotNone(reflex)
        self.assertEqual(reflex.get("action"), "jump")


class ClassifierCircuitBreakerTests(unittest.TestCase):
    def test_http_403_marks_classifier_dead(self):
        import urllib.error
        with patch.object(brain_module.threading.Thread, "start"), \
                patch.object(brain_module.UniversalBrain, "_start_laya_loader"):
            brain = brain_module.UniversalBrain()
        err = urllib.error.HTTPError("https://classifier.dev", 403, "Forbidden", None, None)
        with patch.object(brain_module.urllib.request, "urlopen", side_effect=err):
            result = brain._query_classifier_dev(RUNNER, make_scene())
        self.assertIsNone(result)
        self.assertTrue(brain._classifier_dead)

    def test_dead_classifier_short_circuits_without_network(self):
        with patch.object(brain_module.threading.Thread, "start"), \
                patch.object(brain_module.UniversalBrain, "_start_laya_loader"):
            brain = brain_module.UniversalBrain()
        brain._classifier_dead = True
        with patch.object(brain_module.urllib.request, "urlopen") as urlopen:
            result = brain._query_classifier_dev(RUNNER, make_scene())
        urlopen.assert_not_called()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
