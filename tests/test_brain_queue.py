"""Brain queue backpressure, decision cache and profile-switch staleness."""
import queue
import unittest
from unittest.mock import Mock, patch

import universal_brain as brain_module
from profile_manager import DEFAULT_PROFILES
from universal_vision import UniversalSceneState

RUNNER = next(p for p in DEFAULT_PROFILES if p.id == "runner_3lane")
BALATRO = next(p for p in DEFAULT_PROFILES if p.id == "pc_balatro")


def make_brain():
    """Offline brain with the worker thread suppressed (tests drive the queue)."""
    with patch.object(brain_module.threading.Thread, "start"), \
            patch.object(brain_module.UniversalBrain, "_start_laya_loader"), \
            patch.object(brain_module.UniversalBrain, "_start_llm2jev_loader"), \
            patch.object(brain_module.UniversalBrain, "_start_openjev_loader"):
        return brain_module.UniversalBrain()


def scene(urgency):
    return UniversalSceneState(threat_urgency=urgency, game_phase="active")


class EnqueueBackpressureTests(unittest.TestCase):
    def setUp(self):
        self.brain = make_brain()
        self.brain.fastino_key = ""
        self.addCleanup(self.brain.close)

    def drain(self):
        items = []
        while True:
            try:
                items.append(self.brain._work_queue.get_nowait())
            except queue.Empty:
                return items

    def test_full_queue_drops_stalest_not_newest(self):
        first, second, newest = scene(0.1), scene(0.2), scene(0.9)
        self.assertFalse(self.brain._work_queue.full())
        self.brain._enqueue_scene(RUNNER, first)
        self.brain._enqueue_scene(RUNNER, second)
        self.assertTrue(self.brain._work_queue.full())

        self.brain._enqueue_scene(RUNNER, newest)

        self.assertEqual([s for _, s in self.drain()], [second, newest])

    def test_closed_brain_accepts_nothing(self):
        self.brain.close()
        self.brain._enqueue_scene(RUNNER, scene(0.9))
        # Only the shutdown sentinel remains.
        self.assertEqual(self.drain(), [None])


class DecisionCacheTests(unittest.TestCase):
    def setUp(self):
        self.brain = make_brain()
        self.addCleanup(self.brain.close)

    def test_reset_clears_action_and_records_source(self):
        self.brain.last_decision = {
            "action": "jump",
            "threat_score": 0.8,
            "confidence": 0.9,
            "latency_ms": 12.0,
            "source": "local_laya",
            "target_coords": (10, 20),
        }
        self.brain.reset_decision("profile_switch")
        snap = self.brain.snapshot_decision()
        self.assertEqual(snap["action"], "wait")
        self.assertEqual(snap["source"], "profile_switch")
        self.assertIsNone(snap["target_coords"])

    def test_snapshot_is_a_copy(self):
        snap = self.brain.snapshot_decision()
        snap["action"] = "mutated"
        self.assertNotEqual(self.brain.snapshot_decision()["action"], "mutated")


class ProfileSwitchTests(unittest.TestCase):
    def test_set_profile_resets_cached_decision(self):
        from pilot_core import PilotCore

        core = PilotCore.__new__(PilotCore)
        core.profile_mgr = Mock()
        core.profile_mgr.get_profile.return_value = BALATRO
        core.universal_brain = Mock()

        self.assertTrue(core.set_profile("pc_balatro"))
        core.universal_brain.reset_decision.assert_called_once_with("profile_switch")

    def test_unknown_profile_keeps_current_state(self):
        from pilot_core import PilotCore

        core = PilotCore.__new__(PilotCore)
        core.profile_mgr = Mock()
        core.profile_mgr.get_profile.return_value = None
        core.universal_brain = Mock()

        self.assertFalse(core.set_profile("does_not_exist"))
        core.universal_brain.reset_decision.assert_not_called()


class LatestDecisionTests(unittest.TestCase):
    def test_loop_decision_wins_over_brain_cache(self):
        from pilot_core import PilotCore

        core = PilotCore.__new__(PilotCore)
        core.last_decision = {"action": "jump", "source": "dino"}
        core.universal_brain = Mock()
        core.universal_brain.snapshot_decision.return_value = {"action": "wait"}

        self.assertEqual(core.latest_decision()["action"], "jump")
        core.universal_brain.snapshot_decision.assert_not_called()

    def test_empty_loop_falls_back_to_brain(self):
        from pilot_core import PilotCore

        core = PilotCore.__new__(PilotCore)
        core.last_decision = {}
        core.universal_brain = Mock()
        core.universal_brain.snapshot_decision.return_value = {"action": "slide"}

        self.assertEqual(core.latest_decision()["action"], "slide")


if __name__ == "__main__":
    unittest.main()
