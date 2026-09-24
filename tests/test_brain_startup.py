"""Startup must not require credentials for the optional cloud backend."""
import json
import os
import unittest
from unittest.mock import Mock, patch

import universal_brain as brain_module


class BrainStartupTests(unittest.TestCase):
    def make_brain(self):
        with patch.object(brain_module.threading.Thread, "start"), \
                patch.object(brain_module.UniversalBrain, "_start_laya_loader"), \
                patch.object(brain_module.UniversalBrain, "_start_openjev_loader"):
            return brain_module.UniversalBrain()

    def test_missing_key_skips_sdk_and_keeps_classifier_fallback(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(brain_module, "_TYPESAFE_AVAILABLE", True), \
                patch.object(brain_module, "TypeSafeClient") as client:
            brain = self.make_brain()
            self.assertIsNone(brain.client)
            client.assert_not_called()
            result = {"action": "wait", "confidence": 0.9}
            with patch.object(brain, "_query_bev", return_value=None), \
                    patch.object(brain, "_query_classifier_dev", return_value=result) as fallback:
                brain._evaluate_scene(Mock(), Mock())
            fallback.assert_called_once()
            self.assertEqual(brain.last_decision, result)

    def test_blank_key_skips_sdk(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "   "}, clear=True), \
                patch.object(brain_module, "_TYPESAFE_AVAILABLE", True), \
                patch.object(brain_module, "TypeSafeClient") as client:
            self.assertIsNone(self.make_brain().client)
            client.assert_not_called()

    def test_configured_key_initializes_sdk(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-placeholder"}, clear=True), \
                patch.object(brain_module, "_TYPESAFE_AVAILABLE", True), \
                patch.object(brain_module, "TypeSafeClient") as client:
            self.assertIs(self.make_brain().client, client.return_value)
            client.assert_called_once_with()

    def test_missing_sdk_still_starts(self):
        with patch.object(brain_module, "_TYPESAFE_AVAILABLE", False):
            self.assertIsNone(self.make_brain().client)

    def test_bev_defaults_loopback(self):
        with patch.dict(os.environ, {}, clear=True):
            brain = self.make_brain()
        self.assertEqual(brain.bev_url, "http://127.0.0.1:18781")
        self.assertEqual(brain.bev_model, "bev-bonsai-27b")
        self.assertFalse(brain._bev_up)

    def test_bev_unhealthy_skips_query(self):
        brain = self.make_brain()
        with patch.object(brain, "_bev_healthy", return_value=False):
            self.assertIsNone(brain._query_bev(Mock(), Mock()))

    def test_bev_fusion_inserted_before_typesafe(self):
        brain = self.make_brain()
        calls = []
        with patch.object(brain, "_query_local_llm2jev", return_value=None), \
                patch.object(brain, "_query_local_openjev", return_value=None), \
                patch.object(brain, "_query_bev",
                             side_effect=lambda *a: calls.append("bev") or None), \
                patch.object(brain, "_query_typesafe_sdk",
                             side_effect=lambda *a: calls.append("typesafe") or None), \
                patch.object(brain, "_query_classifier_dev",
                             side_effect=lambda *a: calls.append("classifier") or
                             {"action": "wait", "confidence": 0.9}):
            brain._evaluate_scene(Mock(), Mock())
        self.assertEqual(calls, ["bev", "typesafe", "classifier"])

    def test_bev_parses_systemone_response(self):
        brain = self.make_brain()
        profile = Mock()
        profile.name = "unit"
        profile.category = "action"
        profile.actions = [Mock(name="jump"), Mock(name="wait")]
        profile.actions[0].name = "jump"
        profile.actions[0].description = "Jump now"
        profile.actions[1].name = "wait"
        profile.actions[1].description = "Hold"
        scene = Mock()
        scene.best_target = None

        payload = {
            "answers": {
                "tactical_action": {
                    "type": "choice",
                    "choice": "jump",
                    "probabilities": {"jump": 0.82, "wait": 0.18},
                    "confidence": 0.82,
                },
                "danger": {"type": "score", "score": 3.0, "probabilities": {}},
                "urgent": {"type": "noul", "noul": 0.91},
            }
        }

        class FakeResp:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch.object(brain, "_bev_healthy", return_value=True), \
                patch.object(brain, "_build_verbal_state", return_value="state"), \
                patch.object(brain, "coerce_action_name",
                             side_effect=lambda p, a, s: a), \
                patch.object(brain_module.urllib.request, "urlopen",
                             return_value=FakeResp()):
            result = brain._query_bev(profile, scene)

        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "jump")
        self.assertEqual(result["source"], "local_bev")
        self.assertAlmostEqual(result["confidence"], 0.82, places=2)
        self.assertAlmostEqual(result["threat_score"], 0.91, places=2)

    def test_bev_health_failure_marks_down(self):
        brain = self.make_brain()
        brain._bev_checked_at = 0.0
        with patch.object(brain_module.urllib.request, "urlopen",
                          side_effect=OSError("refused")):
            self.assertFalse(brain._bev_healthy())
        self.assertFalse(brain._bev_up)
