"""Startup must not require credentials for the optional cloud backend."""
import os
import unittest
from unittest.mock import Mock, patch

import universal_brain as brain_module


class BrainStartupTests(unittest.TestCase):
    def make_brain(self):
        with patch.object(brain_module.threading.Thread, "start"), \
                patch.object(brain_module.UniversalBrain, "_start_laya_loader"):
            return brain_module.UniversalBrain()

    def test_missing_key_skips_sdk_and_keeps_classifier_fallback(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(brain_module, "_TYPESAFE_AVAILABLE", True), \
                patch.object(brain_module, "TypeSafeClient") as client:
            brain = self.make_brain()
            self.assertIsNone(brain.client)
            client.assert_not_called()
            result = {"action": "wait", "confidence": 0.9}
            with patch.object(brain, "_query_classifier_dev", return_value=result) as fallback:
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
