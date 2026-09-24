"""Simple Jev cloud classifier tier + local GLiNER2.5-Decide mid-tier wiring."""
import io
import json
import unittest
import urllib.error
from unittest.mock import Mock, patch

import universal_brain as brain_module
from profile_manager import DEFAULT_PROFILES
from universal_vision import UniversalSceneState

RUNNER = next(p for p in DEFAULT_PROFILES if p.id == "runner_3lane")


def make_brain():
    with patch.object(brain_module.threading.Thread, "start"), \
            patch.object(brain_module.UniversalBrain, "_start_laya_loader"), \
            patch.object(brain_module.UniversalBrain, "_start_llm2jev_loader"), \
            patch.object(brain_module.UniversalBrain, "_start_openjev_loader"):
        return brain_module.UniversalBrain()


def make_scene(urgency=0.5):
    return UniversalSceneState(threat_urgency=urgency, game_phase="active")


def _decision(action, conf, source="x"):
    return {
        "action": action,
        "confidence": conf,
        "threat_score": 0.3,
        "latency_ms": 10.0,
        "source": source,
        "target_coords": None,
    }


class SimpleJevQueryTests(unittest.TestCase):
    def test_parses_choice_score_noul(self):
        brain = make_brain()
        brain._simple_jev_last_at = 0.0
        body = {
            "answers": {
                "tactical_action": {"type": "choice", "choice": "dodge_left", "confidence": 0.97},
                "threat_severity": {"type": "score", "score": 3.2},
                "is_urgent_reflex": {"type": "noul", "noul": 0.4},
            }
        }
        resp = Mock()
        resp.__enter__ = Mock(return_value=io.BytesIO(json.dumps(body).encode()))
        resp.__exit__ = Mock(return_value=False)
        with patch.object(brain_module.urllib.request, "urlopen", return_value=resp):
            result = brain._query_simple_jev(RUNNER, make_scene())
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "dodge_left")
        self.assertAlmostEqual(result["confidence"], 0.97, places=3)
        # score 3.2/4 = 0.8; noul 0.4 lower → stays 0.8
        self.assertAlmostEqual(result["threat_score"], 0.8, places=2)
        self.assertEqual(result["source"], "jev_simple_jev")

    def test_noul_raises_threat(self):
        brain = make_brain()
        brain._simple_jev_last_at = 0.0
        body = {
            "answers": {
                "tactical_action": {"choice": "wait", "confidence": 0.6},
                "threat_severity": {"score": 1.0},
                "is_urgent_reflex": {"noul": 0.95},
            }
        }
        resp = Mock()
        resp.__enter__ = Mock(return_value=io.BytesIO(json.dumps(body).encode()))
        resp.__exit__ = Mock(return_value=False)
        with patch.object(brain_module.urllib.request, "urlopen", return_value=resp):
            result = brain._query_simple_jev(RUNNER, make_scene())
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["threat_score"], 0.95, places=2)

    def test_circuit_breaks_on_http_429(self):
        brain = make_brain()
        brain._simple_jev_last_at = 0.0
        err = urllib.error.HTTPError("https://x", 429, "Too Many Requests", None, None)
        with patch.object(brain_module.urllib.request, "urlopen", side_effect=err):
            result = brain._query_simple_jev(RUNNER, make_scene())
        self.assertIsNone(result)
        self.assertTrue(brain._simple_jev_dead)
        # dead circuit: no further HTTP attempt
        with patch.object(brain_module.urllib.request, "urlopen") as mock_open:
            self.assertIsNone(brain._query_simple_jev(RUNNER, make_scene()))
            mock_open.assert_not_called()

    def test_rate_limit_gap(self):
        brain = make_brain()
        import time as _time
        brain._simple_jev_last_at = _time.monotonic()  # just called
        with patch.object(brain_module.urllib.request, "urlopen") as mock_open:
            self.assertIsNone(brain._query_simple_jev(RUNNER, make_scene()))
            mock_open.assert_not_called()

    def test_env_url_and_model(self):
        import os
        env = {
            "SIMPLE_JEV_URL": "https://api.example.com/",
            "SIMPLE_JEV_MODEL": "some/classifier",
        }
        with patch.dict(os.environ, env, clear=True):
            brain = make_brain()
        self.assertEqual(brain.simple_jev_url, "https://api.example.com")
        self.assertEqual(brain.simple_jev_model, "some/classifier")


class GlinerDecideTests(unittest.TestCase):
    def test_parses_schema_result(self):
        brain = make_brain()
        brain.gliner_engine = Mock()
        brain.gliner_engine.classify.return_value = Mock(
            value=Mock(side_effect=lambda name: {
                "tactical_action": "jump",
                "threat_severity": "3",
                "is_urgent_reflex": "no",
            }[name]),
            confidence=Mock(return_value=0.92),
        )
        result = brain._query_gliner_decide(RUNNER, make_scene())
        self.assertIsNotNone(result)
        self.assertEqual(result["action"], "jump")
        self.assertAlmostEqual(result["confidence"], 0.92, places=3)
        self.assertAlmostEqual(result["threat_score"], 0.75, places=2)
        self.assertEqual(result["source"], "local_gliner_decide")

    def test_reflex_yes_raises_threat(self):
        brain = make_brain()
        brain.gliner_engine = Mock()
        brain.gliner_engine.classify.return_value = Mock(
            value=Mock(side_effect=lambda name: {
                "tactical_action": "wait",
                "threat_severity": "0",
                "is_urgent_reflex": "yes",
            }[name]),
            confidence=Mock(return_value=0.7),
        )
        result = brain._query_gliner_decide(RUNNER, make_scene())
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["threat_score"], 0.9)

    def test_none_engine_returns_none(self):
        brain = make_brain()
        brain.gliner_engine = None
        self.assertIsNone(brain._query_gliner_decide(RUNNER, make_scene()))

    def test_lazy_loader_sets_tried(self):
        brain = make_brain()
        brain._gliner_tried = False
        with patch.object(brain_module, "_GLINER_AVAILABLE", True), \
                patch.object(brain_module.threading.Thread, "start") as tstart:
            brain._start_gliner_loader()
        self.assertTrue(brain._gliner_tried)
        tstart.assert_called_once()
        # second call is a no-op
        with patch.object(brain_module.threading.Thread, "start") as tstart2:
            brain._start_gliner_loader()
        tstart2.assert_not_called()


class LocalEscalationOrderTests(unittest.TestCase):
    def test_openjev_sure_skips_gliner(self):
        brain = make_brain()
        brain.openjev_engine = object()
        brain.gliner_engine = None
        with patch.object(brain, "_query_local_openjev", return_value=_decision("dodge_left", 0.95, "local_openjev")), \
                patch.object(brain, "_start_gliner_loader") as lazy:
            result = brain._local_escalation(RUNNER, make_scene())
        self.assertEqual(result["source"], "local_openjev")
        lazy.assert_not_called()

    def test_weak_openjev_triggers_gliner_mid(self):
        brain = make_brain()
        brain.openjev_engine = object()
        brain.llm2jev_engine = None
        brain.gliner_engine = object()  # already loaded
        brain._gliner_tried = True
        with patch.object(brain, "_query_local_openjev", return_value=_decision("maintain_course", 0.40, "local_openjev")), \
                patch.object(brain, "_query_gliner_decide", return_value=_decision("jump", 0.90, "local_gliner_decide")), \
                patch.object(brain, "_start_gliner_loader") as lazy:
            result = brain._local_escalation(RUNNER, make_scene())
        self.assertEqual(result["source"], "local_gliner_decide")
        lazy.assert_not_called()

    def test_weak_openjev_starts_lazy_gliner_load(self):
        brain = make_brain()
        brain.openjev_engine = object()
        brain.gliner_engine = None
        brain._gliner_tried = False
        brain.llm2jev_engine = None
        weak = _decision("maintain_course", 0.40, "local_openjev")
        with patch.object(brain, "_query_local_openjev", return_value=weak), \
                patch.object(brain, "_start_gliner_loader") as lazy:
            result = brain._local_escalation(RUNNER, make_scene())
        lazy.assert_called_once()
        # engine still loading this tick → weak openjev is the best available
        self.assertEqual(result["source"], "local_openjev")

    def test_gliner_beats_llm2jev_when_confident(self):
        brain = make_brain()
        brain.openjev_engine = object()
        brain.gliner_engine = object()
        brain.llm2jev_engine = object()
        with patch.object(brain, "_query_local_openjev", return_value=_decision("maintain_course", 0.50, "local_openjev")), \
                patch.object(brain, "_query_gliner_decide", return_value=_decision("slide", 0.88, "local_gliner_decide")), \
                patch.object(brain, "_query_local_llm2jev") as deep:
            result = brain._local_escalation(RUNNER, make_scene())
        self.assertEqual(result["source"], "local_gliner_decide")
        deep.assert_not_called()


class FusionChainTests(unittest.TestCase):
    def test_simple_jev_before_classifier_dev(self):
        brain = make_brain()
        scene = make_scene()
        calls = []

        def track(name):
            def fn(*a, **k):
                calls.append(name)
                return None
            return fn

        with patch.object(brain, "_local_escalation", side_effect=track("local")), \
                patch.object(brain, "_query_bev", side_effect=track("bev")), \
                patch.object(brain, "_query_typesafe_sdk", side_effect=track("typesafe")), \
                patch.object(brain, "_query_simple_jev", side_effect=track("simple")), \
                patch.object(brain, "_query_classifier_dev", side_effect=track("cdev")):
            brain._evaluate_scene(RUNNER, scene)
        self.assertEqual(calls, ["local", "bev", "typesafe", "simple", "cdev"])

    def test_simple_jev_result_skips_classifier_dev(self):
        brain = make_brain()
        scene = make_scene()
        with patch.object(brain, "_local_escalation", return_value=None), \
                patch.object(brain, "_query_bev", return_value=None), \
                patch.object(brain, "_query_typesafe_sdk", return_value=None), \
                patch.object(brain, "_query_simple_jev", return_value=_decision("dodge_left", 0.9, "jev_simple_jev")), \
                patch.object(brain, "_query_classifier_dev") as cdev:
            brain._evaluate_scene(RUNNER, scene)
        cdev.assert_not_called()
        self.assertEqual(brain.last_decision["source"], "jev_simple_jev")

    def test_sync_path_includes_simple_jev(self):
        brain = make_brain()
        scene = make_scene()
        calls = []

        def track(name):
            def fn(*a, **k):
                calls.append(name)
                return None
            return fn

        with patch.object(brain, "_local_escalation", side_effect=track("local")), \
                patch.object(brain, "_query_bev", side_effect=track("bev")), \
                patch.object(brain, "_query_typesafe_sdk", side_effect=track("typesafe")), \
                patch.object(brain, "_query_simple_jev", side_effect=track("simple")), \
                patch.object(brain, "_query_classifier_dev", side_effect=track("cdev")):
            brain.query_jev_universal(RUNNER, scene)
        self.assertEqual(calls, ["local", "bev", "typesafe", "simple", "cdev"])


if __name__ == "__main__":
    unittest.main()
