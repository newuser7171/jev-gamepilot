"""Local LLM2Jev backend: env gate, single-flight, Choice/Score/Noul mapping, fusion order."""
import os
import time
import unittest
from unittest.mock import Mock, patch

import universal_brain as brain_module
from profile_manager import DEFAULT_PROFILES
from universal_vision import UniversalSceneState

RUNNER = next(p for p in DEFAULT_PROFILES if p.id == "runner_3lane")


def make_brain():
    with patch.object(brain_module.threading.Thread, "start"), \
            patch.object(brain_module.UniversalBrain, "_start_laya_loader"), \
            patch.object(brain_module.UniversalBrain, "_start_llm2jev_loader"):
        return brain_module.UniversalBrain()


class _Ans:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class FakeL2JEngine:
    def __init__(self, answers=None, delay=0.0, raise_exc=None):
        self.answers = answers or {
            "tactical_action": _Ans(choice="jump", confidence=0.91),
            "threat_severity": _Ans(score=3.0),
            "is_urgent_reflex": _Ans(noul=0.85),
        }
        self.delay = delay
        self.raise_exc = raise_exc
        self.calls = 0

    def evaluate(self, request):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.raise_exc:
            raise self.raise_exc
        return Mock(answers=self.answers)


class LLM2JEVStartupTests(unittest.TestCase):
    def test_missing_model_env_skips_loader(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(brain_module, "_LLM2JEV_AVAILABLE", True), \
                patch.object(brain_module.threading, "Thread") as thread_cls:
            brain_module.UniversalBrain._start_llm2jev_loader(Mock(llm2jev_model="", _llm2jev_tried=False))
            thread_cls.assert_not_called()

    def test_model_env_starts_loader_thread(self):
        with patch.dict(os.environ, {"LLM2JEV_MODEL": "x"}, clear=False), \
                patch.object(brain_module, "_LLM2JEV_AVAILABLE", True), \
                patch.object(brain_module.threading, "Thread") as thread_cls:
            brain = Mock(llm2jev_model="x", _llm2jev_tried=False)
            brain_module.UniversalBrain._start_llm2jev_loader(brain)
            thread_cls.assert_called_once()

    def test_unavailable_package_skips_loader(self):
        with patch.object(brain_module, "_LLM2JEV_AVAILABLE", False), \
                patch.dict(os.environ, {"LLM2JEV_MODEL": "x"}, clear=False), \
                patch.object(brain_module.threading, "Thread") as thread_cls:
            brain = Mock(llm2jev_model="x", _llm2jev_tried=False)
            brain_module.UniversalBrain._start_llm2jev_loader(brain)
            thread_cls.assert_not_called()


class QueryLocalLLM2JEVTests(unittest.TestCase):
    def setUp(self):
        self.brain = make_brain()
        self.scene = UniversalSceneState()

    def test_engine_none_returns_none(self):
        self.brain.llm2jev_engine = None
        self.assertIsNone(self.brain._query_local_llm2jev(RUNNER, self.scene))

    def test_maps_answers_to_decision_dict(self):
        self.brain.llm2jev_engine = FakeL2JEngine()
        self.brain.llm2jev_model = "fake-model"
        res = self.brain._query_local_llm2jev(RUNNER, self.scene)
        self.assertIsNotNone(res)
        self.assertEqual(res["source"], "local_llm2jev")
        self.assertEqual(res["action"], "jump")
        self.assertEqual(res["threat_score"], 0.75)
        self.assertEqual(res["confidence"], 0.91)
        self.assertGreater(res["latency_ms"], 0)

    def test_coerces_action_onto_profile_actions(self):
        self.brain.llm2jev_engine = FakeL2JEngine(
            answers={
                "tactical_action": _Ans(choice="totally_not_a_real_action", confidence=0.4),
                "threat_severity": _Ans(score=1.0),
                "is_urgent_reflex": _Ans(noul=0.2),
            }
        )
        self.brain.llm2jev_model = "fake-model"
        res = self.brain._query_local_llm2jev(RUNNER, self.scene)
        self.assertIsNotNone(res)
        valid = {a.name for a in RUNNER.actions} | {"wait", "maintain_course", "stand_idle"}
        self.assertIn(res["action"], valid)

    def test_exception_returns_none(self):
        self.brain.llm2jev_engine = FakeL2JEngine(raise_exc=RuntimeError("boom"))
        self.brain.llm2jev_model = "fake-model"
        self.assertIsNone(self.brain._query_local_llm2jev(RUNNER, self.scene))

    def test_single_flight_skips_concurrent_call(self):
        slow = FakeL2JEngine(delay=0.15)
        self.brain.llm2jev_engine = slow
        self.brain.llm2jev_model = "fake-model"
        holder = {}

        def run():
            holder["res"] = self.brain._query_local_llm2jev(RUNNER, self.scene)

        t = __import__("threading").Thread(target=run)
        t.start()
        time.sleep(0.03)
        concurrent = self.brain._query_local_llm2jev(RUNNER, self.scene)
        t.join(timeout=2.0)
        self.assertIsNone(concurrent)
        self.assertIsNotNone(holder.get("res"))
        self.assertEqual(slow.calls, 1)


class EvaluateSceneFusionOrderTests(unittest.TestCase):
    def setUp(self):
        self.brain = make_brain()
        self.brain.laya_agent = None
        self.scene = UniversalSceneState()

    def test_local_llm2jev_preferred_over_cloud(self):
        local = {"action": "jump", "threat_score": 0.5, "confidence": 0.9,
                 "latency_ms": 10.0, "source": "local_llm2jev", "target_coords": None}
        cloud = {"action": "slide", "threat_score": 0.4, "confidence": 0.8,
                 "latency_ms": 800.0, "source": "jev_system_one_cloud", "target_coords": None}
        self.brain.llm2jev_engine = FakeL2JEngine()
        self.brain.client = Mock()
        with patch.object(self.brain, "_query_local_llm2jev", return_value=local) as p_local, \
                patch.object(self.brain, "_query_typesafe_sdk", return_value=cloud) as p_cloud, \
                patch.object(self.brain, "_query_classifier_dev", return_value=None):
            self.brain._evaluate_scene(RUNNER, self.scene)
        p_local.assert_called_once()
        p_cloud.assert_not_called()
        self.assertEqual(self.brain.last_decision["source"], "local_llm2jev")
        self.assertEqual(self.brain.last_decision["action"], "jump")

    def test_cloud_fallback_when_local_returns_none(self):
        cloud = {"action": "slide", "threat_score": 0.4, "confidence": 0.8,
                 "latency_ms": 800.0, "source": "jev_system_one_cloud", "target_coords": None}
        self.brain.llm2jev_engine = FakeL2JEngine()
        self.brain.client = Mock()
        with patch.object(self.brain, "_query_local_llm2jev", return_value=None), \
                patch.object(self.brain, "_query_typesafe_sdk", return_value=cloud) as p_cloud, \
                patch.object(self.brain, "_query_classifier_dev", return_value=None):
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "x"}):
                self.brain._evaluate_scene(RUNNER, self.scene)
        p_cloud.assert_called_once()
        self.assertEqual(self.brain.last_decision["source"], "jev_system_one_cloud")

    def test_no_engine_keeps_cloud_path(self):
        cloud = {"action": "jump", "threat_score": 0.6, "confidence": 0.85,
                 "latency_ms": 700.0, "source": "jev_system_one_cloud", "target_coords": None}
        self.brain.llm2jev_engine = None
        self.brain.client = Mock()
        with patch.object(self.brain, "_query_local_llm2jev") as p_local, \
                patch.object(self.brain, "_query_typesafe_sdk", return_value=cloud), \
                patch.object(self.brain, "_query_classifier_dev", return_value=None):
            with patch.dict(os.environ, {"TYPESAFE_API_KEY": "x"}):
                self.brain._evaluate_scene(RUNNER, self.scene)
        p_local.assert_not_called()
        self.assertEqual(self.brain.last_decision["source"], "jev_system_one_cloud")


if __name__ == "__main__":
    unittest.main()
