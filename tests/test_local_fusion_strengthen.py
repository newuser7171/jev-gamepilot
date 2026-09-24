"""Local fusion strengthen: env tunables, openjev-first escalation, threat gates, noul blend."""
import json
import os
import unittest
from unittest.mock import Mock, patch

import universal_brain as brain_module
from profile_manager import DEFAULT_PROFILES
from universal_vision import UniversalSceneState, UniversalEntity

RUNNER = next(p for p in DEFAULT_PROFILES if p.id == "runner_3lane")


def make_brain():
    with patch.object(brain_module.threading.Thread, "start"), \
            patch.object(brain_module.UniversalBrain, "_start_laya_loader"), \
            patch.object(brain_module.UniversalBrain, "_start_llm2jev_loader"), \
            patch.object(brain_module.UniversalBrain, "_start_openjev_loader"):
        return brain_module.UniversalBrain()


def _decision(action, conf, threat=0.2, source="local"):
    return {
        "action": action,
        "confidence": conf,
        "threat_score": threat,
        "latency_ms": 12.0,
        "source": source,
        "target_coords": None,
    }


class EnvTunableTests(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            brain = make_brain()
        self.assertAlmostEqual(brain.laya_fast_path_conf, 0.88, places=5)
        self.assertAlmostEqual(brain.local_accept_conf, 0.72, places=5)
        self.assertAlmostEqual(brain.fast_path_threat_gate, 0.60, places=5)

    def test_env_overrides(self):
        env = {
            "LAYA_FAST_PATH_CONF": "0.91",
            "LOCAL_ACCEPT_CONF": "0.55",
            "FAST_PATH_THREAT_GATE": "0.40",
        }
        with patch.dict(os.environ, env, clear=True):
            brain = make_brain()
        self.assertAlmostEqual(brain.laya_fast_path_conf, 0.91, places=5)
        self.assertAlmostEqual(brain.local_accept_conf, 0.55, places=5)
        self.assertAlmostEqual(brain.fast_path_threat_gate, 0.40, places=5)

    def test_invalid_env_falls_back(self):
        with patch.dict(os.environ, {"LAYA_FAST_PATH_CONF": "abc"}, clear=True):
            brain = make_brain()
        self.assertAlmostEqual(brain.laya_fast_path_conf, 0.88, places=5)

    def test_env_float_helper(self):
        self.assertEqual(brain_module._env_float("NO_SUCH_ENV_XYZ", 1.5), 1.5)
        with patch.dict(os.environ, {"NO_SUCH_ENV_XYZ": "2.25"}):
            self.assertEqual(brain_module._env_float("NO_SUCH_ENV_XYZ", 1.5), 2.25)


class FastPathThreatGateTests(unittest.TestCase):
    def setUp(self):
        self.brain = make_brain()

    def test_high_conf_non_passive_commits(self):
        scene = UniversalSceneState(threat_urgency=0.9)
        self.assertTrue(self.brain._laya_fast_path_ok(_decision("jump", 0.95, 0.9), scene))

    def test_high_conf_passive_under_threat_blocks(self):
        scene = UniversalSceneState(threat_urgency=0.75)
        self.assertFalse(self.brain._laya_fast_path_ok(_decision("wait", 0.95, 0.8), scene))

    def test_high_conf_passive_clear_scene_commits(self):
        scene = UniversalSceneState(threat_urgency=0.1)
        self.assertTrue(self.brain._laya_fast_path_ok(_decision("wait", 0.95, 0.1), scene))

    def test_below_threshold_blocks(self):
        scene = UniversalSceneState(threat_urgency=0.0)
        self.assertFalse(self.brain._laya_fast_path_ok(_decision("jump", 0.5), scene))

    def test_laya_threat_blocks_passive_even_if_scene_calm(self):
        scene = UniversalSceneState(threat_urgency=0.1)
        self.assertFalse(self.brain._laya_fast_path_ok(_decision("wait", 0.95, 0.85), scene))


class LocalEscalationOrderTests(unittest.TestCase):
    def setUp(self):
        self.brain = make_brain()
        self.scene = UniversalSceneState()

    def test_confident_openjev_skips_llm2jev(self):
        self.brain.openjev_engine = object()
        self.brain.llm2jev_engine = object()
        with patch.object(self.brain, "_query_local_openjev",
                          return_value=_decision("jump", 0.9, source="local_openjev")) as p_o, \
                patch.object(self.brain, "_query_local_llm2jev") as p_l:
            res = self.brain._local_escalation(RUNNER, self.scene)
        p_o.assert_called_once()
        p_l.assert_not_called()
        self.assertEqual(res["source"], "local_openjev")

    def test_weak_openjev_escalates_to_llm2jev(self):
        self.brain.openjev_engine = object()
        self.brain.llm2jev_engine = object()
        with patch.object(self.brain, "_query_local_openjev",
                          return_value=_decision("wait", 0.4, source="local_openjev")), \
                patch.object(self.brain, "_query_local_llm2jev",
                             return_value=_decision("jump", 0.85, source="local_llm2jev")) as p_l:
            res = self.brain._local_escalation(RUNNER, self.scene)
        p_l.assert_called_once()
        self.assertEqual(res["source"], "local_llm2jev")

    def test_llm2jev_only_when_openjev_absent(self):
        self.brain.openjev_engine = None
        self.brain.llm2jev_engine = object()
        with patch.object(self.brain, "_query_local_llm2jev",
                          return_value=_decision("jump", 0.8, source="local_llm2jev")) as p_l:
            res = self.brain._local_escalation(RUNNER, self.scene)
        p_l.assert_called_once()
        self.assertEqual(res["source"], "local_llm2jev")

    def test_no_local_returns_none(self):
        self.brain.openjev_engine = None
        self.brain.llm2jev_engine = None
        self.assertIsNone(self.brain._local_escalation(RUNNER, self.scene))

    def test_weak_llm2jev_keeps_weak_openjev(self):
        self.brain.openjev_engine = object()
        self.brain.llm2jev_engine = object()
        with patch.object(self.brain, "_query_local_openjev",
                          return_value=_decision("jump", 0.5, source="local_openjev")), \
                patch.object(self.brain, "_query_local_llm2jev",
                             return_value=_decision("wait", 0.3, source="local_llm2jev")):
            res = self.brain._local_escalation(RUNNER, self.scene)
        self.assertEqual(res["source"], "local_openjev")


class ThreatDisagreementTests(unittest.TestCase):
    def test_high_threat_prefers_non_passive_jev(self):
        brain = make_brain()
        scene = UniversalSceneState(threat_urgency=0.8)
        fused = brain._resolve_disagreement(
            _decision("wait", 0.9, 0.2, "local_laya_brain"),
            _decision("jump", 0.8, 0.9, "local_openjev"),
            scene,
        )
        self.assertEqual(fused["action"], "jump")
        self.assertEqual(fused["source"], "threat_preferred_jev")

    def test_high_threat_prefers_non_passive_laya(self):
        brain = make_brain()
        scene = UniversalSceneState(threat_urgency=0.8)
        fused = brain._resolve_disagreement(
            _decision("jump", 0.7, 0.9, "local_laya_brain"),
            _decision("wait", 0.85, 0.2, "local_openjev"),
            scene,
        )
        self.assertEqual(fused["action"], "jump")
        self.assertEqual(fused["source"], "threat_preferred_laya")

    def test_calm_scene_prefers_higher_conf(self):
        brain = make_brain()
        scene = UniversalSceneState(threat_urgency=0.1)
        fused = brain._resolve_disagreement(
            _decision("wait", 0.6, 0.1, "local_laya_brain"),
            _decision("jump", 0.9, 0.3, "local_openjev"),
            scene,
        )
        self.assertEqual(fused["action"], "jump")
        self.assertEqual(fused["source"], "jev_escalation")


class EvaluateSceneLocalPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.brain = make_brain()
        self.brain.laya_agent = None
        self.brain.client = None
        self.scene = UniversalSceneState()

    def test_local_openjev_preferred_over_cloud(self):
        self.brain.openjev_engine = object()
        with patch.object(self.brain, "_query_local_openjev",
                          return_value=_decision("jump", 0.9, source="local_openjev")) as p_o, \
                patch.object(self.brain, "_query_typesafe_sdk") as p_cloud, \
                patch.object(self.brain, "_query_classifier_dev", return_value=None), \
                patch.object(self.brain, "_query_bev", return_value=None):
            self.brain._evaluate_scene(RUNNER, self.scene)
        p_o.assert_called_once()
        p_cloud.assert_not_called()
        self.assertEqual(self.brain.last_decision["source"], "local_openjev")

    def test_llm2jev_called_when_openjev_missing(self):
        self.brain.openjev_engine = None
        self.brain.llm2jev_engine = object()
        with patch.object(self.brain, "_query_local_llm2jev",
                          return_value=_decision("jump", 0.9, source="local_llm2jev")) as p_l, \
                patch.object(self.brain, "_query_typesafe_sdk"), \
                patch.object(self.brain, "_query_classifier_dev", return_value=None), \
                patch.object(self.brain, "_query_bev", return_value=None):
            self.brain._evaluate_scene(RUNNER, self.scene)
        p_l.assert_called_once()
        self.assertEqual(self.brain.last_decision["source"], "local_llm2jev")


class LLM2JEVNoulBlendTests(unittest.TestCase):
    def test_noul_raises_threat_when_score_low(self):
        brain = make_brain()
        brain.llm2jev_engine = Mock()
        brain.llm2jev_model = "fake"

        class Ans:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        brain.llm2jev_engine.evaluate = lambda req: Mock(answers={
            "tactical_action": Ans(choice="jump", confidence=0.7),
            "threat_severity": Ans(score=1.0),  # 0.25
            "is_urgent_reflex": Ans(noul=0.9),  # should raise to 0.9
        })
        with patch.object(brain, "_build_verbal_state", return_value="s"):
            res = brain._query_local_llm2jev(RUNNER, UniversalSceneState())
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res["threat_score"], 0.9, places=2)


class VerbalStateStrengthenTests(unittest.TestCase):
    def setUp(self):
        self.brain = make_brain()

    def test_includes_phase_urgency_threat_count(self):
        threat = UniversalEntity(x=100, y=200, w=30, h=40, entity_type="threat")
        scene = UniversalSceneState(
            threats=[threat],
            threat_urgency=0.9,
            game_phase="in_battle",
            recommended_action="jump",
        )
        text = self.brain._build_verbal_state(RUNNER, scene)
        self.assertIn("in_battle", text)
        self.assertIn("critical", text.lower())
        self.assertIn("Threat count", text)
        self.assertIn("jump", text)

    def test_clear_scene_no_false_threat_language(self):
        scene = UniversalSceneState(threat_urgency=0.0, threats=[], game_phase="active")
        text = self.brain._build_verbal_state(RUNNER, scene)
        self.assertNotIn("critical", text.lower())
        self.assertNotIn("Threat count", text)


class OpenJevRubricTests(unittest.TestCase):
    def test_choice_instructions_include_rubric(self):
        brain = make_brain()
        brain.openjev_engine = Mock()
        captured = {}

        def decide(state, questions):
            captured["questions"] = questions
            return [
                {"probabilities": {"jump": 0.8, "wait": 0.2}},
                {"probabilities": {"0": 0.1, "1": 0.1, "2": 0.2, "3": 0.3, "4": 0.3}},
                {"noul": 0.7},
            ]

        brain.openjev_engine.decide = decide
        with patch.object(brain, "_build_verbal_state", return_value="s"):
            res = brain._query_local_openjev(RUNNER, UniversalSceneState())
        self.assertIsNotNone(res)
        q0 = captured["questions"][0]
        self.assertIn("Allowed answers and rubric:", q0["instructions"])
        self.assertIn("jump", q0["instructions"])


if __name__ == "__main__":
    unittest.main()
