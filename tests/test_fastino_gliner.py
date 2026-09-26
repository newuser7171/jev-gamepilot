"""Fastino hosted GLiNER2.5-Decide tier: parse shapes, circuit, cooldown, escalation order."""
import io
import json
import os
import time
import unittest
import urllib.error
from unittest.mock import patch

import universal_brain as brain_module
from profile_manager import DEFAULT_PROFILES
from universal_vision import UniversalSceneState

RUNNER = next(p for p in DEFAULT_PROFILES if p.id == "runner_3lane")


def make_brain(key="fast_sk_test"):
    with patch.object(brain_module.threading.Thread, "start"), \
            patch.object(brain_module.UniversalBrain, "_start_laya_loader"), \
            patch.object(brain_module.UniversalBrain, "_start_llm2jev_loader"), \
            patch.object(brain_module.UniversalBrain, "_start_openjev_loader"):
        brain = brain_module.UniversalBrain()
    brain.fastino_key = key
    brain.fastino_timeout = 0.5
    return brain


def make_scene(urgency=0.5):
    return UniversalSceneState(threat_urgency=urgency, game_phase="active")


class _Resp:
    def __init__(self, body):
        self._buf = io.BytesIO(json.dumps(body).encode("utf-8"))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._buf.read()


def _chat(content_obj):
    return {"choices": [{"message": {"content": json.dumps(content_obj)}}]}


def _http(code, msg="err"):
    return urllib.error.HTTPError("https://api.fastino.ai/v1/chat/completions", code, msg, None, None)


class FastinoParseTests(unittest.TestCase):
    def test_flat_string_content(self):
        brain = make_brain()
        body = _chat({
            "tactical_action": "dodge_left",
            "threat_severity": "3",
            "is_urgent_reflex": "no",
        })
        with patch.object(brain_module.urllib.request, "urlopen", return_value=_Resp(body)):
            res = brain._query_fastino_gliner(RUNNER, make_scene())
        self.assertEqual(res["action"], "dodge_left")
        self.assertEqual(res["threat_score"], 0.75)
        self.assertEqual(res["confidence"], 0.80)
        self.assertEqual(res["source"], "fastino_gliner_decide")

    def test_label_confidence_dicts_and_reflex_raise_threat(self):
        brain = make_brain()
        body = _chat({
            "tactical_action": {"label": "dodge_right", "confidence": 0.93},
            "threat_severity": {"label": "0", "confidence": 0.9},
            "is_urgent_reflex": {"label": "yes", "confidence": 0.8},
        })
        with patch.object(brain_module.urllib.request, "urlopen", return_value=_Resp(body)):
            res = brain._query_fastino_gliner(RUNNER, make_scene())
        self.assertEqual(res["action"], "dodge_right")
        self.assertEqual(res["confidence"], 0.93)
        self.assertGreaterEqual(res["threat_score"], 0.9)

    def test_nested_classifications_wrapper(self):
        brain = make_brain()
        body = _chat({
            "classifications": {
                "tactical_action": {"value": "wait", "confidence": 0.7},
                "threat_severity": {"value": "2"},
                "is_urgent_reflex": {"value": "no"},
            }
        })
        with patch.object(brain_module.urllib.request, "urlopen", return_value=_Resp(body)):
            res = brain._query_fastino_gliner(RUNNER, make_scene())
        self.assertEqual(res["action"], "wait")
        self.assertEqual(res["confidence"], 0.7)
        self.assertEqual(res["threat_score"], 0.5)

    def test_probability_map_argmax(self):
        brain = make_brain()
        body = _chat({
            "tactical_action": {"probabilities": {"dodge_left": 0.6, "wait": 0.4}},
            "threat_severity": "1",
            "is_urgent_reflex": "no",
        })
        with patch.object(brain_module.urllib.request, "urlopen", return_value=_Resp(body)):
            res = brain._query_fastino_gliner(RUNNER, make_scene())
        self.assertEqual(res["action"], "dodge_left")
        self.assertAlmostEqual(res["confidence"], 0.6, places=3)

    def test_label_list_and_percent_confidence(self):
        brain = make_brain()
        body = _chat({
            "tactical_action": {"label": ["dodge_left"], "confidence": 87},
            "threat_severity": "4",
            "is_urgent_reflex": "no",
        })
        with patch.object(brain_module.urllib.request, "urlopen", return_value=_Resp(body)):
            res = brain._query_fastino_gliner(RUNNER, make_scene())
        self.assertEqual(res["action"], "dodge_left")
        self.assertAlmostEqual(res["confidence"], 0.87, places=3)
        self.assertEqual(res["threat_score"], 1.0)

    def test_unknown_action_coerced_to_wait(self):
        brain = make_brain()
        body = _chat({
            "tactical_action": "zzz_bogus",
            "threat_severity": "0",
            "is_urgent_reflex": "no",
        })
        with patch.object(brain_module.urllib.request, "urlopen", return_value=_Resp(body)):
            res = brain._query_fastino_gliner(RUNNER, make_scene())
        self.assertEqual(res["action"], "wait")

    def test_unparseable_content_returns_none_without_killing_tier(self):
        brain = make_brain()
        body = {"choices": [{"message": {"content": "definitely not json"}}]}
        with patch.object(brain_module.urllib.request, "urlopen", return_value=_Resp(body)):
            res = brain._query_fastino_gliner(RUNNER, make_scene())
        self.assertIsNone(res)
        self.assertFalse(brain._fastino_dead)
        self.assertEqual(brain._fastino_retry_at, 0.0)


class FastinoCircuitTests(unittest.TestCase):
    def test_401_kills_tier_and_skips_future_http(self):
        brain = make_brain()
        with patch.object(brain_module.urllib.request, "urlopen", side_effect=_http(401)) as u:
            self.assertIsNone(brain._query_fastino_gliner(RUNNER, make_scene()))
        self.assertTrue(brain._fastino_dead)
        self.assertIsNone(brain._query_fastino_gliner(RUNNER, make_scene()))
        self.assertEqual(u.call_count, 1)

    def test_503_billing_sets_cooldown_not_dead(self):
        brain = make_brain()
        with patch.object(brain_module.urllib.request, "urlopen", side_effect=_http(503)) as u:
            self.assertIsNone(brain._query_fastino_gliner(RUNNER, make_scene()))
        self.assertFalse(brain._fastino_dead)
        self.assertGreater(brain._fastino_retry_at, time.monotonic())
        self.assertIsNone(brain._query_fastino_gliner(RUNNER, make_scene()))
        self.assertEqual(u.call_count, 1)

    def test_402_sets_long_cooldown(self):
        brain = make_brain()
        with patch.object(brain_module.urllib.request, "urlopen", side_effect=_http(402)):
            self.assertIsNone(brain._query_fastino_gliner(RUNNER, make_scene()))
        self.assertFalse(brain._fastino_dead)
        self.assertGreaterEqual(brain._fastino_retry_at, time.monotonic() + 55.0)

    def test_rate_gap_blocks_second_call(self):
        brain = make_brain()
        brain._fastino_last_at = time.monotonic()
        with patch.object(brain_module.urllib.request, "urlopen") as u:
            self.assertIsNone(brain._query_fastino_gliner(RUNNER, make_scene()))
        u.assert_not_called()

    def test_missing_key_makes_no_http(self):
        brain = make_brain(key="")
        with patch.object(brain_module.urllib.request, "urlopen") as u:
            self.assertIsNone(brain._query_fastino_gliner(RUNNER, make_scene()))
        u.assert_not_called()

    def test_request_shape_headers_and_schema(self):
        brain = make_brain()
        body = _chat({"tactical_action": "wait", "threat_severity": "0", "is_urgent_reflex": "no"})
        with patch.object(brain_module.urllib.request, "urlopen", return_value=_Resp(body)) as u:
            brain._query_fastino_gliner(RUNNER, make_scene())
        req = u.call_args[0][0]
        self.assertTrue(req.full_url.endswith("/chat/completions"))
        self.assertEqual(req.get_header("X-api-key"), "fast_sk_test")
        sent = json.loads(req.data.decode("utf-8"))
        self.assertEqual(sent["model"], brain.fastino_model)
        tasks = [c["task"] for c in sent["schema"]["classifications"]]
        self.assertEqual(tasks, ["tactical_action", "threat_severity", "is_urgent_reflex"])
        self.assertTrue(sent["include_confidence"])
        self.assertEqual(u.call_args[1].get("timeout"), 0.5)


class FastinoEscalationTests(unittest.TestCase):
    def test_hosted_confident_return_skips_local_loader_and_llm2jev(self):
        brain = make_brain()
        hosted = {"action": "dodge_left", "threat_score": 0.75, "confidence": 0.95,
                  "latency_ms": 120.0, "source": "fastino_gliner_decide", "target_coords": None}
        brain.llm2jev_engine = object()
        with patch.object(brain, "_query_fastino_gliner", return_value=hosted), \
                patch.object(brain, "_start_gliner_loader") as loader, \
                patch.object(brain, "_query_local_llm2jev") as deep:
            res = brain._local_escalation(RUNNER, make_scene())
        self.assertEqual(res["source"], "fastino_gliner_decide")
        loader.assert_not_called()
        deep.assert_not_called()

    def test_hosted_absent_still_lazily_loads_local_gliner(self):
        brain = make_brain()
        with patch.object(brain, "_query_fastino_gliner", return_value=None), \
                patch.object(brain, "_start_gliner_loader") as loader:
            res = brain._local_escalation(RUNNER, make_scene())
        loader.assert_called_once()
        self.assertIsNone(res)

    def test_hosted_weak_local_better_wins(self):
        brain = make_brain()
        brain.gliner_engine = object()
        weak = {"action": "wait", "threat_score": 0.2, "confidence": 0.5,
                "latency_ms": 100.0, "source": "fastino_gliner_decide", "target_coords": None}
        local = {"action": "dodge_left", "threat_score": 0.7, "confidence": 0.88,
                 "latency_ms": 2700.0, "source": "local_gliner_decide", "target_coords": None}
        with patch.object(brain, "_query_fastino_gliner", return_value=weak), \
                patch.object(brain, "_query_gliner_decide", return_value=local):
            res = brain._local_escalation(RUNNER, make_scene())
        self.assertEqual(res["source"], "local_gliner_decide")

    def test_hosted_weak_kept_when_local_absent(self):
        brain = make_brain()
        brain._gliner_tried = True
        weak = {"action": "wait", "threat_score": 0.2, "confidence": 0.5,
                "latency_ms": 100.0, "source": "fastino_gliner_decide", "target_coords": None}
        with patch.object(brain, "_query_fastino_gliner", return_value=weak), \
                patch.object(brain, "_start_gliner_loader") as loader:
            res = brain._local_escalation(RUNNER, make_scene())
        loader.assert_not_called()
        self.assertEqual(res["source"], "fastino_gliner_decide")
        self.assertEqual(res["confidence"], 0.5)


class FastinoConfigTests(unittest.TestCase):
    def test_env_overrides(self):
        env = {
            "FASTINO_URL": "https://fast.example/v1/",
            "FASTINO_GLINER_MODEL": "fastino/custom-decide",
            "FASTINO_TIMEOUT": "2.5",
        }
        with patch.dict(os.environ, env), \
                patch.object(brain_module.threading.Thread, "start"), \
                patch.object(brain_module.UniversalBrain, "_start_laya_loader"), \
                patch.object(brain_module.UniversalBrain, "_start_llm2jev_loader"), \
                patch.object(brain_module.UniversalBrain, "_start_openjev_loader"):
            brain = brain_module.UniversalBrain()
        self.assertEqual(brain.fastino_url, "https://fast.example/v1")
        self.assertEqual(brain.fastino_model, "fastino/custom-decide")
        self.assertEqual(brain.fastino_timeout, 2.5)


if __name__ == "__main__":
    unittest.main()
