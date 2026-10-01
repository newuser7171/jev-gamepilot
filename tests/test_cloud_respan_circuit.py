"""Cloud tier: Respan gateway repoint + fail-streak circuits (item 3)."""
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy

os.environ["LOCAL_STRATEGY_DIR"] = ""  # never boot the model inside unit tests

from adapters.clash_adapter import ClashBattleAdapter
from clash_jev.state import BattleState, HandCard, LaneView, Towers

ENV_CLOUD = {
    "RESPAN_API_KEY": "test-key",
    "RESPAN_BASE_URL": "https://api.respan.ai/api/",
    "TYPESAFE_API_KEY": "test-typesafe-key",
}


def make_state(elixir=5, elapsed_s=60.0):
    return BattleState(
        elapsed_s=elapsed_s,
        elixir=elixir,
        hand=(HandCard(0, "knight", True),),
        left=LaneView(),
        right=LaneView(),
        towers=Towers(),
        seconds_at_full_elixir=0.0,
        units=(),
        enemy_elixir_estimate=None,
    )


def make_adapter():
    return ClashBattleAdapter()


class StubTypesafeClient:
    def __init__(self, conf=0.9, fail=False, strategy="push_left"):
        self.calls = 0
        self.conf = conf
        self.fail = fail
        self.strategy = strategy

    def system_one(self, state, q):
        self.calls += 1
        if self.fail:
            raise RuntimeError("402 payment required")
        ans = SimpleNamespace(
            choice=self.strategy,
            probabilities={self.strategy: self.conf},
        )
        return SimpleNamespace(choices={"strategy": ans})


class RespanQueryTests(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter()
        self.state = make_state()

    def test_json_reply_maps_to_strategy_source(self):
        self.adapter._respan_chat = Mock(
            return_value='{"strategy": "save_elixir", "confidence": 0.91}'
        )
        pick = self.adapter._query_respan_strategy(self.state)
        self.assertEqual(pick, ("save_elixir", 0.91, "respan_cloud"))
        self.assertEqual(self.adapter._respan_fail_streak, 0)
        self.adapter._respan_chat.assert_called_once()

    def test_fenced_reply_and_percent_confidence(self):
        self.adapter._respan_chat = Mock(
            return_value='sure, here:\n```json\n{"strategy": "push_left", "confidence": 80}\n```'
        )
        pick = self.adapter._query_respan_strategy(self.state)
        self.assertEqual(pick[0], "push_left")
        self.assertAlmostEqual(pick[1], 0.80, places=5)

    def test_invented_strategy_name_rejected(self):
        self.adapter._respan_chat = Mock(
            return_value='{"strategy": "fly_to_moon", "confidence": 0.99}'
        )
        self.assertIsNone(self.adapter._query_respan_strategy(self.state))

    def test_low_conf_rejected_but_gateway_counts_healthy(self):
        self.adapter._respan_chat = Mock(
            return_value='{"strategy": "push_right", "confidence": 0.30}'
        )
        self.assertIsNone(self.adapter._query_respan_strategy(self.state))
        self.assertEqual(self.adapter._respan_fail_streak, 0)

    def test_http_failures_open_circuit_after_three(self):
        import urllib.error

        self.adapter._respan_chat = Mock(
            side_effect=urllib.error.URLError("401 activation_required")
        )
        for _ in range(3):
            self.assertIsNone(self.adapter._query_respan_strategy(self.state))
        self.assertEqual(self.adapter._respan_fail_streak, 3)
        self.assertGreater(self.adapter._respan_dead_until, time.time())
        # Fourth attempt: circuit open, gateway not contacted at all.
        self.assertIsNone(self.adapter._query_respan_strategy(self.state))
        self.assertEqual(self.adapter._respan_chat.call_count, 3)
        self.assertIsNotNone(self.adapter._respan_last_error)

    def test_success_resets_fail_streak(self):
        self.adapter._respan_fail_streak = 2
        self.adapter._respan_dead_until = 0.0
        self.adapter._respan_chat = Mock(
            return_value='{"strategy": "defend_centre", "confidence": 0.7}'
        )
        pick = self.adapter._query_respan_strategy(self.state)
        self.assertEqual(pick[0], "defend_centre")
        self.assertEqual(self.adapter._respan_fail_streak, 0)


class TypesafeCircuitTests(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter()
        self.state = make_state()

    def test_success_returns_legacy_source(self):
        self.adapter.jev_client = StubTypesafeClient(conf=0.9)
        with patch.dict(os.environ, ENV_CLOUD):
            pick = self.adapter._query_typesafe_strategy(self.state)
        self.assertEqual(pick, ("push_left", 0.9, "jev_system_one"))
        self.assertEqual(self.adapter._cloud_fail_streak, 0)

    def test_low_conf_rejected(self):
        self.adapter.jev_client = StubTypesafeClient(conf=0.4)
        with patch.dict(os.environ, ENV_CLOUD):
            self.assertIsNone(self.adapter._query_typesafe_strategy(self.state))

    def test_failures_open_circuit_after_three(self):
        self.adapter.jev_client = StubTypesafeClient(fail=True)
        with patch.dict(os.environ, ENV_CLOUD):
            for _ in range(3):
                self.assertIsNone(self.adapter._query_typesafe_strategy(self.state))
            self.assertEqual(self.adapter.jev_client.calls, 3)
            self.assertGreater(self.adapter._cloud_dead_until, time.time())
            self.assertIsNone(self.adapter._query_typesafe_strategy(self.state))
            self.assertEqual(self.adapter.jev_client.calls, 3)

    def test_missing_key_short_circuits(self):
        self.adapter.jev_client = StubTypesafeClient()
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": ""}):
            self.assertIsNone(self.adapter._query_typesafe_strategy(self.state))
        self.assertEqual(self.adapter.jev_client.calls, 0)


class DecideCloudGateTests(unittest.TestCase):
    def setUp(self):
        self.adapter = make_adapter()
        self.state = make_state(elixir=5, elapsed_s=60.0)

    def _decide(self):
        frame = numpy.zeros((32, 32, 3), dtype=numpy.uint8)
        with patch.object(self.adapter, "extract_state_from_frame", return_value=self.state):
            return self.adapter.decide(frame, threat_urgency=0.1)

    def test_respan_preferred_over_typesafe(self):
        self.adapter.jev_client = StubTypesafeClient(conf=0.9)
        with patch.dict(os.environ, ENV_CLOUD), \
                patch.object(self.adapter, "_respan_chat",
                             return_value='{"strategy": "save_elixir", "confidence": 0.85}'):
            res = self._decide()
        self.assertEqual(res["source"], "respan_cloud")
        self.assertEqual(res["strategy"], "save_elixir")
        self.assertEqual(self.adapter.jev_client.calls, 0)

    def test_respan_circuit_open_falls_to_typesafe(self):
        self.adapter.jev_client = StubTypesafeClient(conf=0.9)
        self.adapter._respan_dead_until = time.time() + 999.0
        with patch.dict(os.environ, ENV_CLOUD), \
                patch.object(self.adapter, "_respan_chat",
                             side_effect=AssertionError("gateway must not be hit")):
            res = self._decide()
        self.assertEqual(res["source"], "jev_system_one")
        self.assertEqual(self.adapter.jev_client.calls, 1)

    def test_both_circuits_open_costs_zero_network(self):
        self.adapter.jev_client = StubTypesafeClient(fail=True)
        with patch.dict(os.environ, ENV_CLOUD):
            for _ in range(3):  # open both circuits
                self.adapter._respan_fail_streak += 1
                self.adapter._query_typesafe_strategy(self.state)
            calls = self.adapter.jev_client.calls
            with patch.object(self.adapter, "_respan_chat",
                              side_effect=AssertionError("gateway must not be hit")):
                res = self._decide()
        self.assertEqual(res["source"], "tactical_reflex")
        self.assertEqual(self.adapter.jev_client.calls, calls)

    def test_no_cloud_env_falls_straight_to_reflex(self):
        self.adapter.jev_client = None
        with patch.dict(os.environ, {"RESPAN_API_KEY": "", "RESPAN_BASE_URL": "",
                                     "TYPESAFE_API_KEY": ""}):
            res = self._decide()
        self.assertEqual(res["source"], "tactical_reflex")


if __name__ == "__main__":
    unittest.main()
