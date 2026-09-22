"""Regression: Clash tempo planner — opening, rate thresholds, counter-push window."""
import unittest

from adapters.clash_adapter import TacticalReflexPolicy
from clash_jev.state import BattleState, HandCard, LaneView


def make_state(
    elixir=5,
    elapsed_s=60.0,
    left=None,
    right=None,
    hand=None,
    seconds_at_full_elixir=0.0,
):
    return BattleState(
        elapsed_s=elapsed_s,
        elixir=elixir,
        hand=tuple(hand) if hand else (HandCard(0, "knight", True),),
        left=left or LaneView(),
        right=right or LaneView(),
        seconds_at_full_elixir=seconds_at_full_elixir,
    )


class TempoThresholdTests(unittest.TestCase):
    def test_single_rate_thresholds(self):
        self.assertEqual(TacticalReflexPolicy.tempo_thresholds(make_state()), (5, 8))

    def test_double_rate_thresholds(self):
        self.assertEqual(TacticalReflexPolicy.tempo_thresholds(make_state(elapsed_s=150)), (3, 6))

    def test_triple_rate_thresholds(self):
        self.assertEqual(TacticalReflexPolicy.tempo_thresholds(make_state(elapsed_s=300)), (2, 5))


class OpeningTempoTests(unittest.TestCase):
    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_opening_mid_elixir_pushes_or_cycles_not_save(self):
        for e in (4, 5, 6, 7):
            strat = self.policy.evaluate_strategy(make_state(elixir=e, elapsed_s=5.0))
            self.assertNotEqual(strat, "save_elixir", f"elixir={e} must not open-save")
            self.assertIn(strat, ("cycle", "push_left", "push_right"))

    def test_opening_low_elixir_still_saves(self):
        self.assertEqual(
            self.policy.evaluate_strategy(make_state(elixir=2, elapsed_s=3.0)),
            "save_elixir",
        )

    def test_opening_high_elixir_prefers_push(self):
        self.assertIn(
            self.policy.evaluate_strategy(make_state(elixir=7, elapsed_s=8.0)),
            ("push_left", "push_right"),
        )


class RateAwareSaveTests(unittest.TestCase):
    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_single_elixir_four_saves(self):
        self.assertEqual(
            self.policy.evaluate_strategy(make_state(elixir=4, elapsed_s=60.0)),
            "save_elixir",
        )

    def test_double_elixir_four_does_not_save(self):
        strat = self.policy.evaluate_strategy(make_state(elixir=4, elapsed_s=150.0))
        self.assertNotEqual(strat, "save_elixir")

    def test_double_elixir_two_saves(self):
        self.assertEqual(
            self.policy.evaluate_strategy(make_state(elixir=2, elapsed_s=150.0)),
            "save_elixir",
        )

    def test_single_push_at_eight(self):
        strat = self.policy.evaluate_strategy(make_state(elixir=8, elapsed_s=60.0))
        self.assertIn(strat, ("push_left", "push_right", "build_push"))

    def test_double_push_at_six(self):
        strat = self.policy.evaluate_strategy(make_state(elixir=6, elapsed_s=150.0))
        self.assertIn(strat, ("push_left", "push_right", "build_push"))


class CounterPushWindowTests(unittest.TestCase):
    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_recent_defend_converts_to_counter_push(self):
        threat_lane = LaneView(enemy_on_my_side=1)
        self.assertEqual(
            self.policy.evaluate_strategy(
                make_state(elixir=6, elapsed_s=40.0, left=threat_lane),
                threat_urgency=0.5,
            ),
            "defend_left",
        )
        survivors = LaneView(mine_on_my_side=1)
        strat = self.policy.evaluate_strategy(
            make_state(elixir=5, elapsed_s=45.0, left=survivors),
            threat_urgency=0.0,
        )
        self.assertEqual(strat, "counter_push_left")

    def test_stale_defend_does_not_force_counter(self):
        # No prior defend: survivors alone at mid elixir should not force counter
        # (may still cycle/push via ladder — just never defend ghost).
        survivors = LaneView(mine_on_my_side=1)
        strat = self.policy.evaluate_strategy(
            make_state(elixir=5, elapsed_s=90.0, left=survivors),
            threat_urgency=0.0,
        )
        self.assertFalse(strat.startswith("defend"))


class LeakGuardTests(unittest.TestCase):
    def test_full_elixir_always_spends(self):
        policy = TacticalReflexPolicy()
        strat = policy.evaluate_strategy(make_state(elixir=10, elapsed_s=30.0))
        self.assertIn(strat, ("push_left", "push_right", "build_push"))


if __name__ == "__main__":
    unittest.main()
