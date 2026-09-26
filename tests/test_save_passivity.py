"""Anti-passivity breaker: third consecutive save with banked elixir cycles instead (item 4)."""
import os
import unittest

os.environ["LOCAL_STRATEGY_DIR"] = ""  # never boot the model inside unit tests

from adapters.clash_adapter import TacticalReflexPolicy
from clash_jev.state import BattleState, HandCard, LaneView, Towers


def make_state(elixir=3, elapsed_s=60.0, hand=None):
    return BattleState(
        elapsed_s=elapsed_s,
        elixir=elixir,
        hand=hand or (HandCard(0, "knight", True),),
        left=LaneView(),
        right=LaneView(),
        towers=Towers(),
        seconds_at_full_elixir=0.0,
        units=(),
        enemy_elixir_estimate=None,
    )


class SaveStreakTests(unittest.TestCase):
    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_third_consecutive_save_cycles(self):
        state = make_state(elixir=3)
        self.assertEqual(self.policy.evaluate_strategy(state), "save_elixir")
        self.assertEqual(self.policy.evaluate_strategy(state), "save_elixir")
        self.assertEqual(self.policy.evaluate_strategy(state), "cycle")
        # Streak consumed — the next hold starts a fresh count.
        self.assertEqual(self.policy.evaluate_strategy(state), "save_elixir")

    def test_attack_choice_resets_the_streak(self):
        self.assertEqual(self.policy.evaluate_strategy(make_state(elixir=3)), "save_elixir")
        pushed = self.policy.evaluate_strategy(make_state(elixir=6))
        self.assertNotEqual(pushed, "save_elixir")
        self.assertEqual(self.policy.evaluate_strategy(make_state(elixir=3)), "save_elixir")
        # streak was 1+1 after the push — a save now must NOT convert yet.
        self.assertEqual(self.policy.evaluate_strategy(make_state(elixir=3)), "save_elixir")

    def test_full_hand_expensive_cards_never_converts(self):
        fat_hand = (HandCard(0, "giant", True), HandCard(1, "fireball", True))
        state = make_state(elixir=3, hand=fat_hand)
        for _ in range(4):
            self.assertEqual(self.policy.evaluate_strategy(state), "save_elixir")

    def test_fresh_battle_boundary_clears_inherited_streak(self):
        self.policy._save_streak = 2  # stale holds from the previous battle
        fresh = make_state(elixir=3, elapsed_s=1.0)
        self.assertEqual(self.policy.evaluate_strategy(fresh), "save_elixir")
        self.assertEqual(self.policy._save_streak, 1)

    def test_too_poor_to_convert_stays_saved(self):
        # elixir 2 fails the affordability gate — the hold is honest, not passive.
        state = make_state(elixir=2)
        for _ in range(3):
            self.assertEqual(self.policy.evaluate_strategy(state), "save_elixir")
        self.assertEqual(self.policy._save_streak, 3)

    def test_conversion_never_masks_live_threat_defence(self):
        threat = make_state(elixir=3, elapsed_s=60.0)
        threat = BattleState(
            elapsed_s=60.0,
            elixir=3,
            hand=threat.hand,
            left=LaneView(enemy_on_my_side=1),
            right=LaneView(),
            towers=Towers(),
            seconds_at_full_elixir=0.0,
            units=(),
            enemy_elixir_estimate=None,
        )
        for _ in range(4):
            self.assertEqual(self.policy.evaluate_strategy(threat), "defend_left")


if __name__ == "__main__":
    unittest.main()
