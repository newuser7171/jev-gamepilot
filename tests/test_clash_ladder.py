"""Regression: top-ladder defend/cycle/trade fixes (no giant-cycle, no bridge-defend, centre pull)."""
import unittest

from adapters.clash_adapter import TacticalReflexPolicy
from clash_jev.state import BattleState, HandCard, LaneView, Towers
from clash_jev.units import Unit


def make_state(
    elixir=8,
    elapsed_s=60.0,
    left=None,
    right=None,
    hand=None,
    units=None,
    enemy_elixir_estimate=None,
    towers=None,
    seconds_at_full_elixir=0.0,
):
    return BattleState(
        elapsed_s=elapsed_s,
        elixir=elixir,
        hand=tuple(hand) if hand else (HandCard(0, "knight", True),),
        left=left or LaneView(),
        right=right or LaneView(),
        towers=towers or Towers(),
        seconds_at_full_elixir=seconds_at_full_elixir,
        units=tuple(units) if units else (),
        enemy_elixir_estimate=enemy_elixir_estimate,
    )


class CycleNeverPicksWinConditionTests(unittest.TestCase):
    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_cycle_prefers_cheap_over_giant(self):
        state = make_state(
            elixir=8,
            hand=[
                HandCard(0, "giant", True),
                HandCard(1, "bomber", True),
                HandCard(2, "archers", True),
            ],
        )
        card = self.policy.select_card("cycle", state)
        self.assertIsNotNone(card)
        self.assertNotEqual(card.name, "giant")
        self.assertIn(card.name, ("bomber", "archers"))

    def test_cycle_giant_only_when_no_cheap_ready(self):
        state = make_state(
            elixir=8,
            hand=[
                HandCard(0, "giant", True),
                HandCard(1, "fireball", True),  # filtered on cycle
            ],
        )
        card = self.policy.select_card("cycle", state)
        self.assertIsNotNone(card)
        self.assertEqual(card.name, "giant")


class DefendNeverPicksWinConditionTests(unittest.TestCase):
    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_defend_prefers_fighter_over_giant(self):
        state = make_state(
            elixir=8,
            hand=[
                HandCard(0, "giant", True),
                HandCard(1, "knight", True),
            ],
            left=LaneView(enemy_on_my_side=1),
            units=(Unit(owner="enemy", x=0.3, y=0.55, health=1.0, name="mini_pekka"),),
        )
        card = self.policy.select_card("defend_left", state)
        self.assertIsNotNone(card)
        self.assertEqual(card.name, "knight")

    def test_defend_giant_only_when_no_fighter(self):
        state = make_state(
            elixir=8,
            hand=[HandCard(0, "giant", True)],
            left=LaneView(enemy_on_my_side=1),
        )
        # Only win-con ready — fighters list empty → giant may be last resort
        card = self.policy.select_card("defend_left", state)
        self.assertIsNotNone(card)


class CentrePullTests(unittest.TestCase):
    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_hog_triggers_centre_pull_strategy(self):
        state = make_state(
            elixir=6,
            left=LaneView(enemy_on_my_side=1),
            units=(Unit(owner="enemy", x=0.3, y=0.55, health=1.0, name="hog"),),
        )
        strat = self.policy.evaluate_strategy(state, threat_urgency=0.6)
        self.assertEqual(strat, "defend_centre")

    def test_building_only_flagged(self):
        state = make_state(
            elixir=6,
            units=(Unit(owner="enemy", x=0.7, y=0.6, health=1.0, name="royal_giant"),),
            right=LaneView(enemy_on_my_side=1),
        )
        self.assertTrue(self.policy._needs_centre_pull(state))

    def test_flying_win_condition_not_centre_pull(self):
        state = make_state(
            elixir=6,
            units=(Unit(owner="enemy", x=0.3, y=0.55, health=1.0, name="balloon"),),
            left=LaneView(enemy_on_my_side=1),
        )
        self.assertFalse(self.policy._needs_centre_pull(state))


class DefendSquareTests(unittest.TestCase):
    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_defend_left_never_bridge(self):
        card = HandCard(0, "knight", True)
        state = make_state(
            elixir=8,
            hand=[card],
            left=LaneView(enemy_on_my_side=1),
        )
        sq = self.policy.select_square("defend_left", card, state)
        self.assertIsNotNone(sq)
        self.assertNotEqual(sq.name, "left_bridge")
        self.assertIn(sq.name, ("left_tower_front", "centre_king_front", "centre"))

    def test_defend_centre_prefers_king_front(self):
        card = HandCard(0, "cannon", True)
        state = make_state(
            elixir=8,
            hand=[card],
            left=LaneView(enemy_on_my_side=1),
            units=(Unit(owner="enemy", x=0.3, y=0.55, health=1.0, name="giant"),),
        )
        sq = self.policy.select_square("defend_centre", card, state)
        self.assertIsNotNone(sq)
        self.assertIn(sq.name, ("centre_king_front", "centre"))


class ElixirTradeTests(unittest.TestCase):
    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_cheap_swarm_not_answered_by_giant_trade_penalty(self):
        # Two named goblins ≈ 2 elixir threat value; giant overpays hard vs knight.
        state = make_state(
            elixir=8,
            hand=[
                HandCard(0, "giant", True),
                HandCard(1, "knight", True),
            ],
            left=LaneView(enemy_on_my_side=2),
            units=(
                Unit(owner="enemy", x=0.28, y=0.5, health=1.0, name="goblins"),
                Unit(owner="enemy", x=0.32, y=0.52, health=1.0, name="goblins"),
            ),
        )
        card = self.policy.select_card("defend_left", state)
        self.assertIsNotNone(card)
        # fighters filter already drops giant; knight wins
        self.assertEqual(card.name, "knight")

    def test_threat_value_sums_named_costs(self):
        state = make_state(
            elixir=8,
            units=(
                Unit(owner="enemy", x=0.3, y=0.55, health=1.0, name="hog"),  # 4
                Unit(owner="enemy", x=0.35, y=0.5, health=1.0, name="musketeer"),  # 4
            ),
        )
        self.assertGreaterEqual(self.policy._threat_elixir_value(state), 8.0)


class WeightedThreatTests(unittest.TestCase):
    def test_tank_outweighs_skeleton_count(self):
        tank = Unit(owner="enemy", x=0.3, y=0.6, health=1.0, name="giant")
        skel_a = Unit(owner="enemy", x=0.7, y=0.5, health=1.0, name="skeletons")
        skel_b = Unit(owner="enemy", x=0.72, y=0.5, health=1.0, name="skeletons")
        skel_c = Unit(owner="enemy", x=0.74, y=0.5, health=1.0, name="skeletons")
        state = make_state(elixir=8, units=(tank, skel_a, skel_b, skel_c))
        wl, wr = TacticalReflexPolicy._weighted_threat_sides(state)
        self.assertGreater(wl, wr)


class MidGameWeakTowerTests(unittest.TestCase):
    def test_midgame_commits_to_melted_lane(self):
        policy = TacticalReflexPolicy()
        for _ in range(4):
            strat = policy.evaluate_strategy(
                make_state(
                    elixir=8,
                    elapsed_s=90.0,
                    towers=Towers(
                        enemy_left=1.0,
                        enemy_right=0.4,
                        enemy_king=1.0,
                        my_left=1.0,
                        my_right=1.0,
                        my_king=1.0,
                    ),
                )
            )
            self.assertEqual(strat, "push_right", strat)


class CounterPushBothLanesTests(unittest.TestCase):
    def test_picks_busier_survivor_lane(self):
        policy = TacticalReflexPolicy()
        policy._last_defend_elapsed = 40.0
        state = make_state(
            elixir=6,
            elapsed_s=45.0,
            left=LaneView(mine_on_my_side=1),
            right=LaneView(mine_on_my_side=3),
        )
        strat = policy.evaluate_strategy(state, threat_urgency=0.0)
        self.assertEqual(strat, "counter_push_right")


class OpponentSeenDeckTests(unittest.TestCase):
    def test_seen_cards_accumulates_and_resets(self):
        from clash_jev.opponent import OpponentElixir

        opp = OpponentElixir()
        units = (Unit(owner="enemy", x=0.3, y=0.5, health=1.0, name="hog"),)
        opp.update(1.0, "single", units)
        # Second frame: same body is "known" — no double-count of elixir, but name stays.
        opp.update(2.0, "single", units)
        self.assertIn("hog", opp.seen_cards)
        # Fresh unit arrives
        opp.update(3.0, "single", units + (Unit(owner="enemy", x=0.7, y=0.5, health=1.0, name="musketeer"),))
        self.assertIn("musketeer", opp.seen_cards)
        opp.reset()
        self.assertEqual(opp.seen_cards, {})
        self.assertEqual(opp.elixir, 5.0)


class CycleSkipsBuildingsTests(unittest.TestCase):
    def test_cycle_prefers_troop_over_hut(self):
        policy = TacticalReflexPolicy()
        state = make_state(
            elixir=8,
            hand=[
                HandCard(0, "goblin_hut", True),
                HandCard(1, "archers", True),
            ],
        )
        card = policy.select_card("cycle", state)
        self.assertIsNotNone(card)
        self.assertEqual(card.name, "archers")


class SpellClusterTests(unittest.TestCase):
    def test_spell_aims_at_densest_cluster(self):
        policy = TacticalReflexPolicy()
        card = HandCard(0, "arrows", True)
        # Tight pair near centre-left vs a lone far-right body: arrows should take the pair.
        state = make_state(
            elixir=8,
            hand=[card],
            units=(
                Unit(owner="enemy", x=0.30, y=0.55, health=1.0, name="goblins"),
                Unit(owner="enemy", x=0.34, y=0.57, health=1.0, name="goblins"),
                Unit(owner="enemy", x=0.72, y=0.50, health=1.0, name="spear_goblins"),
            ),
        )
        sq = policy.select_square("defend_left", card, state)
        self.assertIsNotNone(sq)
        # Cluster squares are enemy_*_left_*; lone is right.
        self.assertIn("_left_", sq.name)


if __name__ == "__main__":
    unittest.main()
