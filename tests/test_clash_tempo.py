"""Regression: Clash tempo planner, spell gating, counter-push window, phase stickiness."""
import unittest

from adapters.clash_adapter import TacticalReflexPolicy
from clash_jev.moves import spell_targets
from clash_jev.state import BattleState, HandCard, LaneView
from clash_jev.units import Unit
from universal_vision import UniversalVision


def make_state(
    elixir=5,
    elapsed_s=60.0,
    left=None,
    right=None,
    hand=None,
    seconds_at_full_elixir=0.0,
    units=None,
    enemy_elixir_estimate=None,
    towers=None,
):
    from clash_jev.state import Towers

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


class SpellGatingTests(unittest.TestCase):
    """Fireball/arrows must never free-chip on cycle/push or empty-board defend."""

    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def _hand(self, *names):
        return [HandCard(i, n, True) for i, n in enumerate(names)]

    def test_cycle_never_picks_spell(self):
        state = make_state(elixir=8, hand=self._hand("fireball", "spear_goblins"))
        card = self.policy.select_card("cycle", state)
        self.assertIsNotNone(card)
        self.assertNotEqual(card.name, "fireball")

    def test_push_never_picks_spell(self):
        state = make_state(elixir=8, hand=self._hand("fireball", "giant"))
        card = self.policy.select_card("push_left", state)
        self.assertIsNotNone(card)
        self.assertNotEqual(card.name, "fireball")

    def test_single_threat_defend_skips_spell(self):
        state = make_state(
            elixir=8,
            hand=self._hand("fireball"),
            left=LaneView(enemy_on_my_side=1),
        )
        self.assertIsNone(self.policy.select_card("defend_left", state))

    def test_multi_threat_defend_may_use_spell(self):
        state = make_state(
            elixir=8,
            hand=self._hand("fireball", "knight"),
            left=LaneView(enemy_on_my_side=2),
            right=LaneView(enemy_on_my_side=1),
        )
        card = self.policy.select_card("defend_centre", state)
        self.assertIsNotNone(card)

    def test_spell_square_requires_enemy_troop(self):
        card = HandCard(0, "fireball", True)
        empty = make_state(elixir=8, hand=[card])
        squares = spell_targets(card, empty)
        # No enemy units → only standing towers in the legal set.
        self.assertFalse(
            any(sq.name.startswith("enemy_") and "tower" not in sq.name for sq in squares)
        )
        self.assertIsNone(self.policy.select_square("defend_left", card, empty))

    def test_spell_square_hits_troop_not_tower(self):
        card = HandCard(0, "fireball", True)
        state = make_state(
            elixir=8,
            hand=[card],
            units=(Unit(owner="enemy", x=0.3, y=0.6, health=1.0),),
        )
        sq = self.policy.select_square("defend_left", card, state)
        self.assertIsNotNone(sq)
        self.assertTrue(sq.name.startswith("enemy_"))
        self.assertNotIn("tower", sq.name)


class PhaseHysteresisTests(unittest.TestCase):
    """False main_menu blips must not exit battle or reset the match clock."""

    def setUp(self):
        self.vision = UniversalVision()

    def test_single_menu_blip_stays_in_battle(self):
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("in_battle"), "in_battle")
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("main_menu"), "in_battle")
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("in_battle"), "in_battle")

    def test_sustained_menu_leaves_battle(self):
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("in_battle"), "in_battle")
        results = [self.vision.apply_clash_phase_hysteresis("main_menu") for _ in range(4)]
        self.assertEqual(results[-1], "main_menu")
        self.assertTrue(all(r == "in_battle" for r in results[:3]))

    def test_matchmaking_and_game_over_streak_like_menu(self):
        # Single false game_over / matchmaking mid-battle must NOT exit —
        # it zeroed _push_counter via note_battle_start and pinned cycle right.
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("in_battle"), "in_battle")
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("game_over"), "in_battle")
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("in_battle"), "in_battle")
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("matchmaking"), "in_battle")
        # Fresh streak: sustained game_over still leaves after N frames
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("in_battle"), "in_battle")
        results = [self.vision.apply_clash_phase_hysteresis("game_over") for _ in range(4)]
        self.assertEqual(results[-1], "game_over")
        self.assertTrue(all(r == "in_battle" for r in results[:3]))

    def test_blip_then_battle_clears_menu_streak(self):
        self.vision.apply_clash_phase_hysteresis("in_battle")
        self.vision.apply_clash_phase_hysteresis("main_menu")
        self.vision.apply_clash_phase_hysteresis("main_menu")
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("in_battle"), "in_battle")
        # Streak reset — two more menu frames still not enough
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("main_menu"), "in_battle")
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("main_menu"), "in_battle")

    def test_mixed_non_battle_blips_stay_in_battle(self):
        # game_over blip, then matchmaking blip — streak must not treat them
        # as one continuous exit; each is interrupted by in_battle.
        self.vision.apply_clash_phase_hysteresis("in_battle")
        self.vision.apply_clash_phase_hysteresis("game_over")
        self.vision.apply_clash_phase_hysteresis("in_battle")
        self.vision.apply_clash_phase_hysteresis("matchmaking")
        self.vision.apply_clash_phase_hysteresis("in_battle")
        self.assertEqual(self.vision.apply_clash_phase_hysteresis("main_menu"), "in_battle")


class OpponentElixirGateTests(unittest.TestCase):
    """Never open a fresh push into a loaded opponent with no board to convert."""

    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_loaded_enemy_no_presence_saves_or_cycles(self):
        # Mid-game, enemy at 8+, we trail by 3+, empty board: hold, do not push.
        strat = self.policy.evaluate_strategy(
            make_state(elixir=5, elapsed_s=90.0, enemy_elixir_estimate=8.5)
        )
        self.assertIn(strat, ("save_elixir", "cycle"))

    def test_loaded_enemy_with_presence_still_counter_pushes(self):
        # Survivors on our side convert regardless of enemy elixir estimate.
        self.policy._last_defend_elapsed = 85.0
        strat = self.policy.evaluate_strategy(
            make_state(
                elixir=5,
                elapsed_s=90.0,
                left=LaneView(mine_on_my_side=1),
                enemy_elixir_estimate=8.5,
            )
        )
        self.assertEqual(strat, "counter_push_left")

    def test_comfortable_enemy_lead_still_pushes_at_threshold(self):
        # Enemy at 8 but we also at 8 (within 3): push gate does not fire.
        strat = self.policy.evaluate_strategy(
            make_state(elixir=8, elapsed_s=90.0, enemy_elixir_estimate=8.0)
        )
        self.assertIn(strat, ("push_left", "push_right", "build_push"))

    def test_unknown_enemy_estimate_does_not_block(self):
        # None estimate: no gate — normal push ladder at threshold.
        strat = self.policy.evaluate_strategy(make_state(elixir=8, elapsed_s=90.0))
        self.assertIn(strat, ("push_left", "push_right", "build_push"))


class TankSupportBufferTests(unittest.TestCase):
    """Giant/Golem from the back needs +2 elixir after the tank for a support troop."""

    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_giant_at_push_threshold_without_buffer_cycles(self):
        # Single rate push_at=8, Giant costs 5, buffer needed=2 → 5+2=7 < 8 is fine,
        # but at exactly push_at with Giant in hand: 8 >= 5+2 → build_push allowed.
        # At elixir=7 (below push_at) we never reach build_push — save/cycle path.
        # Force the tank branch: elixir=8, Giant in hand → build_push (buffer ok).
        state = make_state(
            elixir=8,
            elapsed_s=90.0,
            hand=(HandCard(0, "giant", True), HandCard(1, "knight", True)),
        )
        strat = self.policy.evaluate_strategy(state)
        self.assertEqual(strat, "build_push")

    def test_giant_no_buffer_below_tank_plus_two_cycles(self):
        # Double rate (pre-endgame): push_at=6. Giant=5, need 5+2=7 > 6 → no build_push.
        # Above save_below (3): cycle a cheap card rather than naked-tank.
        state = make_state(
            elixir=6,
            elapsed_s=140.0,
            hand=(HandCard(0, "giant", True), HandCard(1, "goblins", True)),
        )
        strat = self.policy.evaluate_strategy(state)
        self.assertEqual(strat, "cycle")

    def test_giant_no_buffer_below_save_floor_saves(self):
        # Double rate save_below=3: elixir=2 → save, not cycle.
        state = make_state(
            elixir=2,
            elapsed_s=140.0,
            hand=(HandCard(0, "giant", True),),
        )
        strat = self.policy.evaluate_strategy(state)
        self.assertEqual(strat, "save_elixir")

    def test_giant_with_buffer_builds(self):
        # Double rate pre-endgame: elixir=7 >= 5+2 → build_push.
        state = make_state(
            elixir=7,
            elapsed_s=140.0,
            hand=(HandCard(0, "giant", True),),
        )
        strat = self.policy.evaluate_strategy(state)
        self.assertEqual(strat, "build_push")


class EndgameUrgencyTests(unittest.TestCase):
    """Last 30s + overtime: clock beats elixir etiquette; finish the weaker tower."""

    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_endgame_behind_bypasses_enemy_gate(self):
        # Mid-game gate would save/cycle here — endgame+behind must attack.
        from clash_jev.state import Towers

        strat = self.policy.evaluate_strategy(
            make_state(
                elixir=5,
                elapsed_s=160.0,
                enemy_elixir_estimate=9.0,
                towers=Towers(enemy_left=1.0, enemy_right=1.0, enemy_king=1.0,
                              my_left=0.3, my_right=1.0, my_king=1.0),
            )
        )
        self.assertIn(strat, ("push_left", "push_right"))

    def test_midgame_behind_still_gates(self):
        from clash_jev.state import Towers

        strat = self.policy.evaluate_strategy(
            make_state(
                elixir=5,
                elapsed_s=90.0,
                enemy_elixir_estimate=9.0,
                towers=Towers(enemy_left=1.0, enemy_right=1.0, enemy_king=1.0,
                              my_left=0.3, my_right=1.0, my_king=1.0),
            )
        )
        self.assertIn(strat, ("save_elixir", "cycle"))

    def test_endgame_behind_lowers_push_threshold(self):
        from clash_jev.state import Towers

        # Double-rate push_at is 6; endgame+behind → 4. Elixir 4 must push, not save.
        strat = self.policy.evaluate_strategy(
            make_state(
                elixir=4,
                elapsed_s=160.0,
                towers=Towers(enemy_left=1.0, enemy_right=1.0, enemy_king=1.0,
                              my_left=0.0, my_right=1.0, my_king=1.0),
            )
        )
        self.assertIn(strat, ("push_left", "push_right"))

    def test_endgame_targets_weaker_tower(self):
        from clash_jev.state import Towers

        # Right tower melted → every endgame push goes right (pocket/finish).
        for _ in range(6):
            strat = self.policy.evaluate_strategy(
                make_state(
                    elixir=8,
                    elapsed_s=170.0,
                    towers=Towers(enemy_left=1.0, enemy_right=0.2, enemy_king=1.0,
                                  my_left=1.0, my_right=1.0, my_king=1.0),
                )
            )
            self.assertEqual(strat, "push_right", strat)

    def test_endgame_skips_slow_build_push(self):
        from clash_jev.state import Towers

        # Giant in hand + endgame: no build_push — bridge rush instead.
        strat = self.policy.evaluate_strategy(
            make_state(
                elixir=8,
                elapsed_s=155.0,
                hand=(HandCard(0, "giant", True),),
                towers=Towers(enemy_left=1.0, enemy_right=1.0, enemy_king=1.0,
                              my_left=1.0, my_right=1.0, my_king=1.0),
            )
        )
        self.assertIn(strat, ("push_left", "push_right"))


class CycleLaneTests(unittest.TestCase):
    """cycle strategy must alternate bridges — no left-bridge monoculture."""

    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_cycle_default_fallback_alternates(self):
        card = HandCard(0, "knight", True)
        state = make_state(elixir=8, hand=[card])
        names = []
        for _ in range(4):
            sq = self.policy.select_square("cycle", card, state)
            self.assertIsNotNone(sq)
            names.append(sq.name)
        self.assertIn("left_bridge", names)
        self.assertIn("right_bridge", names)
        # No three-in-a-row same lane on empty legal set
        for i in range(len(names) - 2):
            self.assertFalse(names[i] == names[i + 1] == names[i + 2], names)

    def test_push_counter_does_not_skew_cycle_parity(self):
        """evaluate_strategy push increments must not flip cycle's next lane."""
        card = HandCard(0, "knight", True)
        state = make_state(elixir=9, hand=[card])
        # First cycle tap
        sq1 = self.policy.select_square("cycle", card, state)
        self.assertIsNotNone(sq1)
        # Simulate a push decision bumping the shared push counter
        self.policy._push_counter += 3
        # Next cycle tap must still be the opposite lane
        sq2 = self.policy.select_square("cycle", card, state)
        self.assertIsNotNone(sq2)
        self.assertNotEqual(sq1.name, sq2.name, f"{sq1.name} then {sq2.name}")


if __name__ == "__main__":
    unittest.main()
