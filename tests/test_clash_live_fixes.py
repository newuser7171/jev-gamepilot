"""Regression: unknown-hand fallback, tower-bar phone geometry, select_card known-preference,
clash max-idle pressure gate (frozen-pilot fix)."""
import unittest
from unittest.mock import patch

import numpy

from adapters.clash_adapter import TacticalReflexPolicy
from clash_jev.hand import (
    HandReader,
    player_deck,
)
from clash_jev.perception import Layout, Perception, _TOWER_BAR_COLOUR, to_reference
from clash_jev.state import BattleState, HandCard, LaneView
from phone_pilot import clash_max_idle, clash_blind_force_action


def make_state(
    elixir=8,
    elapsed_s=60.0,
    left=None,
    right=None,
    hand=None,
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
        seconds_at_full_elixir=0.0,
        units=tuple(units) if units else (),
        enemy_elixir_estimate=enemy_elixir_estimate,
    )


class UnknownPreferenceTests(unittest.TestCase):
    """select_card must never blind-deploy an unknown slot while a known ready card exists."""

    def setUp(self):
        self.policy = TacticalReflexPolicy()

    def test_known_card_beats_unknown_on_cycle(self):
        state = make_state(
            elixir=8,
            hand=[
                HandCard(0, "unknown", True),
                HandCard(1, "knight", True),
                HandCard(2, "fireball", True),
            ],
        )
        card = self.policy.select_card("cycle", state)
        self.assertIsNotNone(card)
        self.assertEqual(card.name, "knight")

    def test_unknown_blocked_below_cost_floor(self):
        # Unknown assumes cost 4 — elixir 3 must wait, not blind-deploy into a toast.
        state = make_state(
            elixir=3,
            hand=[
                HandCard(0, "unknown", True),
                HandCard(1, "giant", True),  # cost 5 > elixir 3
                HandCard(2, "fireball", True),  # cost 4 > elixir 3
            ],
        )
        self.assertIsNone(self.policy.select_card("cycle", state))
        self.assertIsNone(self.policy.select_card("defend_centre", state))

    def test_unknown_only_when_no_known_ready(self):
        # defend with multi-body; giant unaffordable at 4 → unknown is the floor once elixir >= 4
        state = make_state(
            elixir=4,
            hand=[
                HandCard(0, "unknown", True),
                HandCard(1, "giant", True),  # cost 5 > elixir 4
                HandCard(2, "arrows", True),  # stripped on empty-ish / not preferred
            ],
            left=LaneView(enemy_on_my_side=3),
            right=LaneView(enemy_on_my_side=1),
            units=(
                __import__("clash_jev.units", fromlist=["Unit"]).Unit(
                    owner="enemy", x=0.3, y=0.5, health=1.0, name="goblins"
                ),
                __import__("clash_jev.units", fromlist=["Unit"]).Unit(
                    owner="enemy", x=0.35, y=0.55, health=1.0, name="goblins"
                ),
                __import__("clash_jev.units", fromlist=["Unit"]).Unit(
                    owner="enemy", x=0.4, y=0.52, health=1.0, name="goblins"
                ),
            ),
        )
        card = self.policy.select_card("defend_centre", state)
        self.assertIsNotNone(card)
        if card.name == "unknown":
            self.assertFalse(
                any(
                    c.name != "unknown"
                    and TacticalReflexPolicy._assumed_cost(c.name) <= 4
                    and c.ready
                    and c.name != "giant"
                    for c in state.hand
                ),
                "a known ready affordable card was available",
            )
        else:
            self.assertNotEqual(card.name, "giant")

    def test_all_unknown_still_returns_something(self):
        state = make_state(
            elixir=8,
            hand=[HandCard(i, "unknown", True) for i in range(4)],
        )
        card = self.policy.select_card("cycle", state)
        self.assertIsNotNone(card)
        self.assertEqual(card.name, "unknown")


class CatalogHandFallbackTests(unittest.TestCase):
    """HandReader must fall through to CatalogBank when ShapeBank cannot name a slot."""

    def test_catalog_accepts_in_deck_below_any_bar(self):
        reader = HandReader()
        with patch.object(reader.bank, "match", return_value=(None, 0.0, 1.0)), patch.object(
            reader.catalog, "match", return_value=("knight", 0.36, 0.01)
        ), patch.object(reader, "_read_next", return_value=None), patch(
            "clash_jev.hand._is_empty", return_value=False
        ), patch("clash_jev.hand._is_lit", return_value=True), patch(
            "clash_jev.hand._reference", side_effect=lambda f: f
        ), patch(
            "clash_jev.hand._detail", return_value=numpy.zeros(224, dtype=numpy.float32)
        ), patch(
            "clash_jev.hand._shape", return_value=numpy.zeros(224, dtype=numpy.float32)
        ):
            hand = reader.read(numpy.zeros((633, 419, 3), dtype=numpy.uint8))
        self.assertTrue(all(c.name == "knight" for c in hand), [c.name for c in hand])

    def test_catalog_rejects_weak_non_deck(self):
        reader = HandReader()
        # skeleton not in PLAYER_DECK — 0.40 is below _CATALOG_HAND_ANY_SCORE (0.45)
        with patch.object(reader.bank, "match", return_value=(None, 0.0, 1.0)), patch.object(
            reader.catalog, "match", return_value=("skeletons", 0.40, 0.05)
        ), patch.object(reader, "_read_next", return_value=None), patch(
            "clash_jev.hand._is_empty", return_value=False
        ), patch("clash_jev.hand._is_lit", return_value=True), patch(
            "clash_jev.hand._reference", side_effect=lambda f: f
        ), patch(
            "clash_jev.hand._detail", return_value=numpy.zeros(224, dtype=numpy.float32)
        ), patch(
            "clash_jev.hand._shape", return_value=numpy.zeros(224, dtype=numpy.float32)
        ):
            hand = reader.read(numpy.zeros((633, 419, 3), dtype=numpy.uint8))
        self.assertTrue(all(c.name == "unknown" for c in hand), [c.name for c in hand])

    def test_catalog_accepts_confident_non_deck(self):
        reader = HandReader()
        with patch.object(reader.bank, "match", return_value=(None, 0.0, 1.0)), patch.object(
            reader.catalog, "match", return_value=("skeletons", 0.52, 0.08)
        ), patch.object(reader, "_read_next", return_value=None), patch(
            "clash_jev.hand._is_empty", return_value=False
        ), patch("clash_jev.hand._is_lit", return_value=True), patch(
            "clash_jev.hand._reference", side_effect=lambda f: f
        ), patch(
            "clash_jev.hand._detail", return_value=numpy.zeros(224, dtype=numpy.float32)
        ), patch(
            "clash_jev.hand._shape", return_value=numpy.zeros(224, dtype=numpy.float32)
        ):
            hand = reader.read(numpy.zeros((633, 419, 3), dtype=numpy.uint8))
        self.assertTrue(all(c.name == "skeletons" for c in hand), [c.name for c in hand])

    def test_player_deck_is_stale_vs_live_skeletons(self):
        # Documents the live finding: Skeletons appeared in hand but not in taught deck.
        self.assertNotIn("skeletons", player_deck())


class InDeckThinLeadTests(unittest.TestCase):
    """Greyed hand art: solid in-deck bank score with a thin lead must stay known, and must
    block a non-deck catalog false-accept (golem / flying_machine / …)."""

    def _read(self, reader, bank_ret, catalog_ret):
        with patch.object(reader.bank, "match", return_value=bank_ret), patch.object(
            reader.catalog, "match", return_value=catalog_ret
        ), patch.object(reader, "_read_next", return_value=None), patch(
            "clash_jev.hand._is_empty", return_value=False
        ), patch("clash_jev.hand._is_lit", return_value=True), patch(
            "clash_jev.hand._reference", side_effect=lambda f: f
        ), patch(
            "clash_jev.hand._detail", return_value=numpy.zeros(224, dtype=numpy.float32)
        ), patch(
            "clash_jev.hand._shape", return_value=numpy.zeros(224, dtype=numpy.float32)
        ):
            return reader.read(numpy.zeros((633, 419, 3), dtype=numpy.uint8))

    def test_in_deck_thin_lead_stays_known(self):
        # Live pattern: fireball/giant/archers at 0.52-0.57 with lead < 0.08.
        reader = HandReader()
        hand = self._read(reader, ("knight", 0.55, 0.01), ("golem", 0.90, 0.50))
        self.assertTrue(all(c.name == "knight" for c in hand), [c.name for c in hand])

    def test_in_deck_near_blocks_non_deck_catalog(self):
        # Bank has a competent in-deck near-miss (>= _IN_DECK_NEAR) but not enough for known;
        # catalog must not promote a non-deck name over it.
        reader = HandReader()
        hand = self._read(reader, ("giant", 0.35, 0.10), ("flying_machine", 0.55, 0.10))
        self.assertTrue(all(c.name == "unknown" for c in hand), [c.name for c in hand])

    def test_confident_non_deck_catalog_still_wins_without_bank(self):
        # Real loadout change: bank has no candidate, catalog may name a non-deck card.
        reader = HandReader()
        hand = self._read(reader, (None, 0.0, 1.0), ("skeletons", 0.52, 0.08))
        self.assertTrue(all(c.name == "skeletons" for c in hand), [c.name for c in hand])


class TowerBarPhoneGeometryTests(unittest.TestCase):
    """enemy_right must cover the measured phone pink bar; enemy hue must include 148-151."""

    def test_enemy_right_box_covers_measured_slot(self):
        box = dict(Layout().tower_bars)["enemy_right"]
        x0, y0, x1, y1 = box
        # Measured A35 pink fragments: ref ~x 0.77-0.84, y 0.065-0.11 (above old 0.139 strip).
        self.assertLessEqual(y0, 0.078, box)
        self.assertGreaterEqual(y1, 0.16, box)
        self.assertLessEqual(x0, 0.64, box)
        self.assertGreaterEqual(x1, 0.85, box)

    def test_enemy_hue_covers_live_148_151(self):
        (low, high), _sat, _val = _TOWER_BAR_COLOUR["enemy"]
        self.assertLessEqual(low, 148)
        self.assertGreaterEqual(high, 151)

    def test_bar_pixels_sees_hue_150_pink_in_enemy_right(self):
        # Full reference canvas with a pink run inside the (widened) enemy_right box.
        layout = Layout()
        box = dict(layout.tower_bars)["enemy_right"]
        height, width = 633, 419
        hsv = numpy.zeros((height, width, 3), dtype=numpy.uint8)
        # hue=150, sat=160, val=180 across the measured strip
        x0, y0, x1, y1 = box
        # Place the bar where the live phone read it: y ~70-80, x ~330-350.
        cy0, cy1 = max(0, int(0.11 * height)), int(0.13 * height)
        cx0, cx1 = int(0.78 * width), int(0.84 * width)
        hsv[cy0:cy1, cx0:cx1] = (150, 160, 180)
        # _bar_pixels is a static method on Perception; name prefix picks "enemy" colour.
        lit = Perception._bar_pixels(hsv, "enemy_right", box)
        self.assertGreaterEqual(lit, 1, f"box={box} lit={lit}")

    def test_enemy_left_mirror_box_unchanged(self):
        box = dict(Layout().tower_bars)["enemy_left"]
        self.assertEqual(box, (0.20, 0.139, 0.36, 0.163))


class ClashMaxIdleGateTests(unittest.TestCase):
    """Frozen-pilot bug: urgency 0.55 + elixir 4 must NOT hard-hold at 9999."""

    def test_no_play_still_hard_holds(self):
        self.assertEqual(
            clash_max_idle(
                action_name="wait",
                strat="save_elixir",
                phase="in_battle",
                threat_urgency=0.0,
                elixir=3,
            ),
            9999.0,
        )

    def test_calm_adapter_wait_hard_holds(self):
        self.assertEqual(
            clash_max_idle(
                action_name="wait",
                strat="cycle",
                phase="in_battle",
                threat_urgency=0.30,
                elixir=4,
            ),
            9999.0,
        )

    def test_pressure_wait_is_finite_not_frozen(self):
        # The live freeze case: urgency 0.55, elixir 4, adapter returning wait.
        self.assertEqual(
            clash_max_idle(
                action_name="wait",
                strat="defend_left",
                phase="in_battle",
                threat_urgency=0.55,
                elixir=4,
            ),
            2.50,
        )

    def test_high_urgency_wait_also_finite(self):
        self.assertEqual(
            clash_max_idle(
                action_name="wait",
                strat="defend_right",
                phase="in_battle",
                threat_urgency=0.90,
                elixir=7,
            ),
            2.50,
        )

    def test_high_urgency_below_floor_hard_holds(self):
        # Elixir < 4: adapter wait means nothing affordable — never force into a toast.
        self.assertEqual(
            clash_max_idle(
                action_name="wait",
                strat="defend_right",
                phase="in_battle",
                threat_urgency=0.90,
                elixir=2,
            ),
            9999.0,
        )
        self.assertEqual(
            clash_max_idle(
                action_name="wait",
                strat="push_right",
                phase="in_battle",
                threat_urgency=0.55,
                elixir=3,
            ),
            9999.0,
        )

    def test_main_menu_and_game_over_short(self):
        self.assertEqual(
            clash_max_idle(
                action_name="wait", strat="", phase="main_menu", threat_urgency=0.0, elixir=0
            ),
            1.50,
        )
        self.assertEqual(
            clash_max_idle(
                action_name="wait", strat="", phase="game_over", threat_urgency=0.0, elixir=0
            ),
            1.00,
        )

    def test_deploy_action_not_affected_by_wait_gate(self):
        # Deploy is not a wait — gate never hard-holds; elixir>=5 → default path.
        self.assertEqual(
            clash_max_idle(
                action_name="deploy_card_left",
                strat="push_left",
                phase="in_battle",
                threat_urgency=0.0,
                elixir=6,
            ),
            2.20,
        )
        # Low elixir still gets the softer 5.5s floor when not a wait.
        self.assertEqual(
            clash_max_idle(
                action_name="deploy_card_left",
                strat="push_left",
                phase="in_battle",
                threat_urgency=0.0,
                elixir=3,
            ),
            5.50,
        )


class BlindForceElixirGateTests(unittest.TestCase):
    """Last-resort idle promotion must never tap a card slot below the cost floor."""

    def test_low_elixir_stays_wait(self):
        for e in (0, 1, 2, 3):
            self.assertEqual(
                clash_blind_force_action(
                    strat="defend_right",
                    phase="in_battle",
                    threat_urgency=0.9,
                    nearest_threat=object(),
                    elixir=e,
                    total_actions=10,
                ),
                "wait",
                f"elixir={e}",
            )

    def test_save_elixir_never_forces(self):
        for strat in ("save_elixir", "hold_elixir_for_threat"):
            self.assertEqual(
                clash_blind_force_action(
                    strat=strat,
                    phase="in_battle",
                    threat_urgency=0.9,
                    nearest_threat=object(),
                    elixir=8,
                    total_actions=1,
                ),
                "wait",
            )

    def test_threat_above_floor_deploys_center(self):
        self.assertEqual(
            clash_blind_force_action(
                strat="defend_right",
                phase="in_battle",
                threat_urgency=0.55,
                nearest_threat=object(),
                elixir=5,
                total_actions=3,
            ),
            "deploy_defense_center",
        )

    def test_calm_above_floor_alternates_lanes(self):
        self.assertEqual(
            clash_blind_force_action(
                strat="cycle",
                phase="in_battle",
                threat_urgency=0.1,
                nearest_threat=None,
                elixir=6,
                total_actions=4,
            ),
            "deploy_card_right",
        )
        self.assertEqual(
            clash_blind_force_action(
                strat="cycle",
                phase="in_battle",
                threat_urgency=0.1,
                nearest_threat=None,
                elixir=6,
                total_actions=5,
            ),
            "deploy_card_left",
        )

    def test_menu_phases_not_battle_deploys(self):
        self.assertEqual(
            clash_blind_force_action(
                strat="", phase="main_menu", threat_urgency=0.0,
                nearest_threat=None, elixir=5, total_actions=0,
            ),
            "start_battle",
        )
        self.assertEqual(
            clash_blind_force_action(
                strat="", phase="game_over", threat_urgency=0.0,
                nearest_threat=None, elixir=5, total_actions=0,
            ),
            "confirm_ok",
        )


class LaneUrgencyPressureTests(unittest.TestCase):
    """Lane-derived urgency floor: bridge/on-half enemies must raise threat_score."""

    def test_bridge_enemy_lifts_urgency_to_at_least_055(self):
        from adapters.clash_adapter import ClashBattleAdapter
        import os

        adapter = ClashBattleAdapter()
        state = make_state(
            elixir=6,
            left=LaneView(enemy_at_bridge=1),
            right=LaneView(),
            hand=[HandCard(0, "knight", True), HandCard(1, "goblins", True)],
        )
        old = os.environ.pop("TYPESAFE_API_KEY", None)
        try:
            with patch.object(
                adapter, "extract_state_from_frame", return_value=state
            ), patch.object(
                adapter.tactical_policy, "evaluate_strategy", return_value="defend_left"
            ), patch.object(
                adapter.tactical_policy,
                "select_card",
                return_value=HandCard(0, "knight", True),
            ), patch.object(
                adapter.tactical_policy,
                "select_square",
                return_value=None,
            ), patch.object(
                adapter, "_observe"
            ):
                decision = adapter.decide(
                    numpy.zeros((633, 419, 3), dtype=numpy.uint8), threat_urgency=0.0
                )
        finally:
            if old is not None:
                os.environ["TYPESAFE_API_KEY"] = old
        self.assertGreaterEqual(float(decision.get("threat_score", 0.0)), 0.55, decision)
        self.assertEqual(decision.get("source"), "tactical_reflex")
        self.assertEqual(decision.get("strategy"), "defend_left")

    def test_cloud_rejects_low_conf_when_not_immediate(self):
        from adapters.clash_adapter import ClashBattleAdapter
        import os

        adapter = ClashBattleAdapter()
        state = make_state(
            elixir=6,
            left=LaneView(),
            right=LaneView(),
            hand=[HandCard(0, "knight", True)],
        )

        class FakeAns:
            choice = "push_left"
            probabilities = {"push_left": 0.30}

        class FakeResult:
            choices = {"strategy": FakeAns()}

        class FakeClient:
            def system_one(self, *a, **k):
                return FakeResult()

        with patch.object(
            adapter, "extract_state_from_frame", return_value=state
        ), patch.object(
            adapter.tactical_policy, "evaluate_strategy", return_value="cycle"
        ), patch.object(
            adapter.tactical_policy,
            "select_card",
            return_value=HandCard(0, "knight", True),
        ), patch.object(
            adapter.tactical_policy,
            "select_square",
            return_value=None,
        ), patch.object(
            adapter, "_observe"
        ), patch.object(
            adapter, "jev_client", FakeClient()
        ), patch.dict(
            os.environ, {"TYPESAFE_API_KEY": "x"}
        ):
            decision = adapter.decide(
                numpy.zeros((633, 419, 3), dtype=numpy.uint8), threat_urgency=0.0
            )
        # immediate_contact false (urgency 0, no lane enemies) → cloud called → conf 0.30 < 0.55 rejected
        self.assertEqual(decision.get("source"), "tactical_reflex")
        self.assertEqual(decision.get("strategy"), "cycle")


if __name__ == "__main__":
    unittest.main()
