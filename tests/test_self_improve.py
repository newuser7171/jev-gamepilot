"""Closed-loop self-improvement: param grid, UCB bandit, journal, canary teach, tempo offsets."""
import json
import tempfile
import unittest
from pathlib import Path

from adapters.clash_adapter import TacticalReflexPolicy
from clash_jev.learn import (
    AUTO_TEACH_STREAK,
    MIN_GAMES_TO_TUNE,
    REJECT_WIN_RATE,
    SelfImprover,
    TuneParams,
    crowns_from_towers,
    outcome_from_crowns,
)
from clash_jev.state import BattleState, HandCard, Towers


def make_state(elixir=5, elapsed_s=60.0, towers=None):
    return BattleState(
        elapsed_s=elapsed_s,
        elixir=elixir,
        hand=(HandCard(0, "knight", True),),
        towers=towers or Towers(),
    )


class TuneParamsGridTests(unittest.TestCase):
    def test_defaults_match_hand_tuned(self):
        p = TuneParams()
        self.assertEqual(p.save_offset, 0)
        self.assertEqual(p.push_offset, 0)
        self.assertEqual(p.enemy_gate_at, 7.0)
        self.assertEqual(p.key(), "+0|+0|7.0")

    def test_neighbours_move_one_axis(self):
        ns = TuneParams().neighbours()
        keys = {n.key() for n in ns}
        self.assertIn("-1|+0|7.0", keys)
        self.assertIn("+1|+0|7.0", keys)
        self.assertIn("+0|-1|7.0", keys)
        self.assertIn("+0|+1|7.0", keys)
        self.assertIn("+0|+0|6.0", keys)
        self.assertIn("+0|+0|8.0", keys)
        self.assertEqual(len(ns), 6)

    def test_from_dict_clamps_to_grid(self):
        p = TuneParams.from_dict({"save_offset": 99, "push_offset": -99, "enemy_gate_at": 12.3})
        self.assertEqual(p.save_offset, 1)
        self.assertEqual(p.push_offset, -1)
        self.assertEqual(p.enemy_gate_at, 8.0)

    def test_from_dict_junk_is_defaults(self):
        self.assertEqual(TuneParams.from_dict(None), TuneParams())
        self.assertEqual(TuneParams.from_dict("nope"), TuneParams())


class BanditRetuneTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.learner = SelfImprover(
            journal_path=root / "journal.jsonl",
            params_path=root / "params.json",
        )
        self.learner.note_battle_start()

    def tearDown(self):
        self._tmp.cleanup()

    def _finish(self, outcome, n=1):
        for _ in range(n):
            if not self.learner.battle_open:
                self.learner.note_battle_start()
            self.learner.note_battle_end(outcome, (0, 0) if outcome != "win" else (3, 0), 120.0)

    def test_cold_cell_moves_after_min_games_losses(self):
        start = self.learner.params.key()
        self._finish("loss", MIN_GAMES_TO_TUNE)
        self.assertNotEqual(self.learner.params.key(), start, "cold cell must adopt a neighbour")

    def test_hot_cell_stays(self):
        start = self.learner.params.key()
        self._finish("win", MIN_GAMES_TO_TUNE + 2)
        self.assertEqual(self.learner.params.key(), start)

    def test_middling_does_not_thrash(self):
        start = self.learner.params.key()
        for _ in range(MIN_GAMES_TO_TUNE):
            if not self.learner.battle_open:
                self.learner.note_battle_start()
            self.learner.note_battle_end("draw", (0, 0), 90.0)
        # score == 0.5 → between reject and hold → hold cell
        self.assertEqual(self.learner.params.key(), start)

    def test_aborted_does_not_count(self):
        start = self.learner.params.key()
        for _ in range(MIN_GAMES_TO_TUNE + 3):
            if not self.learner.battle_open:
                self.learner.note_battle_start()
            self.learner.note_battle_end("aborted", (0, 0), 10.0)
        self.assertEqual(self.learner.params.key(), start)
        self.assertEqual(self.learner.games_on_cell, 0)

    def test_second_end_is_noop(self):
        rec = self.learner.note_battle_end("win", (1, 0), 60.0)
        self.assertIsNotNone(rec)
        again = self.learner.note_battle_end("loss", (0, 1), 60.0)
        self.assertIsNone(again)

    def test_journal_roundtrip(self):
        self.learner.note_battle_end("win", (3, 1), 180.0, behind=True)
        rows = [
            json.loads(line)
            for line in self.learner.journal_path.read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["outcome"], "win")
        self.assertEqual(rows[0]["crowns"], [3, 1])
        self.assertTrue(rows[0]["behind"])
        reloaded = SelfImprover(
            journal_path=self.learner.journal_path,
            params_path=self.learner.params_path,
        )
        self.assertEqual(reloaded.cells[self.learner.params.key()].games, 1)


class CrownOutcomeTests(unittest.TestCase):
    def test_crowns_from_towers(self):
        t = Towers(enemy_left=0.0, enemy_right=0.5, my_left=0.0, my_right=1.0)
        self.assertEqual(crowns_from_towers(t), (1, 1))
        self.assertEqual(outcome_from_crowns((1, 1)), "draw")
        self.assertEqual(outcome_from_crowns((3, 0)), "win")
        self.assertEqual(outcome_from_crowns((0, 2)), "loss")

    def test_none_towers_are_standing(self):
        self.assertEqual(crowns_from_towers(Towers()), (0, 0))


class AutoTeachStreakTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.learner = SelfImprover(
            journal_path=root / "j.jsonl",
            params_path=root / "p.json",
        )
        self.learner.note_battle_start()
        self.fired = []

    def tearDown(self):
        self._tmp.cleanup()

    def _observe(self, score=0.7, lead=0.3, name="unknown", catalog="knight", deck=True):
        return self.learner.observe_hand(
            0,
            name,
            catalog,
            score,
            lead,
            in_player_deck=deck,
            teach_fn=lambda n: self.fired.append(n) or "added",
        )

    def test_teach_fires_on_streak(self):
        for _ in range(AUTO_TEACH_STREAK - 1):
            self.assertIsNone(self._observe())
        self.assertEqual(self._observe(), "added")
        self.assertEqual(self.fired, ["knight"])

    def test_low_score_never_streaks(self):
        for _ in range(10):
            self.assertIsNone(self._observe(score=0.49))
        self.assertEqual(self.fired, [])

    def test_low_lead_never_streaks(self):
        for _ in range(10):
            self.assertIsNone(self._observe(lead=0.10))
        self.assertEqual(self.fired, [])

    def test_not_in_deck_never_streaks(self):
        for _ in range(10):
            self.assertIsNone(self._observe(deck=False))
        self.assertEqual(self.fired, [])

    def test_named_slot_clears_streak(self):
        self.assertIsNone(self._observe())
        self.assertIsNone(self._observe(name="knight"))
        # restart streak from 1
        self.assertIsNone(self._observe())
        self.assertIsNone(self._observe())
        self.assertEqual(self._observe(), "added")


class TempoOffsetTests(unittest.TestCase):
    def test_no_tune_keeps_class_thresholds(self):
        self.assertEqual(TacticalReflexPolicy.tempo_thresholds(make_state()), (5, 8))
        self.assertEqual(TacticalReflexPolicy.tempo_thresholds(make_state(elapsed_s=150)), (3, 6))
        self.assertEqual(TacticalReflexPolicy.tempo_thresholds(make_state(elapsed_s=300)), (2, 5))

    def test_tune_offsets_apply(self):
        tune = TuneParams(save_offset=1, push_offset=-1)
        save, push = TacticalReflexPolicy.tempo_thresholds(make_state(), tune)
        self.assertEqual(save, 6)
        self.assertEqual(push, 7)
        self.assertLess(save, push)

    def test_offsets_never_invert_save_and_push(self):
        # worst case: save +1, push -1 on triple → save 3, push must stay > save
        tune = TuneParams(save_offset=1, push_offset=-1)
        save, push = TacticalReflexPolicy.tempo_thresholds(make_state(elapsed_s=300), tune)
        self.assertEqual(save, 3)
        self.assertGreaterEqual(push, save + 1)

    def test_default_policy_gate_unchanged(self):
        policy = TacticalReflexPolicy()
        # gate needs enemy_e >= 7 and elixir <= enemy_e - 3 → 8.5 / 5 fires
        state = BattleState(
            elapsed_s=60.0,
            elixir=5,
            hand=(HandCard(0, "knight", True),),
            enemy_elixir_estimate=8.5,
        )
        strat = policy.evaluate_strategy(state)
        self.assertIn(strat, ("save_elixir", "cycle"))
        self.assertTrue(policy._gate_fired)

    def test_tuned_gate_8_allows_7_5(self):
        policy = TacticalReflexPolicy(tune=TuneParams(enemy_gate_at=8.0))
        state = BattleState(
            elapsed_s=60.0,
            elixir=4,
            hand=(HandCard(0, "knight", True),),
            enemy_elixir_estimate=7.5,
        )
        policy.evaluate_strategy(state)
        self.assertFalse(policy._gate_fired)

    def test_tuned_gate_still_fires_at_8_5(self):
        policy = TacticalReflexPolicy(tune=TuneParams(enemy_gate_at=8.0))
        state = BattleState(
            elapsed_s=60.0,
            elixir=5,
            hand=(HandCard(0, "knight", True),),
            enemy_elixir_estimate=8.5,
        )
        strat = policy.evaluate_strategy(state)
        self.assertTrue(policy._gate_fired)
        self.assertIn(strat, ("save_elixir", "cycle"))


class CanaryTeachTests(unittest.TestCase):
    def test_verify_returns_list(self):
        from clash_jev.learn import verify_hand_canaries

        fails = verify_hand_canaries()
        self.assertIsInstance(fails, list)

    def test_canary_truth_matches_hand_bank(self):
        from clash_jev.learn import CANARY_TRUTH
        from tests.test_hand_bank import ThreeFrameRegression

        self.assertEqual(CANARY_TRUTH, ThreeFrameRegression.TRUTH)

    def test_teach_rolls_back_on_canary_fail(self):
        """If canaries report failure after add_card, the bank is restored."""
        import clash_jev.learn as learn_mod
        from clash_jev.hand import EXTRA_CARD_SHAPES

        before = EXTRA_CARD_SHAPES.read_bytes() if EXTRA_CARD_SHAPES.exists() else None
        original_verify = learn_mod.verify_hand_canaries
        learn_mod.verify_hand_canaries = lambda: ["b2: broke"]
        try:
            # tiny synthetic frame — add_card will likely no-op/fail, still must not leave broken canaries
            import numpy as np

            frame = np.zeros((2340, 1080, 3), dtype=np.uint8)
            outcome = learn_mod.teach_with_canaries(frame, 0, "knight")
            # either rejected or duplicate/no-op; bank must match pre-teach
            after = EXTRA_CARD_SHAPES.read_bytes() if EXTRA_CARD_SHAPES.exists() else None
            self.assertEqual(before, after)
            self.assertIn(outcome, ("rejected", "duplicate", "added", "replaced"))
        finally:
            learn_mod.verify_hand_canaries = original_verify


class RejectWinRateBoundaryTests(unittest.TestCase):
    def test_reject_is_below_hold(self):
        self.assertLess(REJECT_WIN_RATE, 0.55)


if __name__ == "__main__":
    unittest.main()
