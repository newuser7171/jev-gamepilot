"""Journal artifact filter: sub-10s end-screen rows never teach style/tune (item 5)."""
import json
import tempfile
import unittest
from pathlib import Path

from clash_jev.learn import SelfImprover, is_played_game, MIN_REAL_BATTLE_S


def _row(outcome, elapsed=None, pushes=5):
    row = {"outcome": outcome, "pushes": pushes, "defends": 3, "params": {}}
    if elapsed is not None:
        row["elapsed_s"] = elapsed
    return row


class IsPlayedGameTests(unittest.TestCase):
    def test_real_battle_counts(self):
        self.assertTrue(is_played_game(_row("win", 150.0)))
        self.assertTrue(is_played_game(_row("loss", MIN_REAL_BATTLE_S)))

    def test_artifacts_dropped(self):
        self.assertFalse(is_played_game(_row("win", 4.0)))
        self.assertFalse(is_played_game(_row("win", 9.9)))
        self.assertFalse(is_played_game(_row("loss", 0.0)))

    def test_non_decided_and_malformed_dropped(self):
        self.assertFalse(is_played_game(_row("aborted", 150.0)))
        self.assertFalse(is_played_game(_row("win", "soon")))
        self.assertFalse(is_played_game(_row("win", {})))

    def test_legacy_row_without_elapsed_keeps_counting(self):
        self.assertTrue(is_played_game(_row("win")))


class TuneCellsFilterTests(unittest.TestCase):
    def test_artifact_wins_do_not_inflate_cell_stats(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            journal = root / "journal.jsonl"
            rows = [_row("win", 150.0) for _ in range(2)]
            rows += [_row("win", 5.0, pushes=0) for _ in range(3)]
            journal.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            learner = SelfImprover(
                journal_path=journal, params_path=root / "params.json"
            )
            self.assertEqual(len(learner.cells), 1)
            cell = next(iter(learner.cells.values()))
            self.assertEqual(cell.games, 2)
            self.assertEqual(cell.wins, 2)


class DistilFilterTests(unittest.TestCase):
    def test_all_artifact_journal_yields_no_style(self):
        from clash_jev.playstyle import distil_from_journal

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "battle_journal.jsonl"
            rows = [_row("win", 5.0) for _ in range(6)]
            rows += [_row("loss", 5.0) for _ in range(4)]
            path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            self.assertIsNone(distil_from_journal(path, min_games=5))

    def test_real_games_still_distil(self):
        from clash_jev.playstyle import distil_from_journal

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "battle_journal.jsonl"
            rows = [_row("win", 150.0, pushes=20) for _ in range(6)]
            rows += [_row("loss", 150.0, pushes=4) for _ in range(4)]
            # artifacts mixed in must not change the counts
            rows += [_row("win", 4.0) for _ in range(5)]
            path.write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
            )
            style = distil_from_journal(path, style_id="filter_test", min_games=5)
            self.assertIsNotNone(style)
            self.assertIn("10 games", style.source)  # 6W+4L, artifacts excluded


if __name__ == "__main__":
    unittest.main()
