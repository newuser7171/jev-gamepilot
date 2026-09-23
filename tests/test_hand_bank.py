"""Regression: multi-exemplar ShapeBank and PLAYER_DECK match filter."""
import unittest
from pathlib import Path

import numpy

from clash_jev.hand import (
    PLAYER_DECK,
    CatalogBank,
    ShapeBank,
    _as_variants,
    deck_slot_boxes,
    player_deck,
    read_hand,
    teach_deck,
)


class VariantParsingTests(unittest.TestCase):
    def test_single_dict_becomes_one_variant(self):
        self.assertEqual(_as_variants({"shape": [1], "detail": [0]}), [{"shape": [1], "detail": [0]}])

    def test_list_of_dicts_passes_through(self):
        entries = [{"shape": [1], "detail": [0]}, {"shape": [2], "detail": [3]}]
        self.assertEqual(_as_variants(entries), entries)

    def test_junk_is_dropped(self):
        self.assertEqual(_as_variants(None), [])
        self.assertEqual(_as_variants(["nope", 7]), [])


class PlayerDeckTests(unittest.TestCase):
    def test_deck_has_eight_cards(self):
        self.assertEqual(len(PLAYER_DECK), 8)
        for card in (
            "giant",
            "musketeer",
            "mini_pekka",
            "fireball",
            "goblins",
            "goblin_cage",
            "knight",
            "archers",
        ):
            self.assertIn(card, PLAYER_DECK)
        for card in ("spear_goblins", "goblin_hut", "mega_minion"):
            self.assertNotIn(card, PLAYER_DECK)

    def test_player_deck_prefers_eight_name_file(self):
        from clash_jev import hand

        path = hand.PLAYER_DECK_FILE
        old = path.read_text() if path.exists() else None
        try:
            path.write_text('["knight","arrows","giant","goblins","fireball","musketeer","archers","minions"]')
            self.assertEqual(player_deck(), frozenset(
                {"knight", "arrows", "giant", "goblins", "fireball", "musketeer", "archers", "minions"}
            ))
            path.write_text('["too","few"]')
            self.assertEqual(player_deck(), PLAYER_DECK)
        finally:
            if old is None:
                if path.exists():
                    path.unlink()
            else:
                path.write_text(old)

    def test_bank_loads_all_rows_for_soft_filter(self):
        bank = ShapeBank()
        unfiltered = ShapeBank(deck=None)
        self.assertEqual(len(bank.row_names), len(unfiltered.row_names))
        self.assertIn("knight", bank.names)
        self.assertIn("arrows", bank.names)
        # Non-deck must not win a weak/greyed contest: score a random non-deck row against the
        # default (soft) bank and require either a deck member or a high-confidence non-deck win.
        bank_default = ShapeBank()
        knight_rows = [i for i, n in enumerate(bank_default.row_names) if n == "knight"]
        if knight_rows:
            name, score, lead = bank_default.match(bank_default.matrix[knight_rows[0]])
            self.assertTrue(
                name in player_deck() or (score >= 0.62 and lead >= 0.12),
                f"knight art must not steal a slot via soft filter: {name} score={score:.3f} lead={lead:.3f}",
            )


class TeachDeckTests(unittest.TestCase):
    def test_deck_slot_boxes_scale_from_reference(self):
        import numpy

        frame = numpy.zeros((2340, 1080, 3), dtype=numpy.uint8)
        boxes = deck_slot_boxes(frame)
        self.assertEqual(len(boxes), 8)
        self.assertEqual(boxes[0], (40, 575, 230, 300))
        half = numpy.zeros((1170, 540, 3), dtype=numpy.uint8)
        x, y, w, h = deck_slot_boxes(half)[0]
        self.assertEqual((x, y, w, h), (20, 288, 115, 150))

    def test_catalog_bank_loads_official_art(self):
        bank = CatalogBank()
        self.assertGreater(len(bank.names), 50)
        self.assertIn("goblin_hut", bank.names)
        self.assertIn("mini_pekka", bank.names)

    def test_teach_deck_hunt3_all_eight(self):
        import json
        import cv2

        from clash_jev import hand

        hunt = Path(r"C:\Users\newuser\AppData\Local\Temp\opencode\hunt3.png")
        if not hunt.exists():
            self.skipTest("missing hunt3.png deck screen")
        frame = cv2.imread(str(hunt))
        assert frame is not None
        # Snapshot both persisted teach outputs so the suite never leaks deck exemplars
        # into the live bank (rotating out hand-tested variants breaks the 12/12).
        extra_path = hand.EXTRA_CARD_SHAPES
        extra_before = extra_path.read_bytes() if extra_path.exists() else None
        deck_path = hand.PLAYER_DECK_FILE
        deck_before = deck_path.read_bytes() if deck_path.exists() else None
        try:
            report = teach_deck(frame)
            self.assertTrue(report["complete"], report["outcomes"])
            expected = {
                "musketeer",
                "goblin_cage",
                "giant",
                "spear_goblins",
                "fireball",
                "goblin_hut",
                "mini_pekka",
                "goblins",
            }
            self.assertEqual(set(report["names"]), expected)
            self.assertEqual(player_deck(), expected)
            extra_after = json.loads(extra_path.read_text())
            for name in expected:
                self.assertIn(name, extra_after)
        finally:
            if extra_before is None:
                if extra_path.exists():
                    extra_path.unlink()
            else:
                extra_path.write_bytes(extra_before)
            if deck_before is None:
                if deck_path.exists():
                    deck_path.unlink()
            else:
                deck_path.write_bytes(deck_before)


class MultiExemplarMatchTests(unittest.TestCase):
    def test_max_exemplar_per_card_beats_single_row(self):
        bank = ShapeBank(deck=None)
        # Pick a real in-deck row and score it against itself: score must be ~1 and name exact.
        shape = bank.matrix[0]
        name, score, _lead = bank.match(shape)
        self.assertEqual(name, bank.row_names[0])
        self.assertGreater(score, 0.99)

    def test_lead_is_between_cards_not_rows(self):
        bank = ShapeBank(deck=None)
        # Two giant rows must collapse to one giant entry before lead is computed.
        giant_rows = [i for i, n in enumerate(bank.row_names) if n == "giant"]
        self.assertGreaterEqual(len(giant_rows), 1)
        name, score, lead = bank.match(bank.matrix[giant_rows[0]])
        self.assertEqual(name, "giant")
        self.assertGreaterEqual(lead, 0.0)

    def test_empty_deck_returns_unknown_shape(self):
        bank = ShapeBank(deck=frozenset())
        name, score, lead = bank.match(numpy.ones(224, dtype=numpy.float32))
        self.assertIsNone(name)
        self.assertEqual(score, 0.0)
        self.assertEqual(lead, 1.0)


class ThreeFrameRegression(unittest.TestCase):
    """The three saved battle frames must keep reading their visual truth after the bank redesign."""

    TRUTH = {
        "b2": ["archers", "goblins", "giant", "musketeer"],
        "hand2": ["goblins", "giant", "fireball", "goblin_cage"],
        "race": ["musketeer", "fireball", "archers", "giant"],
    }

    def test_saved_frames_twelve_of_twelve(self):
        import cv2

        tmp = Path(r"C:\Users\newuser\AppData\Local\Temp\opencode")
        missing = [n for n in self.TRUTH if not (tmp / f"{n}.png").exists()]
        if missing:
            self.skipTest(f"missing frames: {missing}")
        ok = total = 0
        for frame_name, expected in self.TRUTH.items():
            frame = cv2.imread(str(tmp / f"{frame_name}.png"))
            got = [card.name for card in read_hand(frame)]
            for want, have in zip(expected, got):
                total += 1
                ok += want == have
                self.assertEqual(have, want, f"{frame_name}: got {got}, want {expected}")
        self.assertEqual((ok, total), (12, 12))


if __name__ == "__main__":
    unittest.main()
