"""Hand reading: which four cards you hold, whether each is playable, and which card comes next.

A card's art is matched against a bank of known cards (`card_shapes.json`, plus cards added with
`clash-jev add-card`). All positions are on the 419x633 reference layout.
"""

import itertools
import json
from pathlib import Path

import cv2
import numpy

from clash_jev.screenmap import ScreenMap
from clash_jev.state import HandCard

CARD_SHAPES = Path(__file__).with_name("card_shapes.json")
EXTRA_CARD_SHAPES = Path(__file__).with_name("extra_card_shapes.json")  # written by `add-card` / teach-deck
# Official royaleapi art vectors (shape + detail + colour) — identifies cards never added by hand.
CATALOG_SHAPES = Path(__file__).with_name("catalog_shapes.json")
# Battle-deck loadout written by `teach-deck`; falls back to PLAYER_DECK below.
PLAYER_DECK_FILE = Path(__file__).with_name("player_deck.json")
# Centre of each hand card, as fractions of the frame.
HAND_TAP_POINTS = ((0.339, 0.886), (0.501, 0.889), (0.649, 0.886), (0.814, 0.889))
# Top-left corner of each hand card and the size of a card, in reference pixels.
SLOTS = ((115, 529), (182, 529), (249, 529), (316, 529))
CARD_WIDTH, CARD_HEIGHT = 54, 66
# The "Next:" thumbnail is a half-size hand card, at this box on the reference layout (x, y, w, h).
_NEXT_BOX = (52, 591, 25, 31)

# Minimum shape-match score in the hand and as the Next thumbnail, and minimum lead over the runner-up.
# Live deck art varies by level/rarity frame — thresholds accept a clear winner without a huge margin.
_HAND_MATCH = 0.42
_NEXT_MATCH = 0.62
_MATCH_MARGIN = 0.08
# A banner ("No card selected") can cover the top of a card. The lower half alone tells cards apart
# less surely, so it is only used when the whole picture is not decisive, and has to be near-perfect.
_LOWER_HALF_MATCH = 0.70
_LOWER_HALF_MARGIN = 0.18
_THUMB = (14, 16)  # width, height of a shape thumbnail
# Fine-detail match, tried first. The right card scores 0.73+ lit, greyed or washed out. A card enlarged
# while it is being dragged scores near 0 here and is left to the shape match.
_DETAIL_SIZE = (28, 32)
_DETAIL_SIGMA = 2.0
_DETAIL_MATCH = 0.38
_EMPTY_TEXTURE = 12.0
_EMPTY_SATURATION = 150.0
# A playable card is in full colour top to bottom: lit cards measure >= 39 mean saturation in their
# weakest band, greyed ones 0. Requiring every band keeps a half-filled card from reading as ready.
_LIT_SATURATION = 15.0
_BANDS = ((4, 18), (18, 32), (32, 46))
_ART = (
    0.15,
    0.09,
    0.85,
    0.73,
)  # the picture inside a card, as fractions of the card: no frame, no cost badge
# The eight cards in this profile's battle deck (default). Matching prefers these so greyed non-deck
# base art (knight false-positives on washed frames) does not win — but a confident non-deck read
# (loadout changed, friendly battle) is accepted instead of forced to unknown.
# `teach-deck` overwrites PLAYER_DECK_FILE; player_deck() prefers that file when it has eight names.
PLAYER_DECK = frozenset(
    {
        "giant",
        "musketeer",
        "mini_pekka",
        "knight",
        "archers",
        "fireball",
        "goblins",
        "goblin_cage",
    }
)
# Non-deck only wins when it is clearly the picture, not a greyed near-miss.
_NON_DECK_SCORE = 0.62
_NON_DECK_LEAD = 0.12
# Official catalog (full-card art) vs battle-deck crops: blend of grey shape and colour.
# Thresholds accept a clear top-1 on the deck screen without needing a large lead.
_CATALOG_SCORE = 0.30
_CATALOG_LEAD = 0.002
_CATALOG_SHAPE_WEIGHT = 0.55
_CATALOG_COLOR_WEIGHT = 0.45
_CATALOG_COLOR_SIZE = (16, 16)
# Catalog fallback on a hand slot: the deck screen threshold is tuned for large crops, so a
# hand-sized crop needs a clearer winner. In-deck names accept lower (art already taught via
# teach-deck lives in the catalog); non-deck names need a confident read (loadout changed).
_CATALOG_HAND_IN_DECK_SCORE = 0.35
_CATALOG_HAND_IN_DECK_LEAD = 0.005
_CATALOG_HAND_ANY_SCORE = 0.45
_CATALOG_HAND_ANY_LEAD = 0.02
# In-deck bank hit with a solid score: greyed/washed hand art keeps the runner-up close, so a
# thin lead must not dump a known deck card into unknown / catalog false-accept (golem, …).
_HAND_IN_DECK_SCORE = 0.40
# Any in-deck bank candidate at least this strong blocks a catalog *non-deck* override.
_IN_DECK_NEAR = 0.30
# Battle-deck grid on a 1080x2340 portrait frame (hunt3.png), scaled to any frame size.
_DECK_REF_SIZE = (1080, 2340)
_DECK_XS = (40, 300, 560, 820)
_DECK_YS = (575, 997)
_DECK_SLOT_SIZE = (230, 300)
# Live frames of the same card disagree by level frame / grey wipe. Keep a few exemplars per name;
# match takes the best row per card, never lets one late teach erase the others.
_MAX_VARIANTS = 4
_VARIANT_MIN_LEAD = 0.97  # cosine against stored variants: higher than this is a duplicate


def _as_variants(entry: object) -> list[dict]:
    if isinstance(entry, list):
        return [item for item in entry if isinstance(item, dict)]
    if isinstance(entry, dict):
        return [entry]
    return []


def player_deck() -> frozenset[str]:
    """Active loadout: player_deck.json from `teach-deck` when valid, else PLAYER_DECK."""
    if PLAYER_DECK_FILE.exists():
        try:
            names = json.loads(PLAYER_DECK_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            names = None
        if isinstance(names, list) and len(names) == 8 and all(isinstance(n, str) and n for n in names):
            return frozenset(names)
    return PLAYER_DECK


def _unit(v: numpy.ndarray) -> numpy.ndarray:
    x = v.astype(numpy.float32).ravel()
    x = x - x.mean()
    norm = numpy.linalg.norm(x)
    return x / norm if norm else x


def _unit_rows(matrix: numpy.ndarray) -> numpy.ndarray:
    x = matrix.astype(numpy.float32)
    x = x - x.mean(axis=1, keepdims=True)
    norm = numpy.linalg.norm(x, axis=1, keepdims=True)
    return x / numpy.where(norm < 1e-9, 1.0, norm)


def _color_vec(image: numpy.ndarray, box: tuple[int, int, int, int]) -> numpy.ndarray:
    """Small colour thumbnail of the card art, L2-normalised (catalog colour channel)."""
    x, y, width, height = box
    art = image[
        y + round(_ART[1] * height) : y + round(_ART[3] * height),
        x + round(_ART[0] * width) : x + round(_ART[2] * width),
    ]
    if art.size == 0:
        art = image
    thumb = cv2.resize(art, _CATALOG_COLOR_SIZE, interpolation=cv2.INTER_AREA)
    flat = thumb.astype(numpy.float32).ravel()
    norm = numpy.linalg.norm(flat)
    return flat / norm if norm else flat


class CatalogBank:
    """Official card art (shape + colour). Used to name battle-deck slots never taught by hand."""

    def __init__(self) -> None:
        self.names: list[str] = []
        self.shapes = numpy.zeros((0, _THUMB[0] * _THUMB[1]), dtype=numpy.float32)
        self.colors = numpy.zeros((0, _CATALOG_COLOR_SIZE[0] * _CATALOG_COLOR_SIZE[1] * 3), dtype=numpy.float32)
        if CATALOG_SHAPES.exists():
            raw = json.loads(CATALOG_SHAPES.read_text())
            names: list[str] = []
            shapes: list[numpy.ndarray] = []
            colors: list[numpy.ndarray] = []
            for name, entry in raw.items():
                if not isinstance(entry, dict) or "shape" not in entry or "color" not in entry:
                    continue
                names.append(name)
                shapes.append(numpy.asarray(entry["shape"], dtype=numpy.float32))
                colors.append(numpy.asarray(entry["color"], dtype=numpy.float32))
            if shapes:
                self.names = names
                self.shapes = _unit_rows(numpy.stack(shapes))
                self.colors = _unit_rows(numpy.stack(colors))

    def match(self, image: numpy.ndarray, box: tuple[int, int, int, int]) -> tuple[str | None, float, float]:
        """(best card, blended score, lead over runner-up) from official art vs a full-card crop."""
        if not self.names:
            return None, 0.0, 1.0
        gray = self.shapes @ _unit(_shape(image, box))
        colour = self.colors @ _unit(_color_vec(image, box))
        blend = _CATALOG_SHAPE_WEIGHT * gray + _CATALOG_COLOR_WEIGHT * colour
        order = numpy.argsort(-blend)
        best = int(order[0])
        score = float(blend[best])
        lead = score - float(blend[int(order[1])]) if len(order) > 1 else score
        if score < _CATALOG_SCORE or lead < _CATALOG_LEAD:
            return None, score, lead
        return self.names[best], score, lead


def deck_slot_boxes(frame: numpy.ndarray) -> list[tuple[int, int, int, int]]:
    """Eight battle-deck card boxes on `frame`, scaled from the 1080x2340 reference grid."""
    height, width = frame.shape[:2]
    ref_w, ref_h = _DECK_REF_SIZE
    sx, sy = width / ref_w, height / ref_h
    slot_w, slot_h = _DECK_SLOT_SIZE
    boxes = []
    for y in _DECK_YS:
        for x in _DECK_XS:
            boxes.append(
                (
                    round(x * sx),
                    round(y * sy),
                    round(slot_w * sx),
                    round(slot_h * sy),
                )
            )
    return boxes


def _remember(name: str, shape: list[float], detail: list[float]) -> str:
    """Append one exemplar under `name`; near-duplicates ignored; rotate past _MAX_VARIANTS."""
    saved = json.loads(EXTRA_CARD_SHAPES.read_text()) if EXTRA_CARD_SHAPES.exists() else {}
    variants = _as_variants(saved.get(name))
    vector = numpy.asarray(shape, dtype=numpy.float32)
    for existing in variants:
        prior = numpy.asarray(existing.get("shape", []), dtype=numpy.float32)
        if prior.shape == vector.shape and float(numpy.dot(prior, vector)) >= _VARIANT_MIN_LEAD:
            return "duplicate"
    variants.append({"shape": shape, "detail": detail})
    outcome = "added"
    if len(variants) > _MAX_VARIANTS:
        variants = variants[-_MAX_VARIANTS:]
        outcome = "replaced"
    saved[name] = variants
    EXTRA_CARD_SHAPES.write_text(json.dumps(saved, separators=(",", ":")))
    return outcome


def teach_deck(frame: numpy.ndarray) -> dict[str, object]:
    """Name and learn all eight battle-deck cards from a deck-screen frame; persist the loadout.

    Bank match first (taught cards), official catalog second (brand-new cards). Writes
    player_deck.json only when every slot has a name, and appends exemplars to the extra bank.
    Deck screen is plain UI — match on the raw frame, not the battle ScreenMap warp.
    """
    boxes = deck_slot_boxes(frame)
    catalog = CatalogBank()
    bank = ShapeBank(deck=None)
    names: list[str | None] = []
    outcomes: list[str] = []
    for box in boxes:
        x, y, width, height = box
        crop = frame[y : y + height, x : x + width]
        if crop.size == 0:
            names.append(None)
            outcomes.append("empty")
            continue
        shape = [round(float(v), 4) for v in _shape(frame, box)]
        detail = [round(float(v), 4) for v in _detail(frame, box)]
        name, score, lead = bank.match(numpy.asarray(shape, dtype=numpy.float32))
        source = "bank"
        if name is None or score < _HAND_MATCH or lead < _MATCH_MARGIN:
            cat_name, cat_score, cat_lead = catalog.match(frame, box)
            if cat_name:
                name, score, lead, source = cat_name, cat_score, cat_lead, "catalog"
        if name is None:
            names.append(None)
            outcomes.append(f"unknown score={score:.3f} lead={lead:.3f}")
            continue
        names.append(name)
        # Only persist brand-new cards (catalog ID). Bank hits already have hand-tested
        # exemplars — appending a deck-screen crop would rotate those out and break battle reads.
        if source == "catalog":
            outcome = _remember(name, shape, detail)
        else:
            outcome = "known"
        outcomes.append(f"{name} via {source} score={score:.3f} {outcome}")
    complete = len(names) == 8 and all(names)
    if complete:
        PLAYER_DECK_FILE.write_text(json.dumps(sorted(str(n) for n in names), separators=(",", ":")))
    return {
        "complete": complete,
        "names": names,
        "outcomes": outcomes,
        "player_deck": player_deck() if complete else player_deck(),
    }


def _shape(reference: numpy.ndarray, box: tuple[int, int, int, int]) -> numpy.ndarray:
    """A card's picture as a normalised greyscale thumbnail: identical lit or greyed, at any size."""
    x, y, width, height = box
    art = reference[
        y + round(_ART[1] * height) : y + round(_ART[3] * height),
        x + round(_ART[0] * width) : x + round(_ART[2] * width),
    ]
    thumb = cv2.resize(cv2.cvtColor(art, cv2.COLOR_BGR2GRAY), _THUMB, interpolation=cv2.INTER_AREA)
    thumb = thumb.astype(numpy.float32).ravel()
    thumb -= thumb.mean()
    norm = numpy.linalg.norm(thumb)
    return thumb / norm if norm else thumb


def _detail(reference: numpy.ndarray, box: tuple[int, int, int, int]) -> numpy.ndarray:
    """A card's picture reduced to its fine detail: every point compared with its own surroundings, so broad
    light and dark areas drop out. The wipe the game draws over a card you cannot afford is a broad area,
    so the lines of the drawing underneath it survive."""
    x, y, width, height = box
    art = reference[
        y + round(_ART[1] * height) : y + round(_ART[3] * height),
        x + round(_ART[0] * width) : x + round(_ART[2] * width),
    ]
    grey = cv2.resize(cv2.cvtColor(art, cv2.COLOR_BGR2GRAY), _DETAIL_SIZE, interpolation=cv2.INTER_AREA)
    grey = grey.astype(numpy.float32)
    surroundings = cv2.GaussianBlur(grey, (0, 0), _DETAIL_SIGMA)
    spread = numpy.sqrt(cv2.GaussianBlur((grey - surroundings) ** 2, (0, 0), _DETAIL_SIGMA)) + 4.0
    detail = ((grey - surroundings) / spread).ravel()
    detail -= detail.mean()
    norm = numpy.linalg.norm(detail)
    return detail / norm if norm else detail


def _lower_half(shape: numpy.ndarray) -> numpy.ndarray:
    half = shape.reshape(_THUMB[1], _THUMB[0])[_THUMB[1] // 2 :].ravel()
    half = half - half.mean()
    norm = numpy.linalg.norm(half)
    return half / norm if norm else half


def _slot_box(slot: int) -> tuple[int, int, int, int]:
    return (*SLOTS[slot], CARD_WIDTH, CARD_HEIGHT)


def _is_empty(reference: numpy.ndarray, slot: int) -> bool:
    """Between a play and the next card landing, a slot shows a flat, strongly blue placeholder.
    Texture / saturation: placeholder 7 / 239, greyed cards 14-20 / 0, lit cards 41-58 / 80-160."""
    x, y = SLOTS[slot]
    art = reference[y + 6 : y + 48, x + 8 : x + 46]
    flat = float(cv2.cvtColor(art, cv2.COLOR_BGR2GRAY).std()) < _EMPTY_TEXTURE
    return flat and float(cv2.cvtColor(art, cv2.COLOR_BGR2HSV)[..., 1].mean()) > _EMPTY_SATURATION


def _is_lit(reference: numpy.ndarray, slot: int) -> bool:
    x, y = SLOTS[slot]
    saturation = cv2.cvtColor(reference[y : y + 50, x + 8 : x + 46], cv2.COLOR_BGR2HSV)[..., 1]
    return min(float(saturation[top:bottom].mean()) for top, bottom in _BANDS) >= _LIT_SATURATION


# Historical ShapeBank API: deck=None is unfiltered; an empty frozenset returns unknown.
# The default (omitted) uses the active loadout from player_deck() / PLAYER_DECK.
_USE_ACTIVE_DECK: object = object()


class ShapeBank:
    """Known card shapes. Each name may hold several exemplar rows; a card's score is its best row."""

    def __init__(self, deck: frozenset[str] | None | object = _USE_ACTIVE_DECK):
        if deck is _USE_ACTIVE_DECK:
            deck = player_deck()
        shapes = json.loads(CARD_SHAPES.read_text())
        if EXTRA_CARD_SHAPES.exists():
            try:
                extra = json.loads(EXTRA_CARD_SHAPES.read_text())
            except json.JSONDecodeError:
                extra = {}  # a truncated teach write must not brick every HandReader
            if isinstance(extra, dict):
                for name, entry in extra.items():
                    shapes[name] = _as_variants(shapes.get(name)) + _as_variants(entry)
        self.deck = deck
        self.row_names: list[str] = []
        matrix_rows: list[numpy.ndarray] = []
        detail_rows: list[numpy.ndarray] = []
        for name, entry in shapes.items():
            for variant in _as_variants(entry):
                if "shape" not in variant or "detail" not in variant:
                    continue
                self.row_names.append(name)
                matrix_rows.append(numpy.asarray(variant["shape"], dtype=numpy.float32))
                detail_rows.append(numpy.asarray(variant["detail"], dtype=numpy.float32))
        if not matrix_rows and (deck is None or deck):
            raise ValueError(f"no usable card shapes in {CARD_SHAPES} / {EXTRA_CARD_SHAPES}")
        shape_dim = _THUMB[0] * _THUMB[1]
        detail_dim = _DETAIL_SIZE[0] * _DETAIL_SIZE[1]
        self.matrix = (
            numpy.stack(matrix_rows)
            if matrix_rows
            else numpy.zeros((0, shape_dim), dtype=numpy.float32)
        )
        self.lower = (
            numpy.stack([_lower_half(shape) for shape in self.matrix])
            if len(self.matrix)
            else numpy.zeros((0, shape_dim // 2), dtype=numpy.float32)
        )
        self.details = (
            numpy.stack(detail_rows)
            if detail_rows
            else numpy.zeros((0, detail_dim), dtype=numpy.float32)
        )

    @property
    def names(self) -> list[str]:
        """Unique card ids, first-seen order (compat with probes that print bank.names)."""
        return list(dict.fromkeys(self.row_names))

    def match(
        self, shape: numpy.ndarray, lower_half: bool = False, detail: bool = False
    ) -> tuple[str | None, float, float]:
        """(best card, its score, its lead over the runner-up). Score = max exemplar for that card.

        Deck members win ties and near-misses. A non-deck card only wins when its score and lead
        clear the non-deck bar — that accepts a real loadout change without letting washed-out
        knight/arrows art steal a greyed slot.
        """
        if detail:
            scores = self.details @ shape
        else:
            scores = self.lower @ _lower_half(shape) if lower_half else self.matrix @ shape
        per_card: dict[str, float] = {}
        for row, name in enumerate(self.row_names):
            value = float(scores[row])
            if name not in per_card or value > per_card[name]:
                per_card[name] = value
        if not per_card:
            return None, 0.0, 1.0
        if self.deck is not None and not self.deck:
            return None, 0.0, 1.0
        order = sorted(per_card, key=per_card.get, reverse=True)  # type: ignore[arg-type]
        best_name = order[0]
        best_score = per_card[best_name]
        second_score = per_card[order[1]] if len(order) > 1 else 0.0
        lead = best_score - second_score

        in_deck = lambda n: self.deck is None or n in self.deck  # noqa: E731
        if not in_deck(best_name):
            deck_order = [n for n in order if in_deck(n)]
            if deck_order:
                deck_name = deck_order[0]
                deck_score = per_card[deck_name]
                deck_lead = deck_score - (per_card[deck_order[1]] if len(deck_order) > 1 else 0.0)
                if best_score >= _NON_DECK_SCORE and lead >= _NON_DECK_LEAD:
                    return best_name, best_score, lead
                return deck_name, deck_score, deck_lead
        return best_name, best_score, lead


class HandReader:
    """Reads the four hand cards and the Next card. Every name comes from the card's picture, never from
    the deck order."""

    def __init__(self):
        self.bank = ShapeBank()
        self.catalog = CatalogBank()
        self.next_card: str | None = None

    def read(self, frame: numpy.ndarray) -> tuple[HandCard, ...]:
        reference = _reference(frame)
        deck = player_deck()
        hand = []
        for slot in range(4):
            if _is_empty(reference, slot):  # the slot is between cards: the new one has not landed yet
                hand.append(HandCard(slot, "unknown", False))
                continue
            name, score, lead = self.bank.match(_detail(reference, _slot_box(slot)), detail=True)
            known = score >= _DETAIL_MATCH and lead >= _MATCH_MARGIN
            # Solid in-deck detail hit: thin lead must not fall through to a wrong catalog name.
            if not known and name in deck and score >= _DETAIL_MATCH:
                known = True
            shape = _shape(reference, _slot_box(slot))
            bank_in_deck_near = name in deck and score >= _IN_DECK_NEAR
            if not known:
                name, score, lead = self.bank.match(shape)
                bank_in_deck_near = bank_in_deck_near or (name in deck and score >= _IN_DECK_NEAR)
                known = score >= _HAND_MATCH and lead >= _MATCH_MARGIN
                if not known and name in deck and score >= _HAND_IN_DECK_SCORE:
                    known = True  # deck card, greyed art: score clears in-deck bar, lead is thin
            if not known:
                name, score, lead = self.bank.match(shape, lower_half=True)
                bank_in_deck_near = bank_in_deck_near or (name in deck and score >= _IN_DECK_NEAR)
                known = score >= _LOWER_HALF_MATCH and lead >= _LOWER_HALF_MARGIN
                if not known and name in deck and score >= _LOWER_HALF_MATCH:
                    known = True
            if not known:
                # Shape bank only knows taught cards. Official catalog names a loadout card
                # that was never in player_deck / teach-deck when the crop is decisive.
                cat_name, cat_score, cat_lead = self.catalog.match(reference, _slot_box(slot))
                if cat_name is not None:
                    in_deck = cat_name in deck
                    if in_deck:
                        known = (
                            cat_score >= _CATALOG_HAND_IN_DECK_SCORE
                            and cat_lead >= _CATALOG_HAND_IN_DECK_LEAD
                        )
                    elif not bank_in_deck_near:
                        # No competent in-deck bank candidate: only then trust a confident
                        # non-deck catalog read (real loadout change).
                        known = (
                            cat_score >= _CATALOG_HAND_ANY_SCORE
                            and cat_lead >= _CATALOG_HAND_ANY_LEAD
                        )
                    if known:
                        name, score, lead = cat_name, cat_score, cat_lead
            hand.append(HandCard(slot, name if known else "unknown", _is_lit(reference, slot)))
        self.next_card = self._read_next(reference)
        return tuple(hand)

    def _read_next(self, reference: numpy.ndarray) -> str | None:
        """The thumbnail sits a pixel or two differently from frame to frame, so a few nearby boxes are
        tried and the best score wins."""
        x, y, width, _ = _NEXT_BOX
        best: tuple[float, str | None, float] = (0.0, None, 0.0)
        for dx, dy, dw in itertools.product((-1, 0, 1), (-2, 0, 2), (0, 1, 2)):
            box = (x + dx, y + dy, width + dw, round((width + dw) * CARD_HEIGHT / CARD_WIDTH))
            if box[1] + box[3] > reference.shape[0]:
                continue
            name, score, lead = self.bank.match(_shape(reference, box))
            if score > best[0]:
                best = (score, name, lead)
        score, name, lead = best
        return name if score >= _NEXT_MATCH and lead >= _MATCH_MARGIN else None


def read_hand(frame: numpy.ndarray) -> tuple[HandCard, ...]:
    """One-off read (snap, add-card checks)."""
    return HandReader().read(frame)


def _reference(frame: numpy.ndarray) -> numpy.ndarray:
    return ScreenMap.for_frame(frame).to_reference(frame)


def add_card(frame: numpy.ndarray, slot: int, name: str) -> str:
    """Teach a card from the art in `slot`. Appends an exemplar; near-duplicates are ignored.

    Returns "added", "duplicate", or "replaced" (oldest variant rotated out at the cap).
    """
    reference = _reference(frame)
    shape = [round(float(value), 4) for value in _shape(reference, _slot_box(slot))]
    detail = [round(float(value), 4) for value in _detail(reference, _slot_box(slot))]
    return _remember(name, shape, detail)
