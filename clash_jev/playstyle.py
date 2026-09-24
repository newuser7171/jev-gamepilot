"""Playstyle: clone how another player (or archetype) plays.

A style is a small set of biases the tactical policy reads every frame —
tempo offsets, defend aggression, counter willingness, cycle preference,
spell eagerness, opening tempo, preferred lane. Load a named style, save a
custom one, or distil a style from battle_journal outcomes.

    style = load_style("hog_cycle")
    policy.style = style
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

STYLE_DIR = Path(__file__).resolve().parent / "styles"
ACTIVE_STYLE_FILE = Path(__file__).resolve().parent / "active_style.json"


@dataclass(frozen=True)
class PlayStyle:
    """Biases applied on top of the tactical reflex defaults."""

    id: str = "default"
    name: str = "Default"
    # Tempo: positive = push sooner / save less (aggressive); negative = patient.
    save_offset: int = 0
    push_offset: int = 0
    # 0.0 = pure reactive defence; 1.0 = always look to counter / convert.
    counter_willingness: float = 0.5
    # 0.0 = never prefer cheap cycle; 1.0 = cycle hard when above floor.
    cycle_bias: float = 0.3
    # 0.0 = spells only on multi-body value; 1.0 = chip / snipe more freely.
    spell_eagerness: float = 0.0
    # Opening: seconds of patience before first push attempt (0 = take river fast).
    opening_patience_s: float = 6.0
    # Preferred attack lane: "left" | "right" | "balanced".
    lane_bias: str = "balanced"
    # 0.0 = hold elixir for defence; 1.0 = dump into pushes at lower thresholds.
    aggression: float = 0.5
    # Free-form note / provenance (player name, URL, date).
    source: str = ""
    # Optional deck this style was recorded with (8 card ids); informational.
    deck: tuple[str, ...] = field(default_factory=tuple)

    def key(self) -> str:
        return (
            f"{self.id}|{self.save_offset:+d}|{self.push_offset:+d}|"
            f"{self.counter_willingness:.2f}|{self.cycle_bias:.2f}|"
            f"{self.spell_eagerness:.2f}|{self.opening_patience_s:.1f}|"
            f"{self.lane_bias}|{self.aggression:.2f}"
        )

    def as_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["deck"] = list(self.deck)
        return raw

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PlayStyle":
        if not isinstance(raw, dict):
            return cls()
        deck = raw.get("deck") or ()
        if isinstance(deck, list):
            deck = tuple(str(x) for x in deck)
        elif not isinstance(deck, tuple):
            deck = ()
        return cls(
            id=str(raw.get("id") or "custom"),
            name=str(raw.get("name") or raw.get("id") or "Custom"),
            save_offset=int(raw.get("save_offset", 0)),
            push_offset=int(raw.get("push_offset", 0)),
            counter_willingness=_clampf(raw.get("counter_willingness", 0.5), 0.0, 1.0),
            cycle_bias=_clampf(raw.get("cycle_bias", 0.3), 0.0, 1.0),
            spell_eagerness=_clampf(raw.get("spell_eagerness", 0.0), 0.0, 1.0),
            opening_patience_s=_clampf(raw.get("opening_patience_s", 6.0), 0.0, 30.0),
            lane_bias=str(raw.get("lane_bias") or "balanced"),
            aggression=_clampf(raw.get("aggression", 0.5), 0.0, 1.0),
            source=str(raw.get("source") or ""),
            deck=deck,
        )


def _clampf(v: Any, lo: float, hi: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return lo
    return max(lo, min(hi, x))


# Archetypes distilled from how top-ladder players pilot each family.
# Not a specific pro's name — a reusable clone of that play pattern.
BUILTIN_STYLES: dict[str, PlayStyle] = {
    "default": PlayStyle(id="default", name="Default", source="hand-tuned baseline"),
    "hog_cycle": PlayStyle(
        id="hog_cycle",
        name="Hog 2.6 Cycle",
        save_offset=0,
        push_offset=-1,
        counter_willingness=0.75,
        cycle_bias=0.9,
        spell_eagerness=0.2,
        opening_patience_s=4.0,
        aggression=0.7,
        source="top-ladder hog 2.6 pattern",
        deck=("hog", "ice_golem", "musketeer", "cannon", "fireball", "the_log", "skeletons", "ice_spirit"),
    ),
    "golem_beatdown": PlayStyle(
        id="golem_beatdown",
        name="Golem Beatdown",
        save_offset=1,
        push_offset=1,
        counter_willingness=0.35,
        cycle_bias=0.1,
        spell_eagerness=0.1,
        opening_patience_s=12.0,
        lane_bias="left",
        aggression=0.55,
        source="beatdown: build from back, ignore small chip",
        deck=("golem", "night_witch", "baby_dragon", "tornado", "lightning", "mega_minion", "elixir_collector", "the_log"),
    ),
    "logbait": PlayStyle(
        id="logbait",
        name="Log Bait",
        save_offset=0,
        push_offset=-1,
        counter_willingness=0.65,
        cycle_bias=0.7,
        spell_eagerness=0.35,
        opening_patience_s=5.0,
        aggression=0.65,
        source="bait: force spell out, punish with barrel",
        deck=("goblin_barrel", "princess", "goblin_gang", "knight", "rocket", "the_log", "dart_goblin", "inferno_tower"),
    ),
    "bridge_spam": PlayStyle(
        id="bridge_spam",
        name="Bridge Spam",
        save_offset=0,
        push_offset=-1,
        counter_willingness=0.8,
        cycle_bias=0.4,
        spell_eagerness=0.15,
        opening_patience_s=3.0,
        aggression=0.85,
        source="constant dual-lane pressure",
        deck=("battle_ram", "bandit", "minions", "electro_wizard", "poison", "royal_ghost", "dark_prince", "fireball"),
    ),
    "control": PlayStyle(
        id="control",
        name="Control",
        save_offset=1,
        push_offset=0,
        counter_willingness=0.9,
        cycle_bias=0.4,
        spell_eagerness=0.1,
        opening_patience_s=8.0,
        aggression=0.35,
        source="positive trades, convert defence",
        deck=("mortar", "miner", "archers", "knight", "the_log", "fireball", "skeletons", "bomb_tower"),
    ),
    "megaknight": PlayStyle(
        id="megaknight",
        name="Mega Knight Punish",
        save_offset=0,
        push_offset=0,
        counter_willingness=0.85,
        cycle_bias=0.35,
        spell_eagerness=0.2,
        opening_patience_s=5.0,
        aggression=0.7,
        source="drop MK on pushes, punish overcommit",
        deck=("mega_knight", "miner", "bats", "wall_breakers", "the_log", "fireball", "dark_prince", "electro_spirit"),
    ),
}


def list_styles() -> list[PlayStyle]:
    """Built-ins plus any JSON files under clash_jev/styles/."""
    out = dict(BUILTIN_STYLES)
    if STYLE_DIR.is_dir():
        for path in sorted(STYLE_DIR.glob("*.json")):
            try:
                style = PlayStyle.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
            out[style.id] = style
    return list(out.values())


def get_style(style_id: str) -> Optional[PlayStyle]:
    if style_id in BUILTIN_STYLES:
        return BUILTIN_STYLES[style_id]
    path = STYLE_DIR / f"{style_id}.json"
    if path.is_file():
        try:
            return PlayStyle.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None
    # Allow full path / bare filename outside the dir
    p = Path(style_id)
    if p.is_file():
        try:
            return PlayStyle.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def save_style(style: PlayStyle) -> Path:
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    path = STYLE_DIR / f"{style.id}.json"
    path.write_text(json.dumps(style.as_dict(), indent=2), encoding="utf-8")
    return path


def set_active(style_id: str) -> PlayStyle:
    """Persist the style phone_pilot loads on next start."""
    style = get_style(style_id)
    if style is None:
        raise KeyError(f"unknown style: {style_id}")
    ACTIVE_STYLE_FILE.write_text(
        json.dumps({"id": style.id, "style": style.as_dict()}, indent=2),
        encoding="utf-8",
    )
    return style


def load_active(default_id: str = "default") -> PlayStyle:
    """Active style for this machine (ACTIVE_STYLE_FILE), else builtin default."""
    if ACTIVE_STYLE_FILE.is_file():
        try:
            raw = json.loads(ACTIVE_STYLE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = None
        if isinstance(raw, dict):
            if isinstance(raw.get("style"), dict):
                return PlayStyle.from_dict(raw["style"])
            if raw.get("id"):
                found = get_style(str(raw["id"]))
                if found is not None:
                    return found
    return BUILTIN_STYLES.get(default_id, BUILTIN_STYLES["default"])


def clone_player(
    name: str,
    *,
    deck: Optional[list[str] | tuple[str, ...]] = None,
    base: str = "default",
    write_deck: bool = True,
    **overrides: Any,
) -> PlayStyle:
    """Create (and save) a named clone: copy `base` and layer overrides.

    Example:
        clone_player("my_ladder", deck=["giant","musketeer",...], aggression=0.8)
    When `deck` has 8 card ids and write_deck, it lands in player_deck.json
    so hand reads and the tactical policy share the same loadout.
    """
    root = get_style(base) or BUILTIN_STYLES["default"]
    kwargs: dict[str, Any] = {}
    for key in (
        "save_offset",
        "push_offset",
        "counter_willingness",
        "cycle_bias",
        "spell_eagerness",
        "opening_patience_s",
        "lane_bias",
        "aggression",
        "source",
    ):
        if key in overrides and overrides[key] is not None:
            kwargs[key] = overrides[key]
    if deck is not None:
        kwargs["deck"] = tuple(deck)
    style = replace(
        root,
        id=_slug(name),
        name=name,
        source=overrides.get("source") or f"cloned from {root.id}",
        **kwargs,
    )
    save_style(style)
    if write_deck and "deck" in kwargs and len(kwargs["deck"]) == 8:
        from clash_jev.hand import PLAYER_DECK_FILE

        PLAYER_DECK_FILE.write_text(
            json.dumps(sorted(str(n) for n in kwargs["deck"]), separators=(",", ":")),
            encoding="utf-8",
        )
    return style


def _slug(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip().lower())
    return out.strip("_") or "custom"


def distil_from_journal(
    journal_path: Optional[Path] = None,
    *,
    style_id: str = "learned",
    min_games: int = 5,
) -> Optional[PlayStyle]:
    """Build a style from recorded wins: more pushes on wins → more aggressive clone.

    Reads clash_jev/battle_journal.jsonl (same file SelfImprover writes).
    Returns None when there is not enough finished data.
    """
    from clash_jev.learn import JOURNAL_PATH

    path = journal_path or JOURNAL_PATH
    if not path.is_file():
        return None
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("outcome") in ("win", "loss", "draw"):
            rows.append(row)
    if len(rows) < min_games:
        return None
    wins = [r for r in rows if r.get("outcome") == "win"]
    losses = [r for r in rows if r.get("outcome") == "loss"]
    if not wins:
        return None

    def avg_push(rs: list[dict]) -> float:
        return sum(float(r.get("pushes") or 0) for r in rs) / len(rs)

    def avg_defend(rs: list[dict]) -> float:
        return sum(float(r.get("defends") or 0) for r in rs) / len(rs)

    w_push, w_def = avg_push(wins), avg_defend(wins)
    l_push, l_def = (avg_push(losses), avg_defend(losses)) if losses else (w_push, w_def)
    # Wins with more pushes than losses → reward aggression.
    aggression = 0.5
    if losses:
        delta = (w_push - l_push) - (w_def - l_def)
        aggression = _clampf(0.5 + delta * 0.02, 0.1, 0.9)
    push_offset = 0 if w_push <= l_push + 1 else (-1 if w_push > l_push + 8 else 0)
    save_offset = 0 if w_def <= l_def + 2 else 1
    style = PlayStyle(
        id=style_id,
        name=f"Learned ({len(wins)}W/{len(losses)}L)",
        save_offset=save_offset,
        push_offset=push_offset,
        counter_willingness=_clampf(0.4 + aggression * 0.4, 0.0, 1.0),
        cycle_bias=0.4,
        aggression=aggression,
        opening_patience_s=6.0 - aggression * 3.0,
        source=f"distilled from {path.name} ({len(rows)} games)",
    )
    save_style(style)
    return style
