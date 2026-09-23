"""Closed-loop self-improvement: journal battles, tune tempo within hard bounds, safe hand teaches.

Design constraints:
- One file, no ML deps. Adjustments are discrete steps over a tiny integer/float grid.
- Every change is reversible: journal records the params used for each battle.
- Canaries (saved regression frames) must keep matching after any bank mutation.
- Defaults match the hand-tuned policy so behaviour is unchanged until enough battles land.

Files (next to this module):
    battle_journal.jsonl   append-only; one JSON object per finished battle
    tuned_params.json      current TuneParams; absent = defaults
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

LEARN_DIR = Path(__file__).resolve().parent
JOURNAL_PATH = LEARN_DIR / "battle_journal.jsonl"
PARAMS_PATH = LEARN_DIR / "tuned_params.json"

# Discrete grids — movement only ever steps to an adjacent cell.
SAVE_OFFSETS: tuple[int, ...] = (-1, 0, 1)
PUSH_OFFSETS: tuple[int, ...] = (-1, 0, 1)
ENEMY_GATES: tuple[float, ...] = (6.0, 7.0, 8.0)

# After this many battles on the current cell, evaluate; never thrash mid-match.
MIN_GAMES_TO_TUNE = 5
# Promote a neighbour only when the current cell is clearly cold.
REJECT_WIN_RATE = 0.34
# Once a cell is hot, lock it for a full window (exploit).
HOLD_WIN_RATE = 0.55
# UCB exploration bonus denominator.
_UCB_C = 0.45
# Journal cap (oldest dropped on rewrite).
MAX_JOURNAL = 400
# Decision streak before a confident catalog hit is auto-taught live.
AUTO_TEACH_STREAK = 3
AUTO_TEACH_CATALOG_SCORE = 0.50
AUTO_TEACH_CATALOG_LEAD = 0.15


@dataclass(frozen=True)
class TuneParams:
    """Bounded knobs the tactical policy reads every frame."""

    save_offset: int = 0          # added to (save_below, push_at).save_below
    push_offset: int = 0          # added to push_at
    enemy_gate_at: float = 7.0    # opponent-elixir push gate threshold

    def key(self) -> str:
        return f"{self.save_offset:+d}|{self.push_offset:+d}|{self.enemy_gate_at:.1f}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "TuneParams":
        if not isinstance(raw, dict):
            return cls()
        save = _nearest(SAVE_OFFSETS, raw.get("save_offset", 0), int)
        push = _nearest(PUSH_OFFSETS, raw.get("push_offset", 0), int)
        gate = _nearest(ENEMY_GATES, raw.get("enemy_gate_at", 7.0), float)
        return cls(save_offset=save, push_offset=push, enemy_gate_at=gate)

    def neighbours(self) -> list["TuneParams"]:
        """Adjacent cells on the grid (one coordinate moves one step)."""
        out: list[TuneParams] = []
        si, pi = SAVE_OFFSETS.index(self.save_offset), PUSH_OFFSETS.index(self.push_offset)
        gi = ENEMY_GATES.index(self.enemy_gate_at)
        if si > 0:
            out.append(TuneParams(SAVE_OFFSETS[si - 1], self.push_offset, self.enemy_gate_at))
        if si < len(SAVE_OFFSETS) - 1:
            out.append(TuneParams(SAVE_OFFSETS[si + 1], self.push_offset, self.enemy_gate_at))
        if pi > 0:
            out.append(TuneParams(self.save_offset, PUSH_OFFSETS[pi - 1], self.enemy_gate_at))
        if pi < len(PUSH_OFFSETS) - 1:
            out.append(TuneParams(self.save_offset, PUSH_OFFSETS[pi + 1], self.enemy_gate_at))
        if gi > 0:
            out.append(TuneParams(self.save_offset, self.push_offset, ENEMY_GATES[gi - 1]))
        if gi < len(ENEMY_GATES) - 1:
            out.append(TuneParams(self.save_offset, self.push_offset, ENEMY_GATES[gi + 1]))
        return out


def _nearest(grid: Iterable, value: Any, cast):
    try:
        v = cast(value)
    except (TypeError, ValueError):
        return grid[1] if len(grid) == 3 else grid[0]
    best = min(grid, key=lambda g: abs(float(g) - float(v)))
    return cast(best)


@dataclass
class CellStats:
    games: int = 0
    wins: int = 0
    draws: int = 0

    @property
    def score(self) -> float:
        """Draws count half. Empty cell scores 0 so UCB explores it first."""
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games

    def as_dict(self) -> dict[str, int]:
        return {"games": self.games, "wins": self.wins, "draws": self.draws}


@dataclass
class BattleRecord:
    ts: float
    outcome: str                    # "win" | "loss" | "draw" | "aborted"
    crowns: tuple[int, int]         # (mine, theirs)
    elapsed_s: float
    params: dict[str, Any]
    strategy_counts: dict[str, int] = field(default_factory=dict)
    elixir_avg: float = 0.0
    deploys: int = 0
    defends: int = 0
    pushes: int = 0
    gate_blocks: int = 0
    behind: bool = False
    auto_taught: list[str] = field(default_factory=list)

    def as_json(self) -> str:
        d = asdict(self)
        d["crowns"] = list(self.crowns)
        return json.dumps(d, separators=(",", ":"))


class SelfImprover:
    """Owns the param cell, UCB stats, journal, and live auto-teach streaks."""

    def __init__(
        self,
        journal_path: Path = JOURNAL_PATH,
        params_path: Path = PARAMS_PATH,
        enabled: bool = True,
    ):
        self.journal_path = Path(journal_path)
        self.params_path = Path(params_path)
        self.enabled = enabled
        self.params = self._load_params()
        self.cells: dict[str, CellStats] = self._load_cells_from_journal()
        # Per-battle accumulators (reset on note_battle_start / battle end).
        self._strategy_counts: dict[str, int] = {}
        self._elixir_sum = 0.0
        self._elixir_n = 0
        self._deploys = 0
        self._defends = 0
        self._pushes = 0
        self._gate_blocks = 0
        self._auto_taught: list[str] = []
        self._streak: dict[int, tuple[str, int]] = {}  # slot -> (name, consecutive)
        self.battle_open = False
        self.current_key = self.params.key()
        self.games_on_cell = 0

    # ── persistence ──────────────────────────────────────────────

    def _load_params(self) -> TuneParams:
        if not self.enabled or not self.params_path.exists():
            return TuneParams()
        try:
            return TuneParams.from_dict(json.loads(self.params_path.read_text()))
        except (OSError, json.JSONDecodeError):
            return TuneParams()

    def _save_params(self) -> None:
        if not self.enabled:
            return
        try:
            self.params_path.write_text(json.dumps(self.params.as_dict(), indent=1))
        except OSError:
            pass

    def _load_cells_from_journal(self) -> dict[str, CellStats]:
        cells: dict[str, CellStats] = {}
        if not self.journal_path.exists():
            return cells
        try:
            lines = self.journal_path.read_text().splitlines()
        except OSError:
            return cells
        for line in lines[-MAX_JOURNAL:]:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("outcome") not in ("win", "loss", "draw"):
                continue
            p = TuneParams.from_dict(row.get("params"))
            key = p.key()
            cell = cells.setdefault(key, CellStats())
            cell.games += 1
            if row["outcome"] == "win":
                cell.wins += 1
            elif row["outcome"] == "draw":
                cell.draws += 1
        return cells

    # ── battle lifecycle ─────────────────────────────────────────

    def note_battle_start(self) -> None:
        self._strategy_counts.clear()
        self._elixir_sum = self._elixir_n = 0
        self._deploys = self._defends = self._pushes = 0
        self._gate_blocks = 0
        self._auto_taught = []
        self._streak = {}
        self.current_key = self.params.key()
        self.battle_open = True

    def observe_decision(
        self,
        *,
        strategy: str,
        elixir: int,
        action: str,
        enemy_gate_fired: bool = False,
        behind: bool = False,
    ) -> None:
        """Called once per adapter.decide() while a battle is open."""
        if not self.battle_open:
            return
        self._strategy_counts[strategy] = self._strategy_counts.get(strategy, 0) + 1
        self._elixir_sum += elixir
        self._elixir_n += 1
        if action == "deploy_clash_card":
            self._deploys += 1
            if strategy.startswith("defend"):
                self._defends += 1
            if strategy.startswith("push") or strategy.startswith("counter") or strategy == "build_push":
                self._pushes += 1
        if enemy_gate_fired:
            self._gate_blocks += 1
        if behind:
            # sticky for the record
            self._strategy_counts["__behind"] = 1

    def note_battle_end(
        self,
        outcome: str,
        crowns: tuple[int, int],
        elapsed_s: float,
        behind: bool = False,
    ) -> Optional[BattleRecord]:
        """Journal the battle, maybe retune. Safe to call once; second call is a no-op."""
        if not self.battle_open:
            return None
        self.battle_open = False
        if outcome not in ("win", "loss", "draw"):
            outcome = "aborted"
        elixir_avg = (self._elixir_sum / self._elixir_n) if self._elixir_n else 0.0
        counts = {k: v for k, v in self._strategy_counts.items() if not k.startswith("__")}
        rec = BattleRecord(
            ts=time.time(),
            outcome=outcome,
            crowns=crowns,
            elapsed_s=round(elapsed_s, 1),
            params=self.params.as_dict(),
            strategy_counts=counts,
            elixir_avg=round(elixir_avg, 2),
            deploys=self._deploys,
            defends=self._defends,
            pushes=self._pushes,
            gate_blocks=self._gate_blocks,
            behind=behind or bool(self._strategy_counts.get("__behind")),
            auto_taught=list(self._auto_taught),
        )
        if self.enabled:
            self._append_journal(rec)
            if outcome in ("win", "loss", "draw"):
                self._account_and_maybe_retune(outcome)
        return rec

    # ── tuning ───────────────────────────────────────────────────

    def _append_journal(self, rec: BattleRecord) -> None:
        try:
            with self.journal_path.open("a", encoding="utf-8") as fh:
                fh.write(rec.as_json() + "\n")
        except OSError:
            pass
        # Cap: rewrite when oversized (rare).
        try:
            if self.journal_path.exists() and self.journal_path.stat().st_size > 512_000:
                lines = [ln for ln in self.journal_path.read_text().splitlines() if ln.strip()]
                self.journal_path.write_text("\n".join(lines[-MAX_JOURNAL:]) + "\n")
        except OSError:
            pass

    def _account_and_maybe_retune(self, outcome: str) -> None:
        key = self.current_key
        cell = self.cells.setdefault(key, CellStats())
        cell.games += 1
        if outcome == "win":
            cell.wins += 1
        elif outcome == "draw":
            cell.draws += 1
        self.games_on_cell += 1
        if self.games_on_cell < MIN_GAMES_TO_TUNE:
            return
        wr = cell.score
        if wr >= HOLD_WIN_RATE:
            # Hot — stay, but keep counting on the same cell.
            self.games_on_cell = 0
            return
        if wr > REJECT_WIN_RATE:
            # Middling — give it more games before moving.
            self.games_on_cell = MIN_GAMES_TO_TUNE  # re-check next battle
            return
        # Cold cell: move to the neighbour with the best UCB1 score.
        best = self._best_neighbour()
        if best is None or best.key() == key:
            self.games_on_cell = 0
            return
        self._adopt(best)

    def _best_neighbour(self) -> Optional[TuneParams]:
        import math

        total = sum(c.games for c in self.cells.values()) or 1
        best: Optional[TuneParams] = None
        best_ucb = -1.0
        for n in self.params.neighbours():
            cell = self.cells.get(n.key(), CellStats())
            if cell.games == 0:
                return n  # untried — explore immediately
            exploitation = cell.score
            exploration = _UCB_C * math.sqrt(math.log(total + 1) / cell.games)
            ucb = exploitation + exploration
            if ucb > best_ucb:
                best_ucb, best = ucb, n
        return best

    def _adopt(self, params: TuneParams) -> None:
        self.params = params
        self.current_key = params.key()
        self.games_on_cell = 0
        self.cells.setdefault(self.current_key, CellStats())
        self._save_params()

    def force_params(self, params: TuneParams) -> None:
        """Tests / CLI: pin a cell without waiting for the bandit."""
        self._adopt(params)

    # ── live auto-teach (hand bank) ──────────────────────────────

    def observe_hand(
        self,
        slot: int,
        name: str,
        catalog_name: Optional[str],
        catalog_score: float,
        catalog_lead: float,
        in_player_deck: bool,
        teach_fn,
    ) -> Optional[str]:
        """Streak a confident catalog id on an unknown slot; teach once it holds.

        teach_fn(name) should call the canary-guarded teach and return its outcome.
        Returns the teach outcome on the frame the teach fires, else None.
        """
        if not self.enabled or name != "unknown":
            self._streak.pop(slot, None)
            return None
        if not catalog_name or not in_player_deck:
            self._streak.pop(slot, None)
            return None
        if catalog_score < AUTO_TEACH_CATALOG_SCORE or catalog_lead < AUTO_TEACH_CATALOG_LEAD:
            self._streak.pop(slot, None)
            return None
        prev_name, count = self._streak.get(slot, ("", 0))
        count = count + 1 if prev_name == catalog_name else 1
        self._streak[slot] = (catalog_name, count)
        if count < AUTO_TEACH_STREAK:
            return None
        outcome = teach_fn(catalog_name)
        self._streak[slot] = (catalog_name, 0)  # don't re-fire every frame
        if outcome in ("added", "replaced"):
            self._auto_taught.append(f"slot{slot}:{catalog_name}:{outcome}")
        return outcome

    # ── introspection ────────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "params": self.params.as_dict(),
            "key": self.params.key(),
            "games_on_cell": self.games_on_cell,
            "cells": {k: v.as_dict() for k, v in sorted(self.cells.items())},
            "journal": str(self.journal_path),
        }


def crowns_from_towers(towers) -> tuple[int, int]:
    """(mine, theirs) from tower HP: 0.0 = destroyed. None counts as standing."""
    def down(*vals: Optional[float]) -> int:
        return sum(1 for v in vals if v is not None and float(v) <= 0.0)

    mine = down(towers.enemy_left, towers.enemy_right, towers.enemy_king)
    theirs = down(towers.my_left, towers.my_right, towers.my_king)
    return mine, theirs


def outcome_from_crowns(crowns: tuple[int, int]) -> str:
    mine, theirs = crowns
    if mine > theirs:
        return "win"
    if mine < theirs:
        return "loss"
    return "draw"


def crowns_from_results_frame(frame_bgr) -> Optional[tuple[int, int]]:
    """Best-effort read of the post-game crown banners. None when not that screen."""
    if frame_bgr is None:
        return None
    try:
        from clash_jev.publish import _results_crowns

        return _results_crowns(frame_bgr)
    except Exception:
        return None


# Saved battle frames used as canaries: a teach that flips any of these is rejected
# and the extra bank is restored (the musketeer rotate-out regression).
CANARY_TRUTH: dict[str, list[str]] = {
    "b2": ["archers", "goblins", "giant", "musketeer"],
    "hand2": ["goblins", "giant", "fireball", "goblin_cage"],
    "race": ["musketeer", "fireball", "archers", "giant"],
}


def _canary_dir() -> Path:
    import os

    return Path(os.environ.get("TEMP", "/tmp")) / "opencode"


def verify_hand_canaries() -> list[str]:
    """Return human-readable failures for every canary frame that exists on disk.

    Missing frames are skipped (CI without captures still runs); present frames
    must still read their full expected hand after a bank mutation.
    """
    import cv2

    from clash_jev.hand import read_hand

    fails: list[str] = []
    root = _canary_dir()
    for stem, expected in CANARY_TRUTH.items():
        path = root / f"{stem}.png"
        if not path.exists():
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        got = [card.name for card in read_hand(frame)]
        if got != expected:
            fails.append(f"{stem}: got {got}, want {expected}")
    return fails


def teach_with_canaries(frame, slot: int, name: str) -> str:
    """add_card, then prove the regression canaries still hold — else roll back.

    Returns add_card's outcome, or "rejected" when a canary broke.
    """
    from clash_jev.hand import EXTRA_CARD_SHAPES, add_card

    backup = EXTRA_CARD_SHAPES.read_bytes() if EXTRA_CARD_SHAPES.exists() else None
    outcome = add_card(frame, slot, name)
    if outcome == "duplicate":
        return outcome
    fails = verify_hand_canaries()
    if not fails:
        return outcome
    if backup is None:
        if EXTRA_CARD_SHAPES.exists():
            EXTRA_CARD_SHAPES.unlink()
    else:
        EXTRA_CARD_SHAPES.write_bytes(backup)
    return "rejected"
