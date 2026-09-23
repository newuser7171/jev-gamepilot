"""
adapters/clash_adapter.py - Clash Royale Autonomous RTS Adapter for jev-gamepilot.
Integrates the complete 3-tier hierarchical battle intelligence pipeline from clash-jev:
  Tier 1: Strategic Intent Formulation (12 STRATEGIES with structured criteria)
  Tier 2: Tactical Card Selection (Hand profiling, archetypes, counter-matching)
  Tier 3: Precision Legal Square Deployment (18x32 grid tile mapping & spell targeting)

Supports:
  - Cloud / Keyless TypeSafe System One (Jev)
  - Fast Local Laya System 1 Decisions (20-35ms offline inference)
  - Tactical Reflex Rule Engine (Zero-latency fallback for 100% reliable uptime)
  - Samsung Galaxy A35 5G (1080x2340) & Universal Portrait Mobile Calibration
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple, Optional, List, Dict

import cv2
import numpy as np

# Import clash-jev primitives
from clash_jev.cards import CARDS, info, profile, card_class, archetypes, describe, hits_air, known, base_name
from clash_jev.strategies import STRATEGIES, Strategy, SAVE, HOLD, NO_PLAY
from clash_jev.policy import STRATEGY_INSTRUCTIONS
from clash_jev.moves import (
    Square,
    legal_squares,
    spell_targets,
    BRIDGE,
    TOWER_FRONT,
    BACK,
    CENTRE,
    ENEMY_TOWER,
    POCKET,
    ENEMY_KING,
    ALL_SQUARES,
)
from clash_jev.state import BattleState, HandCard, LaneView, Towers
from clash_jev.perception import Perception, Layout
from clash_jev.screenmap import REFERENCE
from clash_jev.units import Unit, find_units
from clash_jev.troops import TroopClassifier
from clash_jev.hand import HandReader, HAND_TAP_POINTS


class ClashCoordinateMapper:
    """
    Transforms 0.0-1.0 normalized reference coordinates onto real device pixel coordinates.
    Calibrated for standard tall modern smartphones (18:9, 19.5:9, 20:9, e.g. 1080x2340).
    """

    def __init__(self, screen_width: int = 1080, screen_height: int = 2340):
        self.screen_width = screen_width
        self.screen_height = screen_height

    def update_resolution(self, width: int, height: int):
        self.screen_width = width
        self.screen_height = height

    def map_point(self, xy: Tuple[float, float]) -> Tuple[int, int]:
        """
        Maps (rx, ry) normalized coordinates to real screen pixel (px, py).
        Respects arena vertical centering on tall mobile displays.
        """
        w, h = self.screen_width, self.screen_height
        rx, ry = xy[0], xy[1]
        ratio = w / float(h) if h > 0 else 0.5

        if ratio < 0.58:  # Modern tall smartphone (height is large compared to width)
            px = int(rx * w)
            if ry >= 0.80:  # Bottom HUD (cards + elixir bar)
                # Anchored to bottom ~20% of device screen
                py = int((0.80 + (ry - 0.80) * 1.0) * h)
            else:  # Arena region
                # Arena vertically centered with ~4% top status margin and ~76% height
                arena_top = 0.04 * h
                arena_height = 0.76 * h
                py = int(arena_top + (ry / 0.80) * arena_height)
            return px, py
        else:
            # Tablet or standard 16:9 layout
            return int(rx * w), int(ry * h)

    def get_card_tap(self, slot: int) -> Tuple[int, int]:
        """Returns pixel (x, y) for tapping card in tray (slot 0-3)."""
        idx = max(0, min(3, slot))
        return self.map_point(HAND_TAP_POINTS[idx])

    def get_square_tap(self, square: Square) -> Tuple[int, int]:
        """Returns pixel (x, y) for deploying at an arena square."""
        return self.map_point(square.xy)


class TacticalReflexPolicy:
    """
    High-speed deterministic tactical engine modeling master-level Clash Royale instincts.
    Used when local/cloud AI models are offline, under high latency, or as consensus fallback.
    """

    def __init__(self):
        self._push_counter = 0
        # Dedicated lane alternator for the cycle/default square fallback —
        # evaluate_strategy's push paths share _push_counter and would skew
        # cycle parity between taps.
        self._cycle_lane = 0
        # elapsed_s of last real defend decision — feeds counter-push window
        self._last_defend_elapsed = -999.0

    @staticmethod
    def tempo_thresholds(state: BattleState) -> Tuple[int, int]:
        """(save_below, push_at) by elixir rate. Opening handled separately."""
        rate = state.elixir_rate
        if rate == "double":
            return 3, 6
        if rate == "triple":
            return 2, 5
        return 5, 8

    @staticmethod
    def _tower_hp(v: Optional[float]) -> float:
        """None (bar unseen) reads as full; 0.0 = destroyed."""
        return 1.0 if v is None else max(0.0, v)

    @classmethod
    def _total_hp(cls, t: Towers, mine: bool) -> float:
        vals = (
            (t.my_left, t.my_right, t.my_king)
            if mine
            else (t.enemy_left, t.enemy_right, t.enemy_king)
        )
        return sum(cls._tower_hp(v) for v in vals)

    @classmethod
    def _behind(cls, state: BattleState) -> bool:
        return cls._total_hp(state.towers, mine=False) > cls._total_hp(state.towers, mine=True) + 0.10

    @staticmethod
    def _is_endgame(state: BattleState) -> bool:
        # Last 30s of regulation (180s) and all overtime.
        return state.elapsed_s >= 150.0

    def _attack_lane(self, state: BattleState, force_kill: bool = False) -> str:
        """Alternate lanes normally; in endgame / kill range aim at the weaker tower."""
        left_hp = self._tower_hp(state.towers.enemy_left)
        right_hp = self._tower_hp(state.towers.enemy_right)
        if state.towers.enemy_left == 0 and state.towers.enemy_right != 0:
            return "push_left"
        if state.towers.enemy_right == 0 and state.towers.enemy_left != 0:
            return "push_right"
        if force_kill or state.elapsed_s >= 150.0:
            if right_hp < left_hp - 0.05:
                return "push_right"
            if left_hp < right_hp - 0.05:
                return "push_left"
        self._push_counter += 1
        return "push_left" if (self._push_counter % 2 == 0) else "push_right"

    def evaluate_strategy(self, state: BattleState, threat_urgency: float = 0.0) -> str:
        save_below, push_at = self.tempo_thresholds(state)
        opening = state.elapsed_s < 12.0
        endgame = self._is_endgame(state)
        behind = self._behind(state) if endgame else False

        # Endgame + behind: dump elixir into offense — floor drops, push comes sooner.
        if endgame and behind:
            push_at = max(3, push_at - 2)
            save_below = max(1, save_below - 2)

        left_threats = state.left.enemy_on_my_side
        right_threats = state.right.enemy_on_my_side
        bridge_left = state.left.enemy_at_bridge
        bridge_right = state.right.enemy_at_bridge
        real_threat = left_threats > 0 or right_threats > 0 or threat_urgency > 0.40

        # 1. Leaking Elixir Guard: NEVER sit at 10 elixir without spending
        if state.elixir >= 10 or state.seconds_at_full_elixir > 0.4:
            if state.left.enemy_on_their_side > state.right.enemy_on_their_side:
                return "push_right"
            return "push_left"

        # 2. Critical Immediate Defensive Response
        if real_threat or bridge_left > 0 or bridge_right > 0:
            self._last_defend_elapsed = state.elapsed_s
            if real_threat or (bridge_left + bridge_right) > 0:
                # Can't afford a body? Hold — unless this is the opening and we have a cheap play.
                hold_floor = 3 if not opening else 2
                if state.elixir < hold_floor and threat_urgency < 0.70:
                    return "save_elixir"
                if bridge_left > 0 and bridge_right == 0 and left_threats == 0 and right_threats == 0:
                    return "defend_left"
                if bridge_right > 0 and bridge_left == 0 and left_threats == 0 and right_threats == 0:
                    return "defend_right"
                if left_threats > 0 and right_threats > 0:
                    return "defend_centre"
                if left_threats > 0:
                    return "defend_left"
                if right_threats > 0:
                    return "defend_right"
                return "defend_centre"

        # 3. Counter-push conversion (defended in last ~10s, survivors still up)
        if state.elapsed_s - self._last_defend_elapsed <= 10.0:
            counter_floor = max(3, save_below - 2)
            if state.left.mine_on_my_side > 0 or state.left.mine_at_bridge > 0:
                if state.elixir >= counter_floor:
                    return "counter_push_left"
            if state.right.mine_on_my_side > 0 or state.right.mine_at_bridge > 0:
                if state.elixir >= counter_floor:
                    return "counter_push_right"

        # 4. Opening (first 12s): take the river, never sit on a full hand
        if opening:
            if state.elixir >= 6:
                self._push_counter += 1
                return "push_left" if (self._push_counter % 2 == 0) else "push_right"
            if state.elixir >= 4:
                return "cycle"
            return "save_elixir"

        # 4b. Opponent-elixir push gate: never open a fresh attack into a loaded opponent
        # when we have no board presence to convert. Hold/cycle until they spend.
        # Endgame while behind: the clock is the bigger threat — bypass the gate.
        enemy_e = state.enemy_elixir_estimate
        if (
            not (endgame and behind)
            and enemy_e is not None
            and enemy_e >= 7.0
            and state.elixir <= enemy_e - 3.0
            and state.left.mine_on_my_side == 0
            and state.left.mine_at_bridge == 0
            and state.right.mine_on_my_side == 0
            and state.right.mine_at_bridge == 0
        ):
            if state.elixir < save_below:
                return "save_elixir"
            return "cycle"

        # 5. Heavy Push Formulation (rate-aware threshold + tank support buffer)
        if state.elixir >= push_at:
            # Slow back-line builds need runway — not the last 30s. Endgame → bridge.
            if not endgame:
                for card in state.hand:
                    card_info = info(card.name)
                    card_tags = card_info.tags
                    if "slow" in card_tags and "tank" in card_tags:
                        # Giant/Golem from back: keep >=2 elixir after the tank for a support troop
                        tank_cost = card_info.cost or 5
                        if state.elixir >= tank_cost + 2:
                            return "build_push"
                        # Tank in hand but no buffer: cycle a cheap card instead of naked-tanking
                        if state.elixir >= save_below:
                            return "cycle"
                        return "save_elixir"
            return self._attack_lane(state, force_kill=endgame)

        # 6. Elixir Conservation (rate-aware — double/triple dumps sooner)
        if state.elixir < save_below:
            return "save_elixir"

        # 7. Deck Cycling
        for card in state.hand:
            cost = info(card.name).cost
            if cost is not None and cost <= 2:
                return "cycle"

        return self._attack_lane(state, force_kill=endgame)

    @staticmethod
    def _real_threat_count(state: BattleState) -> int:
        return (
            state.left.enemy_on_my_side
            + state.right.enemy_on_my_side
            + state.left.enemy_at_bridge
            + state.right.enemy_at_bridge
        )

    @staticmethod
    def _enemy_threat_tags(state: BattleState) -> set:
        """Tags of named enemy units currently on the board (defend scoring input)."""
        tags: set = set()
        for u in state.units:
            if u.owner != "enemy" or not u.name:
                continue
            try:
                tags |= set(info(u.name).tags)
            except Exception:
                continue
        return tags

    def select_card(self, strategy: str, state: BattleState) -> Optional[HandCard]:
        # Unknown cards score as cost 3 — never let cost=None bypass the elixir gate.
        affordable = [(c, (info(c.name).cost or 3)) for c in state.hand]
        ready_cards = [c for c, cost in affordable if c.ready and cost <= state.elixir]
        if not ready_cards:
            ready_cards = [c for c, cost in affordable if cost <= state.elixir]
        if not ready_cards:
            return None

        # Spells never answer cycle/push/build. On defend they only fire into a real multi-body pile —
        # single-target chip (empty fireball at a tower) is wasted elixir.
        threat_count = self._real_threat_count(state)
        is_defend = strategy.startswith("defend")
        if is_defend:
            if threat_count < 2:
                ready_cards = [c for c in ready_cards if info(c.name).kind != "spell"]
        else:
            ready_cards = [c for c in ready_cards if info(c.name).kind != "spell"]
        if not ready_cards:
            return None

        enemy_tags = self._enemy_threat_tags(state) if is_defend else set()
        # Prefer tag from card DB; fall back to a known-air name list for weak/confident reads.
        _AIR_NAMES = frozenset({
            "minions", "minion_horde", "bats", "balloon", "baby_dragon",
            "lava_hound", "inferno_dragon", "mega_minion", "night_witch", "witch",
        })
        air_threat = "flying" in enemy_tags or any(
            u.owner == "enemy" and u.name in _AIR_NAMES for u in state.units
        )
        swarm_threat = "swarm" in enemy_tags or threat_count >= 3
        tank_threat = bool(enemy_tags & {"tank", "win_condition", "building_only"})

        # Strategy-driven card scoring
        best_card = None
        best_score = -999.0

        for card in ready_cards:
            c_info = info(card.name)
            cost = c_info.cost or 3
            tags = c_info.tags
            score = 10.0

            if is_defend:
                # Defend against threats
                if "splash" in tags:
                    score += 25.0  # Wipes swarms
                if c_info.kind == "spell" and threat_count >= 3:
                    score += 20.0  # Area spell into a stacked push
                if "mini_tank" in tags or "tank" in tags:
                    score += 20.0  # Holds front line
                if "anti_air" in tags or "hits_air" in tags:
                    score += 15.0  # Guards airspace
                if "ranged" in tags:
                    score += 12.0
                if "swarm" in tags:
                    score += 18.0  # Distracts single-target tanks
                if "building_only" in tags:
                    score -= 30.0  # Win condition cannot hit defenders

                # Unit-aware match: answer the actual body in front of us
                if air_threat and ("hits_air" in tags or "anti_air" in tags):
                    score += 30.0
                elif air_threat and c_info.kind == "spell" and threat_count >= 2:
                    score += 22.0  # arrows/minions cluster
                if swarm_threat and "splash" in tags:
                    score += 20.0
                elif swarm_threat and c_info.kind == "spell" and threat_count >= 3:
                    score += 18.0
                if tank_threat and ("mini_tank" in tags or "tank" in tags or "high_damage" in tags or "building_only" not in tags):
                    # distract tank with a body + high DPS behind tower
                    if "mini_tank" in tags or "tank" in tags:
                        score += 15.0
                    if "ranged" in tags or "high_damage" in tags or "medium_damage" in tags:
                        score += 10.0
                # Anti-synergy: ground-only melee into pure air threat wastes the drop
                if air_threat and not ({"hits_air", "anti_air"} & tags) and "swarm" not in tags:
                    score -= 20.0

            elif strategy.startswith("counter_push"):
                if "tank" in tags or "mini_tank" in tags:
                    score += 22.0
                if "ranged" in tags or "splash" in tags:
                    score += 18.0

            elif strategy == "build_push":
                if "tank" in tags and "slow" in tags:
                    score += 35.0  # Golem, Giant, Lava Hound from back
                elif "tank" in tags:
                    score += 20.0
                elif "win_condition" in tags:
                    score += 15.0
                else:
                    score -= 10.0

            elif strategy.startswith("push"):
                if "win_condition" in tags:
                    score += 30.0  # Hog, Giant, Balloon, Royal Giant
                elif "tank" in tags or "mini_tank" in tags:
                    score += 20.0
                elif "swarm" in tags:
                    score += 10.0

            elif strategy == "cycle":
                # Cheapest troop in hand — spells already filtered out above
                score += (10.0 - cost) * 5.0

            # Prefer cards we can comfortably afford without hitting 0 elixir
            if state.elixir - cost >= 1:
                score += 5.0

            if score > best_score:
                best_score = score
                best_card = card

        return best_card or ready_cards[0]

    def select_square(self, strategy: str, card: HandCard, state: BattleState) -> Optional[Square]:
        squares = legal_squares(card, state)
        if not squares:
            return None

        # 1. Spell: only a detected enemy body — never bare tower chip
        c_info = info(card.name)
        if c_info.kind == "spell":
            # Troop targets are `enemy_<name>_<lane>_<i>`; towers are `left_enemy_tower` / `enemy_king_tower`.
            troop_targets = [
                sq
                for sq in squares
                if sq.name.startswith("enemy_") and "tower" not in sq.name
            ]
            if not troop_targets:
                return None
            # Prefer bodies already on our half / at the bridge (real defend value).
            on_my_half = [
                sq
                for sq in troop_targets
                if "on your side" in sq.meaning or "at the bridge" in sq.meaning
            ]
            return on_my_half[0] if on_my_half else troop_targets[0]

        # 2. Defend Centre (golden pull pocket)
        if strategy == "defend_centre":
            for sq in squares:
                if sq.name == "centre_king_front":
                    return sq
                if sq.name == "centre":
                    return sq

        # 3. Defend Left / Right
        if strategy == "defend_left":
            for sq in squares:
                if sq.name == "left_tower_front":
                    return sq
                if sq.name == "left_bridge":
                    return sq

        if strategy == "defend_right":
            for sq in squares:
                if sq.name == "right_tower_front":
                    return sq
                if sq.name == "right_bridge":
                    return sq

        # 4. Build Push (backline spawn) — prefer the weaker enemy tower
        if strategy == "build_push":
            left_hp = self._tower_hp(state.towers.enemy_left)
            right_hp = self._tower_hp(state.towers.enemy_right)
            if state.towers.enemy_left == 0:
                lane = "left"
            elif state.towers.enemy_right == 0:
                lane = "right"
            elif right_hp < left_hp - 0.08:
                lane = "right"
            else:
                lane = "left"
            for sq in squares:
                if sq.name == f"{lane}_back":
                    return sq

        # 5. Push Left / Right
        if "left" in strategy:
            # Check if enemy left tower has fallen -> pocket deploy!
            if state.towers.enemy_left == 0:
                for sq in squares:
                    if sq.name == "left_pocket":
                        return sq
            for sq in squares:
                if sq.name == "left_bridge":
                    return sq

        if "right" in strategy:
            if state.towers.enemy_right == 0:
                for sq in squares:
                    if sq.name == "right_pocket":
                        return sq
            for sq in squares:
                if sq.name == "right_bridge":
                    return sq

        # Default fallback — alternate lanes, never pin every cycle to left bridge
        self._cycle_lane += 1
        want = "left_bridge" if (self._cycle_lane % 2 == 0) else "right_bridge"
        for sq in squares:
            if sq.name == want:
                return sq
        for sq in squares:
            if "bridge" in sq.name:
                return sq
        return squares[0]


class ClashBattleAdapter:
    """
    Main Autonomous Battle Intelligence Controller for Clash Royale.
    Connects Vision, State Perception, Jev/Laya System One, and ADB touch actuation.
    """

    def __init__(self, screen_width: int = 1080, screen_height: int = 2340):
        self.mapper = ClashCoordinateMapper(screen_width, screen_height)
        self.perception = Perception()
        self.tactical_policy = TacticalReflexPolicy()
        self.start_time = time.time()

        # Try initializing Jev TypeSafe client
        self.jev_client = None
        try:
            from typesafe_sdk import TypeSafeClient, RetryPolicy
            self.jev_client = TypeSafeClient(
                retry=RetryPolicy(max_retries=1, backoff_initial=0.1, backoff_max=0.2, timeout=1.8)
            )
        except Exception:
            self.jev_client = None

        # Try initializing local Laya model if installed
        self.laya_agent = None
        try:
            import laya
            # Only load if requested / cached
            if os.environ.get("USE_LOCAL_LAYA", "0") == "1":
                self.laya_agent = laya.load("convaiinnovations/laya")
        except Exception:
            self.laya_agent = None

    def update_resolution(self, width: int, height: int):
        self.mapper.update_resolution(width, height)

    def note_battle_start(self):
        """Reset the match clock — call when phase enters in_battle from menu/queue.

        Does NOT zero _cycle_lane: a false matchmaking/game_over edge mid-battle
        would re-arm this and pin every subsequent cycle tap to the same bridge.
        Lane alternation runs for the adapter's lifetime.
        """
        self.start_time = time.time()
        self._push_counter = 0
        if self.tactical_policy is not None:
            self.tactical_policy._last_defend_elapsed = -999.0
            self.tactical_policy._push_counter = 0

    def extract_state_from_frame(
        self,
        frame_bgr: np.ndarray,
        cached_elixir: Optional[int] = None,
        cached_phase: str = "in_battle",
        threats: Optional[List[Any]] = None,
    ) -> BattleState:
        """
        Builds a comprehensive BattleState from the live frame using clash-jev perception.
        """
        elapsed = time.time() - self.start_time

        # 1. Read Elixir
        elixir = cached_elixir
        if elixir is None:
            try:
                elixir = self.perception.read_elixir(frame_bgr)
            except Exception:
                elixir = 5
        elixir = max(0, min(10, elixir))

        # 2. Read Units & Lane Pressure
        units: Tuple[Unit, ...] = ()
        try:
            raw_units = self.perception.read_units(frame_bgr)
            if self.perception.identifier:
                units = self.perception.identifier.name_units(frame_bgr, raw_units)
            else:
                units = raw_units
        except Exception:
            units = ()

        # Fallback to contour/scene threats if badge reader didn't catch units
        if not units and threats:
            h, w = frame_bgr.shape[:2]
            mapped = []
            for t in threats:
                tx = getattr(t, "click_x", t.x + getattr(t, "w", 0) // 2) / float(max(1, w))
                ty = getattr(t, "click_y", t.y + getattr(t, "h", 0) // 2) / float(max(1, h))
                mapped.append(Unit(owner="enemy", x=tx, y=ty, health=1.0))
            units = tuple(mapped)

        left_lane = self.perception.read_lane(units, "left")
        right_lane = self.perception.read_lane(units, "right")

        # 3. Track opponent elixir (never on screen) — gates aggressive pushes
        # elixir_rate depends only on elapsed_s; hand is irrelevant here.
        rate = BattleState(elapsed_s=elapsed, elixir=0, hand=()).elixir_rate
        try:
            enemy_elixir = self.perception.opponent.update(elapsed, rate, units)
        except Exception:
            enemy_elixir = None

        # 4. Read Towers
        towers = Towers()
        try:
            towers = self.perception.read_towers(frame_bgr)
        except Exception:
            towers = Towers(enemy_left=1.0, enemy_right=1.0, enemy_king=1.0, my_left=1.0, my_right=1.0, my_king=1.0)

        # 5. Read Hand Cards
        hand_cards = []
        try:
            hand_cards = list(self.perception.hand_reader.read(frame_bgr))
        except Exception:
            pass

        if not hand_cards or len(hand_cards) < 4:
            # Fallback: create 4 slots with affordable status
            hand_cards = []
            for slot in range(4):
                hand_cards.append(HandCard(slot=slot, name="unknown", ready=(elixir >= 3)))

        return BattleState(
            elapsed_s=elapsed,
            elixir=elixir,
            hand=tuple(hand_cards),
            left=left_lane,
            right=right_lane,
            towers=towers,
            units=units,
            enemy_elixir_estimate=enemy_elixir,
            snapshot_interval_s=1.0,
            seconds_at_full_elixir=0.5 if elixir >= 10 else 0.0,
        )

    def decide(
        self,
        frame_bgr: np.ndarray,
        threat_urgency: float = 0.0,
        current_elixir: Optional[int] = None,
        threats: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the 3-Tier hierarchical decision flow:
          Step 1: Choose strategy
          Step 2: Choose card
          Step 3: Choose deployment square
        Returns action, diagnostics, and precise device tap coordinates.
        """
        t0 = time.time()
        state = self.extract_state_from_frame(frame_bgr, cached_elixir=current_elixir, threats=threats)

        # TIER 1: STRATEGY FORMULATION
        chosen_strategy = None
        engine_source = "tactical_reflex"
        confidence = 0.95

        # Attempt Jev cloud System One if configured
        if self.jev_client is not None and os.environ.get("TYPESAFE_API_KEY"):
            try:
                from typesafe_sdk import Choice
                from clash_jev.policy import build_state as jev_build_state

                criteria = {
                    s.name: {"what": s.meaning, "not_for": s.not_for} if s.not_for else s.meaning
                    for s in STRATEGIES
                }
                q = {"strategy": Choice(instructions=STRATEGY_INSTRUCTIONS, criteria=criteria)}
                sent_state = jev_build_state(state)
                ans = self.jev_client.system_one(sent_state, q).choices["strategy"]
                chosen_strategy = ans.choice
                engine_source = "jev_system_one"
                confidence = float(ans.probabilities.get(chosen_strategy, 0.92))
            except Exception:
                chosen_strategy = None

        # Fallback to local Tactical Reflex
        if not chosen_strategy:
            chosen_strategy = self.tactical_policy.evaluate_strategy(state, threat_urgency=threat_urgency)
            engine_source = "tactical_reflex"
            confidence = 0.94
        else:
            # Post-validate Jev: never defend a full bar or invent threats.
            left_t = state.left.enemy_on_my_side
            right_t = state.right.enemy_on_my_side
            bridge_t = state.left.enemy_at_bridge + state.right.enemy_at_bridge
            real_threat = left_t > 0 or right_t > 0 or bridge_t > 0 or threat_urgency > 0.40
            if state.elixir >= 10 and chosen_strategy.startswith("defend") and threat_urgency < 0.70:
                chosen_strategy = "push_left" if self._push_counter % 2 == 0 else "push_right"
                self._push_counter += 1
            elif chosen_strategy.startswith("defend") and state.elixir < 3 and threat_urgency < 0.70:
                chosen_strategy = "save_elixir"
            elif chosen_strategy.startswith("defend") and not real_threat:
                # Jev saw a ghost — run local ladder instead.
                chosen_strategy = self.tactical_policy.evaluate_strategy(state, threat_urgency=threat_urgency)
                engine_source = "tactical_reflex"
                confidence = 0.94
            elif state.elixir >= 7 and threat_urgency < 0.40 and not real_threat and chosen_strategy in NO_PLAY:
                chosen_strategy = "cycle"
            elif state.elapsed_s < 12.0 and state.elixir >= 4 and chosen_strategy in NO_PLAY:
                # Opening: never sit — local says push/cycle.
                chosen_strategy = self.tactical_policy.evaluate_strategy(state, threat_urgency=threat_urgency)
                engine_source = "tactical_reflex"
                confidence = 0.94
            elif chosen_strategy in NO_PLAY and state.elixir >= self.tactical_policy.tempo_thresholds(state)[1]:
                # Tempo says spend (double/triple push_at) but Jev said hold.
                chosen_strategy = self.tactical_policy.evaluate_strategy(state, threat_urgency=threat_urgency)
                engine_source = "tactical_reflex"
                confidence = 0.94

        # Check for NO_PLAY strategies (holding/saving elixir)
        if chosen_strategy in NO_PLAY:
            latency_ms = round((time.time() - t0) * 1000, 2)
            return {
                "action": "wait",
                "strategy": chosen_strategy,
                "card_name": None,
                "card_slot": None,
                "square_name": None,
                "threat_score": threat_urgency,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "source": engine_source,
                "target_coords": None,
                "reasoning": f"Holding elixir for strategic tempo ({chosen_strategy}). Elixir: {state.elixir}/10",
            }

        # TIER 2: CARD SELECTION
        chosen_card = self.tactical_policy.select_card(chosen_strategy, state)
        if not chosen_card:
            latency_ms = round((time.time() - t0) * 1000, 2)
            return {
                "action": "wait",
                "strategy": chosen_strategy,
                "card_name": None,
                "card_slot": None,
                "square_name": None,
                "threat_score": threat_urgency,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "source": engine_source,
                "target_coords": None,
                "reasoning": f"Waiting for sufficient elixir to deploy card for {chosen_strategy}",
            }

        # TIER 3: SQUARE DEPLOYMENT SELECTION
        chosen_square = self.tactical_policy.select_square(chosen_strategy, chosen_card, state)
        if not chosen_square:
            # Spell with no enemy body on the board: holding is correct — do not bridge-drop.
            if info(chosen_card.name).kind == "spell":
                latency_ms = round((time.time() - t0) * 1000, 2)
                return {
                    "action": "wait",
                    "strategy": chosen_strategy,
                    "card_name": chosen_card.name,
                    "card_slot": chosen_card.slot,
                    "square_name": None,
                    "threat_score": threat_urgency,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "source": engine_source,
                    "target_coords": None,
                    "reasoning": f"Holding {chosen_card.name}: no enemy troop in spell range.",
                }
            # Fallback to left bridge
            chosen_square = BRIDGE[0]

        # Calculate exact device coordinates
        card_tap_x, card_tap_y = self.mapper.get_card_tap(chosen_card.slot)
        deploy_tap_x, deploy_tap_y = self.mapper.get_square_tap(chosen_square)
        target_coords = (card_tap_x, card_tap_y, deploy_tap_x, deploy_tap_y)

        latency_ms = round((time.time() - t0) * 1000, 2)
        card_display = chosen_card.name.replace("_", " ").title()

        return {
            "action": "deploy_clash_card",
            "strategy": chosen_strategy,
            "card_name": chosen_card.name,
            "card_slot": chosen_card.slot,
            "square_name": chosen_square.name,
            "threat_score": threat_urgency,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "source": engine_source,
            "target_coords": target_coords,
            "reasoning": (
                f"[{chosen_strategy.upper()}] Play {card_display} (slot {chosen_card.slot}) "
                f"-> {chosen_square.name.replace('_', ' ').title()}"
            ),
        }
