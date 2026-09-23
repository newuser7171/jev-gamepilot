"""
Clash of Clans autonomous raid adapter for jev-gamepilot.
Raid loop: home → find match → deploy full army around enemy base → end battle → loot screen.

Unlike Clash Royale there is no per-second elixir: army is pre-trained. Decisions
are (phase, army slots left, base geometry) → deploy / hold / end_battle.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Normalized army-bar slot centers (portrait 1080x2340), left → right.
COC_ARMY_SLOTS: Tuple[Tuple[float, float], ...] = (
    (0.155, 0.905),
    (0.285, 0.905),
    (0.415, 0.905),
    (0.545, 0.905),
    (0.675, 0.905),
    (0.805, 0.905),
    (0.915, 0.905),
)

# Deploy ring around enemy base in normalized map coords (clockwise from NW).
DEPLOY_RING: Tuple[Tuple[float, float], ...] = (
    (0.22, 0.28),
    (0.50, 0.22),
    (0.78, 0.28),
    (0.86, 0.48),
    (0.78, 0.66),
    (0.50, 0.72),
    (0.22, 0.66),
    (0.14, 0.48),
    (0.35, 0.40),
    (0.65, 0.40),
)

# End after this many deploy taps or raid seconds without a results screen.
MAX_DEPLOYS = 40
MAX_RAID_SECONDS = 165.0
# Seconds to coast after the last deploy before force-ending (troops still fighting).
POST_DEPLOY_END_DELAY = 18.0


def detect_coc_phase(frame_bgr: np.ndarray) -> str:
    """Coarse phase for CoC portrait frames. Returns one of:
    home_village | attack_search | in_raid | results | unknown
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return "unknown"
    h, w = frame_bgr.shape[:2]
    if h < 100 or w < 100:
        return "unknown"

    # Search: red Cancel button (same family as CR matchmaking).
    btn = frame_bgr[int(h * 0.78):int(h * 0.88), int(w * 0.35):int(w * 0.65)]
    if btn.size:
        red = (btn[:, :, 2] > 170) & (btn[:, :, 1] < 100) & (btn[:, :, 0] < 100)
        if int(red.sum()) > 900:
            return "attack_search"

    # Results: dense gold loot row + large purple elixir row mid-screen.
    mid = frame_bgr[int(h * 0.38):int(h * 0.62), int(w * 0.15):int(w * 0.85)]
    if mid.size:
        gold = (mid[:, :, 2] > 170) & (mid[:, :, 1] > 140) & (mid[:, :, 0] < 110)
        purple = (mid[:, :, 2] > 150) & (mid[:, :, 0] > 130) & (mid[:, :, 1] < 120)
        if int(gold.sum()) > 4000 and int(purple.sum()) > 3000:
            return "results"

    # In-raid: destruction % band top-center + full-width troop bar bottom.
    troop_band = frame_bgr[int(h * 0.86):int(h * 0.97), :]
    if troop_band.size:
        gray = cv2.cvtColor(troop_band, cv2.COLOR_BGR2GRAY)
        # High local contrast = army icons, not flat village grass.
        if float(gray.std()) > 42 and float(gray.mean()) < 145:
            top = frame_bgr[int(h * 0.02):int(h * 0.12), int(w * 0.30):int(w * 0.70)]
            if top.size:
                tg = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
                # Timer / % text is bright against darker HUD chip.
                if float(tg.max()) > 200:
                    return "in_raid"

    # Home: solid green Attack button bottom band.
    home_btn = frame_bgr[int(h * 0.72):int(h * 0.84), int(w * 0.40):int(w * 0.92)]
    if home_btn.size:
        orange = (
            (home_btn[:, :, 2] > 150)
            & (home_btn[:, :, 1] > 70)
            & (home_btn[:, :, 1] < 170)
            & (home_btn[:, :, 0] < 90)
        )
        if int(orange.sum()) > 5500:
            return "home_village"

    return "unknown"


def enemy_base_box(frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Axis-aligned box of non-grass structures in the map area (excl. HUD bands).

    Returns (x1, y1, x2, y2) in device pixels, or None.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    roi = frame_bgr[int(h * 0.10):int(h * 0.82), int(w * 0.05):int(w * 0.95)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Green turf (village grass / deploy zone).
    grass = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    # Pale sky / edge chrome often sits above the base — also drop very bright.
    bright = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    structures = cv2.bitwise_not(cv2.bitwise_or(grass, bright))
    structures = cv2.morphologyEx(structures, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    structures = cv2.morphologyEx(structures, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(structures, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 0.02 * roi.shape[0] * roi.shape[1]:
        return None
    x, y, bw, bh = cv2.boundingRect(largest)
    x1 = int(w * 0.05) + x
    y1 = int(h * 0.10) + y
    return (x1, y1, x1 + bw, y1 + bh)


def perimeter_points(box: Tuple[int, int, int, int], w: int, h: int) -> List[Tuple[float, float]]:
    """Normalized points evenly around the box perimeter (map-target safe margin)."""
    x1, y1, x2, y2 = box
    # Pull targets 6% outside the structure cluster so troops land on grass.
    mx = 0.04 * (x2 - x1)
    my = 0.04 * (y2 - y1)
    x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my
    x1, y1 = max(0.02, x1), max(0.12, y1)
    x2, y2 = min(w * 0.98, x2), min(h * 0.80, y2)

    pts: List[Tuple[float, float]] = []
    steps = 4
    for i in range(steps):
        t = (i + 0.5) / steps
        pts.append((x1 + t * (x2 - x1), y1))
        pts.append((x1 + t * (x2 - x1), y2))
        pts.append((x1, y1 + t * (y2 - y1)))
        pts.append((x2, y1 + t * (y2 - y1)))
    return [(px / w, py / h) for px, py in pts]


class CocRaidAdapter:
    """Decide() returns the same dict shape as ClashBattleAdapter for brain/pilot reuse."""

    def __init__(self, screen_width: int = 1080, screen_height: int = 2340):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.start_time = time.time()
        self.raid_open = False
        self.deploy_count = 0
        self._slot_idx = 0
        self._target_idx = 0
        self._last_deploy_at = 0.0
        self._phase = "unknown"
        self._base_box: Optional[Tuple[int, int, int, int]] = None
        self._last_targets: List[Tuple[float, float]] = list(DEPLOY_RING)

    def update_resolution(self, width: int, height: int) -> None:
        self.screen_width = width
        self.screen_height = height

    def note_raid_start(self) -> None:
        self.start_time = time.time()
        self.raid_open = True
        self.deploy_count = 0
        self._slot_idx = 0
        self._target_idx = 0
        self._last_deploy_at = 0.0
        self._base_box = None
        self._last_targets = list(DEPLOY_RING)

    def note_raid_end(self, frame_bgr: Optional[np.ndarray] = None) -> Optional[dict]:
        if not self.raid_open:
            return None
        self.raid_open = False
        elapsed = time.time() - self.start_time
        return {
            "elapsed_s": round(elapsed, 1),
            "deploys": self.deploy_count,
            "phase": self._phase,
            "frame_seen": frame_bgr is not None,
        }

    def _slot_xy(self, slot: int) -> Tuple[int, int]:
        rx, ry = COC_ARMY_SLOTS[slot % len(COC_ARMY_SLOTS)]
        return int(rx * self.screen_width), int(ry * self.screen_height)

    def _next_target(self) -> Tuple[int, int]:
        pts = self._last_targets or list(DEPLOY_RING)
        rx, ry = pts[self._target_idx % len(pts)]
        self._target_idx += 1
        return int(rx * self.screen_width), int(ry * self.screen_height)

    def _wait_decision(self, strategy: str, reasoning: str) -> Dict[str, Any]:
        return {
            "action": "wait",
            "strategy": strategy,
            "card_name": None,
            "card_slot": None,
            "square_name": None,
            "threat_score": 0.0,
            "confidence": 0.95,
            "latency_ms": 0.1,
            "source": "coc_raid",
            "target_coords": None,
            "reasoning": reasoning,
        }

    def decide(
        self,
        frame_bgr: np.ndarray,
        threat_urgency: float = 0.0,
        phase: str = "in_raid",
    ) -> Dict[str, Any]:
        t0 = time.time()
        self._phase = phase or "unknown"
        elapsed = time.time() - self.start_time

        if self._phase in ("home_village", "unknown") and not self.raid_open:
            # Idle on village between raids — pilot handles find_match via phase.
            return self._wait_decision("home", "Waiting on home village (no raid open).")

        if not self.raid_open and self._phase == "in_raid":
            self.note_raid_start()

        if self._phase == "results":
            out = self.note_raid_end(frame_bgr)
            return {
                "action": "confirm_ok",
                "strategy": "collect_loot",
                "card_name": None,
                "card_slot": None,
                "square_name": None,
                "threat_score": 0.0,
                "confidence": 0.99,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "source": "coc_raid",
                "target_coords": None,
                "reasoning": f"Raid results after {elapsed:.0f}s, {out['deploys'] if out else self.deploy_count} deploys.",
            }

        if self._phase == "attack_search":
            return self._wait_decision("search", "Waiting for opponent base to load.")

        if not self.raid_open:
            return self._wait_decision("idle", "Raid adapter not open for this phase.")

        # Refresh deploy geometry from the live frame.
        if frame_bgr is not None and frame_bgr.size:
            box = enemy_base_box(frame_bgr)
            if box is not None:
                self._base_box = box
                self._last_targets = perimeter_points(box, self.screen_width, self.screen_height)

        # Force-end after full army spent + coast time, or hard raid timeout.
        if self.deploy_count >= MAX_DEPLOYS:
            return self._wait_decision("army_spent", "Army fully deployed — waiting for end.")
        if elapsed > MAX_RAID_SECONDS:
            return {
                "action": "end_battle",
                "strategy": "raid_timeout",
                "card_name": None,
                "card_slot": None,
                "square_name": "end_battle",
                "threat_score": threat_urgency,
                "confidence": 0.97,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "source": "coc_raid",
                "target_coords": None,
                "reasoning": f"Raid timeout at {elapsed:.0f}s.",
            }
        if (
            self.deploy_count > 0
            and self._last_deploy_at
            and (time.time() - self._last_deploy_at) > POST_DEPLOY_END_DELAY
            and self.deploy_count >= 14
        ):
            return {
                "action": "end_battle",
                "strategy": "post_deploy_clear",
                "card_name": None,
                "card_slot": None,
                "square_name": "end_battle",
                "threat_score": threat_urgency,
                "confidence": 0.9,
                "latency_ms": round((time.time() - t0) * 1000, 2),
                "source": "coc_raid",
                "target_coords": None,
                "reasoning": "Army deployed and fight coasted — ending for loot.",
            }

        slot = self._slot_idx % len(COC_ARMY_SLOTS)
        sx, sy = self._slot_xy(slot)
        tx, ty = self._next_target()
        self._slot_idx += 1
        self.deploy_count += 1
        self._last_deploy_at = time.time()
        latency_ms = round((time.time() - t0) * 1000, 2)
        return {
            "action": "deploy_troop",
            "strategy": "spread_deploy",
            "card_name": f"troop_slot_{slot}",
            "card_slot": slot,
            "square_name": f"deploy_{self._target_idx}",
            "threat_score": threat_urgency,
            "confidence": 0.93,
            "latency_ms": latency_ms,
            "source": "coc_raid",
            "target_coords": (sx, sy, tx, ty),
            "reasoning": (
                f"Deploy slot {slot} → ({tx},{ty}) "
                f"[{'base-box' if self._base_box else 'ring'} #{self._target_idx}]"
            ),
        }
