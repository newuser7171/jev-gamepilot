"""
Brawl Stars match adapter for jev-gamepilot (landscape).

Loop: menu → find match → in_match (joystick move + attack / super) → victory/defeat → menu.
No elixir: movement is continuous joystick, attacks are timed taps on the right-hand buttons.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Landscape geometry (2340x1080 reference) — normalized centers.
JOY_BASE = (0.14, 0.72)       # left virtual-stick rest
ATTACK_BTN = (0.88, 0.78)     # main attack
SUPER_BTN = (0.78, 0.88)      # super / star power
GADGET_BTN = (0.92, 0.55)     # gadget

# Objective roam points (gem grab / brawl ball style mid-control).
ROAM_RING: Tuple[Tuple[float, float], ...] = (
    (0.50, 0.40),
    (0.40, 0.50),
    (0.60, 0.50),
    (0.50, 0.55),
    (0.35, 0.42),
    (0.65, 0.42),
)

# Cadence bounds (seconds).
MOVE_HOLD_S = 0.35
ATTACK_COOLDOWN_S = 0.85
SUPER_COOLDOWN_S = 2.5


def detect_brawl_phase(frame_bgr: np.ndarray) -> str:
    """menu | matchmaking | in_match | results | unknown (landscape frames)."""
    if frame_bgr is None or frame_bgr.size == 0:
        return "unknown"
    h, w = frame_bgr.shape[:2]
    if h < 80 or w < 80:
        return "unknown"
    # Landscape expected: w >= h. Still tolerate portrait mis-captures.
    portrait = h > w

    # Matchmaking: red Cancel mid-right or full-screen dark wipe + spinning cue.
    cancel = frame_bgr[int(h * 0.35):int(h * 0.65), int(w * 0.40):int(w * 0.60)]
    if cancel.size:
        red = (cancel[:, :, 2] > 170) & (cancel[:, :, 1] < 90) & (cancel[:, :, 0] < 90)
        if int(red.sum()) > 600:
            return "matchmaking"

    # Results: large gold/blue victory band or defeat red-orange band center.
    mid = frame_bgr[int(h * 0.25):int(h * 0.55), int(w * 0.25):int(w * 0.75)]
    if mid.size:
        gold = (mid[:, :, 2] > 180) & (mid[:, :, 1] > 150) & (mid[:, :, 0] < 100)
        blue = (mid[:, :, 0] > 170) & (mid[:, :, 1] > 100) & (mid[:, :, 2] < 120)
        if int(gold.sum()) > 8000 and int(blue.sum()) > 4000:
            return "results"

    # In-match: left joystick ring (light circle) + right attack cluster.
    joy = frame_bgr[int(h * 0.55):int(h * 0.95), 0:int(w * 0.30)]
    atk = frame_bgr[int(h * 0.55):int(h * 0.98), int(w * 0.72):]
    joy_ok = False
    atk_ok = False
    if joy.size:
        hg = cv2.cvtColor(joy, cv2.COLOR_BGR2HSV)
        # Whitish translucent stick chrome.
        stick = cv2.inRange(hg, (0, 0, 160), (180, 50, 255))
        joy_ok = int(stick.sum()) > 0  # presence counted via nonzero below
        joy_ok = int(cv2.countNonZero(stick)) > 2500
    if atk.size:
        ga = cv2.cvtColor(atk, cv2.COLOR_BGR2HSV)
        # Attack button is warm red/orange.
        warm = cv2.inRange(ga, (0, 120, 120), (25, 255, 255))
        warm2 = cv2.inRange(ga, (165, 120, 120), (180, 255, 255))
        atk_ok = int(cv2.countNonZero(warm) + cv2.countNonZero(warm2)) > 1800
    if joy_ok and atk_ok:
        return "in_match"
    if portrait and (joy_ok or atk_ok):
        # Tolerate slight orientation confusion during transitions.
        return "in_match"

    # Menu: large centered Play / event button (bright yellow-green).
    menu = frame_bgr[int(h * 0.55):int(h * 0.90), int(w * 0.35):int(w * 0.75)]
    if menu.size:
        ylw = (
            (menu[:, :, 2] > 190)
            & (menu[:, :, 1] > 170)
            & (menu[:, :, 0] < 120)
        )
        if int(ylw.sum()) > 6000:
            return "menu"

    return "unknown"


def detect_enemy_centroid(frame_bgr: np.ndarray) -> Optional[Tuple[float, float]]:
    """Normalized centroid of red enemy outlines / health bars in the upper playfield."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    # Enemies highlight red — focus center-top 70% (exclude our button chrome).
    roi = frame_bgr[0:int(h * 0.70), int(w * 0.18):int(w * 0.88)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 140, 120), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 140, 120), (180, 255, 255))
    mask = cv2.bitwise_or(red1, red2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n = int(cv2.countNonZero(mask))
    if n < 400:
        return None
    ys, xs = np.where(mask > 0)
    cx = (float(np.median(xs)) + int(w * 0.18)) / float(w)
    cy = float(np.median(ys)) / float(h)
    return (cx, cy)


def detect_super_ready(frame_bgr: np.ndarray) -> bool:
    """Super button charges to a bright yellow/gold ring when ready."""
    if frame_bgr is None or frame_bgr.size == 0:
        return False
    h, w = frame_bgr.shape[:2]
    bx, by = SUPER_BTN
    x1 = max(0, int(bx * w - 0.06 * w))
    x2 = min(w, int(bx * w + 0.06 * w))
    y1 = max(0, int(by * h - 0.10 * h))
    y2 = min(h, int(by * h + 0.10 * h))
    reg = frame_bgr[y1:y2, x1:x2]
    if reg.size == 0:
        return False
    hsv = cv2.cvtColor(reg, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, (25, 160, 180), (40, 255, 255))
    return int(cv2.countNonZero(gold)) > 900


class BrawlMatchAdapter:
    """Returns pilot decision dicts: move_to / attack / use_super / wait / confirm_ok."""

    def __init__(self, screen_width: int = 2340, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.start_time = time.time()
        self.match_open = False
        self._roam_idx = 0
        self._last_attack = 0.0
        self._last_super = 0.0
        self._last_move = 0.0
        self._phase = "unknown"
        self.kills = 0  # optional external increment

    def update_resolution(self, width: int, height: int) -> None:
        self.screen_width = width
        self.screen_height = height

    def note_match_start(self) -> None:
        self.start_time = time.time()
        self.match_open = True
        self._roam_idx = 0
        self._last_attack = 0.0
        self._last_super = 0.0
        self._last_move = 0.0

    def note_match_end(self, frame_bgr: Optional[np.ndarray] = None) -> Optional[dict]:
        if not self.match_open:
            return None
        self.match_open = False
        return {
            "elapsed_s": round(time.time() - self.start_time, 1),
            "phase": self._phase,
            "frame_seen": frame_bgr is not None,
        }

    def _xy(self, rx: float, ry: float) -> Tuple[int, int]:
        return int(rx * self.screen_width), int(ry * self.screen_height)

    def _base(self, strategy: str, action: str, reasoning: str, coords=None, conf=0.95) -> Dict[str, Any]:
        return {
            "action": action,
            "strategy": strategy,
            "card_name": None,
            "card_slot": None,
            "square_name": None,
            "threat_score": 0.0,
            "confidence": conf,
            "latency_ms": 0.1,
            "source": "brawl_match",
            "target_coords": coords,
            "reasoning": reasoning,
        }

    def decide(
        self,
        frame_bgr: np.ndarray,
        threat_urgency: float = 0.0,
        phase: str = "in_match",
    ) -> Dict[str, Any]:
        t0 = time.time()
        self._phase = phase or "unknown"
        now = time.time()

        if self._phase == "menu":
            if self.match_open:
                self.note_match_end(frame_bgr)
            return self._base("menu", "wait", "On Brawl menu — pilot queues match.", conf=0.99)

        if self._phase == "matchmaking":
            return self._base("queue", "wait", "Matchmaking — standby.", conf=0.99)

        if self._phase == "results":
            out = self.note_match_end(frame_bgr)
            return self._base(
                "results",
                "confirm_ok",
                f"Match over ({out['elapsed_s'] if out else '?'}s) — dismiss.",
                conf=0.99,
            )

        if self._phase != "in_match":
            return self._base("unknown_phase", "wait", f"Unrecognized brawl phase {self._phase!r}.")

        if not self.match_open:
            self.note_match_start()

        enemy = detect_enemy_centroid(frame_bgr)
        super_ready = detect_super_ready(frame_bgr)

        # Super first when charged and anyone is visible (or blind mid-fight).
        if super_ready and (now - self._last_super) >= SUPER_COOLDOWN_S:
            self._last_super = now
            return self._base(
                "super_burst",
                "use_super",
                "Super charged — firing.",
                coords=self._xy(*SUPER_BTN),
                conf=0.96,
            )

        # Attack when enemy centroid known and attack is off cooldown.
        if enemy and (now - self._last_attack) >= ATTACK_COOLDOWN_S:
            self._last_attack = now
            # Aimed attack: short joystick flick toward enemy + tap attack.
            jx, jy = self._xy(*JOY_BASE)
            ex, ey = self._xy(enemy[0], enemy[1])
            return self._base(
                "engage_enemy",
                "attack",
                f"Enemy at ({enemy[0]:.2f},{enemy[1]:.2f}) — aim + attack.",
                coords=(jx, jy, ex, ey, *self._xy(*ATTACK_BTN)),
                conf=0.94,
            )

        # No enemy / on cooldown: strafe the objective ring so we don't stand still.
        if (now - self._last_move) >= MOVE_HOLD_S:
            self._last_move = now
            rx, ry = ROAM_RING[self._roam_idx % len(ROAM_RING)]
            self._roam_idx += 1
            jx, jy = self._xy(*JOY_BASE)
            tx, ty = self._xy(rx, ry)
            return self._base(
                "roam_objective",
                "move_to",
                f"Roam point {self._roam_idx} — holding mid control.",
                coords=(jx, jy, tx, ty),
                conf=0.9,
            )

        return self._base("cooldown_hold", "wait", "Move/attack cadence cooling down.")
