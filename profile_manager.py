"""
Profile Manager for Universal Jev-GamePilot.
Manages game configurations, keybindings, vision parameters, and Jev System One objectives.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Dict, List, Optional


@dataclass
class GameAction:
    name: str  # e.g. "jump", "dodge_left", "click_target"
    key: str  # e.g. "space", "left", "a", "mouse_left"
    description: str  # Semantic description fed to Jev Choice criteria
    duration_sec: float = 0.10  # Key hold duration
    is_mouse_click: bool = False


@dataclass
class GameProfile:
    id: str
    name: str
    category: str  # "runner", "platformer", "clicker", "arcade", "custom"
    description: str
    actions: List[GameAction] = field(default_factory=list)
    default_window_keyword: str = ""
    scan_direction: str = "right"  # "right", "left", "omnidirectional", "click_targets"
    auto_restart_key: str = "space"
    is_builtin: bool = False


DEFAULT_PROFILES: List[GameProfile] = [
    GameProfile(
        id="runner_dino",
        name="🦖 Chrome Dino & Obstacle Runner",
        category="runner",
        description="Endless runner dodging ground cacti and aerial birds",
        default_window_keyword="dino",
        scan_direction="right",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="jump",
                key="space",
                description="Vault over oncoming low cacti, clusters, or low-flying birds",
                duration_sec=0.14,
            ),
            GameAction(
                name="duck",
                key="down",
                description="Duck/slide under mid-altitude flying pterodactyl birds",
                duration_sec=0.35,
            ),
            GameAction(
                name="run_normal",
                key="none",
                description="Horizon is clear or obstacle is harmless high bird",
            ),
        ],
    ),
    GameProfile(
        id="runner_3lane",
        name="🏄 3-Lane Runner (Subway Surfers / Mobile)",
        category="runner",
        description="3-lane runner with barriers, trains, and coins",
        default_window_keyword="subway",
        scan_direction="omnidirectional",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="dodge_left",
                key="left",
                description="Shift one lane to the left to avoid a train or obstacle",
                duration_sec=0.08,
            ),
            GameAction(
                name="dodge_right",
                key="right",
                description="Shift one lane to the right to avoid a train or obstacle",
                duration_sec=0.08,
            ),
            GameAction(
                name="jump",
                key="up",
                description="Leap over low hurdles, roadblocks, or jump on train roofs",
                duration_sec=0.12,
            ),
            GameAction(
                name="slide",
                key="down",
                description="Slide under high overhead tunnel signs or barriers",
                duration_sec=0.30,
            ),
            GameAction(
                name="maintain_course",
                key="none",
                description="Current lane is open and safe",
            ),
        ],
    ),
    GameProfile(
        id="flappy_tap",
        name="🐦 Flappy Bird & One-Tap Jumper",
        category="arcade",
        description="Single-button rhythm and obstacle navigation jumper",
        default_window_keyword="flappy",
        scan_direction="right",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="flap",
                key="space",
                description="Flap wings to gain altitude and clear pipes or gaps",
                duration_sec=0.06,
            ),
            GameAction(
                name="glide",
                key="none",
                description="Descend naturally through gap without flapping",
            ),
        ],
    ),
    GameProfile(
        id="chess_copilot",
        name="♟️ Chess.com & Lichess Screen Copilot",
        category="strategy",
        description="Analyzes 8x8 chessboard on screen, evaluates tactical moves, and plays or highlights squares",
        default_window_keyword="chess",
        scan_direction="omnidirectional",
        auto_restart_key="none",
        is_builtin=True,
        actions=[
            GameAction(
                name="play_best_move",
                key="mouse_left",
                description="Execute Jev recommended move by clicking start and target squares",
                is_mouse_click=True,
            ),
            GameAction(
                name="analyze_only",
                key="none",
                description="Calculate and display optimal move overlay without sending clicks",
            ),
        ],
    ),
    GameProfile(
        id="aim_clicker",
        name="🎯 Target Clicker & Aim Trainer (Osu / Aim Labs)",
        category="clicker",
        description="Fast reflex targeting clicking active targets and objects",
        default_window_keyword="aim",
        scan_direction="click_targets",
        auto_restart_key="space",
        is_builtin=True,
        actions=[
            GameAction(
                name="click_target",
                key="mouse_left",
                description="Immediately click the highest priority target on screen",
                is_mouse_click=True,
            ),
            GameAction(
                name="wait",
                key="none",
                description="No active targets in view",
            ),
        ],
    ),
    GameProfile(
        id="retro_platformer",
        name="🕹️ 2D Platformer / Retro Arcade",
        category="platformer",
        description="Side-scrolling platformer with running, jumping, and attacking",
        default_window_keyword="game",
        scan_direction="right",
        auto_restart_key="enter",
        is_builtin=True,
        actions=[
            GameAction(
                name="move_right",
                key="d",
                description="Advance forward towards the right",
                duration_sec=0.25,
            ),
            GameAction(
                name="move_left",
                key="a",
                description="Retreat left away from an approaching hazard",
                duration_sec=0.15,
            ),
            GameAction(
                name="jump",
                key="space",
                description="Jump over pit, platform, or enemy",
                duration_sec=0.18,
            ),
            GameAction(
                name="attack",
                key="j",
                description="Attack or shoot nearby enemy",
                duration_sec=0.08,
            ),
            GameAction(
                name="stand_idle",
                key="none",
                description="Wait for moving platform or hazard to pass",
            ),
        ],
    ),
]


class ProfileManager:
    def __init__(self, profiles_path: str = "profiles.json"):
        self.profiles_path = profiles_path
        self.profiles: Dict[str, GameProfile] = {}
        self.load_profiles()

    def load_profiles(self):
        """Loads default presets and any user custom profiles."""
        self.profiles = {p.id: p for p in DEFAULT_PROFILES}

        if os.path.exists(self.profiles_path):
            try:
                with open(self.profiles_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        actions = [GameAction(**a) for a in item.get("actions", [])]
                        item["actions"] = actions
                        profile = GameProfile(**item)
                        self.profiles[profile.id] = profile
            except Exception as e:
                print(f"[ProfileManager] Error loading profiles.json: {e}")

    def save_custom_profiles(self):
        """Persists custom profiles to JSON."""
        custom_list = [
            asdict(p) for p in self.profiles.values() if not p.is_builtin
        ]
        try:
            with open(self.profiles_path, "w", encoding="utf-8") as f:
                json.dump(custom_list, f, indent=2)
        except Exception as e:
            print(f"[ProfileManager] Error saving profiles: {e}")

    def list_profiles(self) -> List[GameProfile]:
        return list(self.profiles.values())

    def get_profile(self, profile_id: str) -> Optional[GameProfile]:
        return self.profiles.get(profile_id)

    def add_custom_profile(
        self,
        profile_id: str,
        name: str,
        category: str,
        description: str,
        actions: List[GameAction],
        window_keyword: str = "",
        scan_direction: str = "right",
    ) -> GameProfile:
        prof = GameProfile(
            id=profile_id,
            name=name,
            category=category,
            description=description,
            actions=actions,
            default_window_keyword=window_keyword,
            scan_direction=scan_direction,
            is_builtin=False,
        )
        self.profiles[profile_id] = prof
        self.save_custom_profiles()
        return prof

    def delete_profile(self, profile_id: str) -> bool:
        if profile_id in self.profiles and not self.profiles[profile_id].is_builtin:
            del self.profiles[profile_id]
            self.save_custom_profiles()
            return True
        return False
