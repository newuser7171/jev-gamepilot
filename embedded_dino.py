"""
Embedded Native Dino Arena for Jev-GamePilot.
A high-performance Tkinter Canvas implementation of the Chrome T-Rex runner
that runs directly inside the HUD with zero window-occlusion issues.
"""

import math
import random
import time
import tkinter as tk
from typing import Any, Callable, Dict, List, Optional
from adapters.dino_adapter import DinoEntity, DinoGameState, ObstacleEntity
from jev_brain import JevBrain


class EmbeddedDinoArena(tk.Canvas):
    def __init__(self, master, width=780, height=260, **kwargs):
        super().__init__(
            master,
            width=width,
            height=height,
            bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#2b2f42",
            **kwargs,
        )
        self.arena_w = width
        self.arena_h = height
        self.ground_y = int(height * 0.78)

        # Game physics & variables
        self.speed = 6.0
        self.score = 0
        self.hi_score = 0
        self.is_game_over = False
        self.is_night = False
        self.frame_count = 0
        self.is_loop_running = False

        # Dino entity
        self.dino_x = 55
        self.dino_y = self.ground_y - 44
        self.dino_w = 40
        self.dino_h = 44
        self.vy = 0.0
        self.gravity = 0.70
        self.jump_force = -12.2
        self.is_grounded = True
        self.is_ducking = False

        # Obstacles
        self.obstacles: List[Dict[str, Any]] = []
        self.next_obstacle_dist = 90.0

        # AI Autopilot
        self.autopilot_enabled = False
        self.brain: Optional[JevBrain] = None
        self.telemetry_cb: Optional[Callable[[Dict[str, Any]], None]] = None
        self.last_ai_action_time = 0.0
        self.total_jumps = 0
        self.total_ducks = 0

        # Bind keyboard controls for manual testing
        self.bind("<Button-1>", lambda e: self.focus_set())
        self.bind("<space>", lambda e: self._on_user_jump())
        self.bind("<Up>", lambda e: self._on_user_jump())
        self.bind("<Down>", lambda e: self._on_user_duck())
        self.bind("<KeyRelease-Down>", lambda e: self._on_user_duck_release())

    def start(self):
        """Starts the internal 60 FPS animation loop."""
        if not self.is_loop_running:
            self.is_loop_running = True
            self._tick()

    def stop(self):
        self.is_loop_running = False

    def reset_game(self):
        self.speed = 6.0
        self.score = 0
        self.is_game_over = False
        self.obstacles.clear()
        self.dino_y = self.ground_y - 44
        self.dino_h = 44
        self.vy = 0.0
        self.is_grounded = True
        self.is_ducking = False
        self.next_obstacle_dist = 85.0
        self.is_night = False
        self.configure(bg="#ffffff")

    def _on_user_jump(self):
        if self.is_game_over:
            self.reset_game()
        elif self.is_grounded:
            self.jump()

    def _on_user_duck(self):
        if not self.is_game_over:
            self.duck(True)

    def _on_user_duck_release(self):
        self.duck(False)

    def jump(self):
        if self.is_grounded:
            self.vy = self.jump_force
            self.is_grounded = False
            self.total_jumps += 1

    def duck(self, active: bool = True):
        self.is_ducking = active
        if active:
            self.total_ducks += 1
            if not self.is_grounded:
                self.vy += 4.5  # Fast-fall!

    def set_autopilot(
        self,
        enabled: bool,
        brain: Optional[JevBrain] = None,
        cb: Optional[Callable] = None,
    ):
        self.autopilot_enabled = enabled
        self.brain = brain
        self.telemetry_cb = cb

    def _spawn_obstacle(self):
        r = random.random()
        if self.score > 250 and r < 0.35:
            # Pterodactyl Bird
            alt_choices = [
                self.ground_y - 28,  # Low bird: Jump over
                self.ground_y - 50,  # Mid bird: Duck under
                self.ground_y - 75,  # High bird: Safe
            ]
            alt = random.choice(alt_choices)
            alt_type = (
                "bird_low"
                if alt == alt_choices[0]
                else (
                    "bird_mid"
                    if alt == alt_choices[1]
                    else "bird_high"
                )
            )
            self.obstacles.append(
                {
                    "type": alt_type,
                    "x": self.arena_w + 10,
                    "y": alt,
                    "w": 38,
                    "h": 26,
                }
            )
        else:
            # Cactus
            is_large = random.random() > 0.5
            is_cluster = random.random() > 0.65
            w = 58 if is_cluster else (24 if is_large else 18)
            h = 46 if is_large else 34
            self.obstacles.append(
                {
                    "type": (
                        "cactus_cluster"
                        if is_cluster
                        else (
                            "cactus_large"
                            if is_large
                            else "cactus_small"
                        )
                    ),
                    "x": self.arena_w + 10,
                    "y": self.ground_y - h,
                    "w": w,
                    "h": h,
                }
            )

        self.next_obstacle_dist = random.randint(110, 240) + max(
            40, int(200 - self.speed * 8)
        )

    def _get_current_game_state(self) -> DinoGameState:
        state = DinoGameState()
        state.ground_y = self.ground_y
        state.is_game_over = self.is_game_over
        state.is_night_mode = self.is_night
        state.game_speed_px_sec = self.speed * 60.0

        dino_state = (
            "jumping"
            if not self.is_grounded
            else ("ducking" if self.is_ducking else "running")
        )
        state.dino = DinoEntity(
            x=self.dino_x,
            y=int(self.dino_y),
            w=self.dino_w,
            h=self.dino_h,
            state=dino_state,
        )

        detected_obs: List[ObstacleEntity] = []
        dino_front = self.dino_x + self.dino_w
        for o in self.obstacles:
            dist = int(o["x"] - dino_front)
            if dist >= -10:
                time_ms = (dist / max(1.0, self.speed * 60.0)) * 1000.0
                detected_obs.append(
                    ObstacleEntity(
                        x=int(o["x"]),
                        y=int(o["y"]),
                        w=int(o["w"]),
                        h=int(o["h"]),
                        distance_from_dino=dist,
                        obstacle_type=o["type"],
                        time_to_impact_ms=time_ms,
                    )
                )

        detected_obs.sort(key=lambda o: o.distance_from_dino)
        state.obstacles = detected_obs
        if detected_obs:
            state.nearest_obstacle = detected_obs[0]
            dist = detected_obs[0].distance_from_dino
            jump_threshold = max(95, int(self.speed * 18.0))
            duck_threshold = max(110, int(self.speed * 20.0))

            if detected_obs[0].obstacle_type == "bird_high":
                state.recommended_action = "run_normal"
                state.threat_score = 0.05
                state.action_urgency = 0.0
            elif detected_obs[0].obstacle_type == "bird_mid":
                if dist <= duck_threshold:
                    state.recommended_action = "duck"
                    state.threat_score = 0.96
                    state.action_urgency = 1.0
                else:
                    state.recommended_action = "run_normal"
                    state.threat_score = 0.35
                    state.action_urgency = max(0.0, 1.0 - (dist / 280.0))
            else:
                if dist <= jump_threshold:
                    state.recommended_action = "jump"
                    state.threat_score = 0.98
                    state.action_urgency = 1.0
                else:
                    state.recommended_action = "run_normal"
                    state.threat_score = 0.30
                    state.action_urgency = max(0.0, 1.0 - (dist / 260.0))
        else:
            state.recommended_action = "run_normal"
            state.threat_score = 0.0
            state.action_urgency = 0.0

        return state

    def _tick(self):
        if not self.is_loop_running:
            return

        t0 = time.perf_counter()

        if not self.is_game_over:
            self.frame_count += 1
            self.score += 1
            if self.score > self.hi_score:
                self.hi_score = self.score

            if self.score % 600 == 0:
                self.is_night = not self.is_night
                self.configure(bg="#1c1d24" if self.is_night else "#ffffff")

            if self.score % 120 == 0 and self.speed < 13.5:
                self.speed += 0.20

            # Dino Physics
            if not self.is_grounded:
                self.dino_y += self.vy
                self.vy += self.gravity
                base_h = 26 if self.is_ducking else 44
                if self.dino_y >= self.ground_y - base_h:
                    self.dino_y = self.ground_y - base_h
                    self.vy = 0.0
                    self.is_grounded = True
            else:
                self.dino_h = 26 if self.is_ducking else 44
                self.dino_w = 52 if self.is_ducking else 40
                self.dino_y = self.ground_y - self.dino_h

            # Obstacles move left
            self.next_obstacle_dist -= self.speed
            if self.next_obstacle_dist <= 0:
                self._spawn_obstacle()

            for i in range(len(self.obstacles) - 1, -1, -1):
                obs = self.obstacles[i]
                obs["x"] -= self.speed

                # Collision box check
                hit_x = (self.dino_x + self.dino_w - 5 > obs["x"] + 4) and (
                    self.dino_x + 5 < obs["x"] + obs["w"] - 4
                )
                hit_y = (self.dino_y + self.dino_h - 4 > obs["y"] + 3) and (
                    self.dino_y + 4 < obs["y"] + obs["h"] - 3
                )

                if hit_x and hit_y:
                    self.is_game_over = True

                if obs["x"] + obs["w"] < -30:
                    self.obstacles.pop(i)

        # AI Autopilot Decision & Action
        game_state = self._get_current_game_state()
        decision = {
            "action": game_state.recommended_action,
            "threat_score": game_state.threat_score,
            "confidence": 0.98,
            "source": "jev_reflex_engine",
            "latency_ms": 0.5,
        }

        if self.autopilot_enabled:
            now = time.time()
            if self.is_game_over:
                if now - self.last_ai_action_time > 1.2:
                    self.reset_game()
                    self.last_ai_action_time = now
            else:
                if (
                    game_state.recommended_action == "jump"
                    and self.is_grounded
                    and now - self.last_ai_action_time > 0.18
                ):
                    self.jump()
                    self.last_ai_action_time = now
                elif (
                    game_state.recommended_action == "duck"
                    and now - self.last_ai_action_time > 0.12
                ):
                    self.duck(True)
                    # Automatically schedule duck release after obstacle passes
                    self.after(320, lambda: self.duck(False))
                    self.last_ai_action_time = now

        # Draw Canvas Frame
        self._render_scene(game_state)

        # Dispatch Telemetry to HUD
        if self.telemetry_cb:
            self.telemetry_cb(
                {
                    "fps": 60.0,
                    "state": game_state,
                    "decision": decision,
                    "jumps": self.total_jumps,
                    "ducks": self.total_ducks,
                    "score": self.score,
                    "hi_score": self.hi_score,
                    "is_armed": self.autopilot_enabled,
                }
            )

        # Next Frame (target ~60 FPS / 16ms)
        dt = time.perf_counter() - t0
        delay_ms = max(5, int(16 - (dt * 1000)))
        self.after(delay_ms, self._tick)

    def _render_scene(self, state: DinoGameState):
        self.delete("all")
        fg = "#e8eaed" if self.is_night else "#535353"
        accent = "#00ffcc" if self.is_night else "#1a73e8"

        # 1. Ground Plane
        self.create_line(
            0,
            self.ground_y,
            self.arena_w,
            self.ground_y,
            fill=fg,
            width=2,
        )

        # Ground texture ticks
        for i in range(0, self.arena_w, 45):
            ox = (i - int(self.frame_count * self.speed * 0.4)) % self.arena_w
            self.create_line(
                ox,
                self.ground_y + 4,
                ox + 12,
                self.ground_y + 4,
                fill=fg,
                width=1,
            )

        # 2. Draw Obstacles (with AI bounding boxes)
        for i, obs in enumerate(self.obstacles):
            is_nearest = i == 0
            box_outline = (
                "#ff2a55" if (is_nearest and self.autopilot_enabled) else fg
            )

            if "bird" in obs["type"]:
                # Draw Pterodactyl Bird
                self.create_rectangle(
                    obs["x"] + 8,
                    obs["y"] + 8,
                    obs["x"] + 32,
                    obs["y"] + 20,
                    fill=fg,
                    outline="",
                )
                self.create_polygon(
                    obs["x"],
                    obs["y"] + 14,
                    obs["x"] + 8,
                    obs["y"] + 10,
                    obs["x"] + 8,
                    obs["y"] + 18,
                    fill=fg,
                )
                # Wing
                wing_up = (self.frame_count // 7) % 2 == 0
                if wing_up:
                    self.create_polygon(
                        obs["x"] + 14,
                        obs["y"] + 8,
                        obs["x"] + 22,
                        obs["y"] - 8,
                        obs["x"] + 26,
                        obs["y"] + 8,
                        fill=fg,
                    )
                else:
                    self.create_polygon(
                        obs["x"] + 14,
                        obs["y"] + 20,
                        obs["x"] + 22,
                        obs["y"] + 30,
                        obs["x"] + 26,
                        obs["y"] + 20,
                        fill=fg,
                    )
            else:
                # Draw Cactus
                self.create_rectangle(
                    obs["x"] + (obs["w"] / 2) - 4,
                    obs["y"],
                    obs["x"] + (obs["w"] / 2) + 4,
                    obs["y"] + obs["h"],
                    fill=fg,
                    outline="",
                )
                self.create_rectangle(
                    obs["x"],
                    obs["y"] + 10,
                    obs["x"] + obs["w"],
                    obs["y"] + 16,
                    fill=fg,
                    outline="",
                )
                self.create_rectangle(
                    obs["x"],
                    obs["y"] + 4,
                    obs["x"] + 6,
                    obs["y"] + 16,
                    fill=fg,
                    outline="",
                )
                self.create_rectangle(
                    obs["x"] + obs["w"] - 6,
                    obs["y"] + 6,
                    obs["x"] + obs["w"],
                    obs["y"] + 16,
                    fill=fg,
                    outline="",
                )

            # AI Bounding Box overlay
            if self.autopilot_enabled:
                self.create_rectangle(
                    obs["x"] - 2,
                    obs["y"] - 2,
                    obs["x"] + obs["w"] + 2,
                    obs["y"] + obs["h"] + 2,
                    outline=box_outline,
                    width=2 if is_nearest else 1,
                    dash=(3, 2) if not is_nearest else (),
                )
                if is_nearest:
                    self.create_text(
                        obs["x"],
                        obs["y"] - 12,
                        text=f"{obs['type'].upper()} [{int(obs['x'] - (self.dino_x + self.dino_w))}px]",
                        fill="#ff2a55",
                        font=("Consolas", 8, "bold"),
                        anchor="w",
                    )

        # 3. Draw Dino
        d_outline = (
            "#00ffcc"
            if self.autopilot_enabled
            else ("#ff9900" if self.is_ducking else fg)
        )
        if self.is_ducking:
            self.create_rectangle(
                self.dino_x,
                self.dino_y + 6,
                self.dino_x + self.dino_w - 8,
                self.dino_y + self.dino_h,
                fill=fg,
                outline="",
            )
            self.create_rectangle(
                self.dino_x + 14,
                self.dino_y,
                self.dino_x + self.dino_w,
                self.dino_y + 12,
                fill=fg,
                outline="",
            )
        else:
            self.create_rectangle(
                self.dino_x + 8,
                self.dino_y + 12,
                self.dino_x + 32,
                self.dino_y + 34,
                fill=fg,
                outline="",
            )
            self.create_rectangle(
                self.dino_x + 16,
                self.dino_y,
                self.dino_x + 38,
                self.dino_y + 14,
                fill=fg,
                outline="",
            )
            self.create_rectangle(
                self.dino_x + 10,
                self.dino_y + 34,
                self.dino_x + 16,
                self.dino_y + 44,
                fill=fg,
                outline="",
            )
            self.create_rectangle(
                self.dino_x + 24,
                self.dino_y + 34,
                self.dino_x + 30,
                self.dino_y + 44,
                fill=fg,
                outline="",
            )
            self.create_rectangle(
                self.dino_x,
                self.dino_y + 16,
                self.dino_x + 10,
                self.dino_y + 24,
                fill=fg,
                outline="",
            )

        # AI Dino Tracking Box
        if self.autopilot_enabled:
            self.create_rectangle(
                self.dino_x - 3,
                self.dino_y - 3,
                self.dino_x + self.dino_w + 3,
                self.dino_y + self.dino_h + 3,
                outline="#00ffcc",
                width=2,
            )

        # 4. HUD Scores
        score_str = f"HI {str(self.hi_score).zfill(5)}  {str(self.score).zfill(5)}"
        self.create_text(
            self.arena_w - 15,
            20,
            text=score_str,
            fill=fg,
            font=("Consolas", 12, "bold"),
            anchor="e",
        )

        # AI Mode Tag
        if self.autopilot_enabled:
            self.create_text(
                15,
                20,
                text="🤖 JEV SYSTEM ONE AUTOPILOT ACTIVE",
                fill="#00ffcc",
                font=("Consolas", 10, "bold"),
                anchor="w",
            )
        else:
            self.create_text(
                15,
                20,
                text="🎮 MANUAL MODE (Click to focus, SPACE=Jump, DOWN=Duck)",
                fill="#888888",
                font=("Segoe UI", 9),
                anchor="w",
            )

        # 5. Game Over Screen
        if self.is_game_over:
            self.create_text(
                self.arena_w / 2,
                self.arena_h / 2 - 20,
                text="G A M E   O V E R",
                fill=fg,
                font=("Consolas", 18, "bold"),
            )
            self.create_text(
                self.arena_w / 2,
                self.arena_h / 2 + 15,
                text="🔄 AI Auto-Restarting..."
                if self.autopilot_enabled
                else "Press SPACE to restart",
                fill="#ff9900" if self.autopilot_enabled else fg,
                font=("Segoe UI", 11, "bold"),
            )
