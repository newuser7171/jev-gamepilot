"""
Android Phone Game Adapter for Jev-GamePilot & Laya.
Connects directly to an Android phone over USB or Wi-Fi via ADB:
1. High-speed real-time screen capture with continuous background buffer.
2. Auto-detection of active foreground game package (Subway Surfers, Fruit Ninja, Earn to Die, etc.).
3. Microsecond gesture and touch injection (swipe left/right/up/down, slice, throttle, tap).
4. Connects phone video frames to Universal Vision and Laya System One Brain.
"""

import os
import re
import sys
import time
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

# Common ADB paths on Windows
POSSIBLE_ADB_PATHS = [
    r"C:\Users\newuser\Downloads\platform-tools-latest-windows\platform-tools\adb.exe",
    os.path.expanduser(r"~\Downloads\platform-tools-latest-windows\platform-tools\adb.exe"),
    "adb",
    os.path.expanduser(r"~\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
]

# Package to profile ID mapping
PACKAGE_PROFILE_MAP = {
    "com.kiloo.subwaysurf": "runner_3lane",
    "com.halfbrick.fruitninjafree": "mobile_fruit_ninja",
    "com.notdoppler.earntodie2": "mobile_earntodie2",
    "com.supercell.clashroyale": "mobile_clash_royale",
    "com.ea.gp.fifamobile": "mobile_fifa",
    "com.paradyme.solarsmash": "mobile_solarsmash",
    "com.candywriter.bitlife": "mobile_bitlife",
    "com.miniclip.plagueinc": "chess_copilot",
    "ru.playsoftware.j2meloader": "retro_platformer",
    "xyz.aethersx2.android": "mobile_universal",
    "org.shadps4.android": "mobile_universal",
    "dev.eden.eden_emulator": "mobile_universal",
    "com.izzy2lost.psx2": "retro_platformer",
}


def find_adb() -> str:
    """Locates the working adb.exe binary."""
    for p in POSSIBLE_ADB_PATHS:
        try:
            res = subprocess.run([p, "version"], capture_output=True, text=True, timeout=3)
            if res.returncode == 0:
                return p
        except Exception:
            continue
    return "adb"


class AdbController:
    def __init__(self, device_serial: Optional[str] = None):
        self.adb_bin = find_adb()
        self.device_serial = device_serial
        self.screen_width = 1080
        self.screen_height = 2400
        self.is_connected = False

        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_id = 0
        self._new_frame_event = threading.Event()
        self._streaming = False
        self._stream_thread: Optional[threading.Thread] = None

        self._check_connection()

    def _cmd_prefix(self) -> List[str]:
        if self.device_serial:
            return [self.adb_bin, "-s", self.device_serial]
        return [self.adb_bin]

    def _check_connection(self):
        devices = self.get_devices()
        if devices:
            if not self.device_serial:
                self.device_serial = devices[0]
            self.is_connected = True
            self.query_screen_size()
        else:
            self.is_connected = False

    def get_devices(self) -> List[str]:
        """Returns list of connected Android device serial numbers."""
        try:
            res = subprocess.run(
                [self.adb_bin, "devices"],
                capture_output=True,
                text=True,
                timeout=6,
            )
            devices = []
            for line in res.stdout.strip().splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    devices.append(parts[0])
            return devices
        except Exception:
            return []

    def detect_foreground_package(self) -> Optional[str]:
        """Detects the package name of the active app running on phone screen."""
        try:
            cmd = self._cmd_prefix() + ["shell", "dumpsys", "window", "displays"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            match = re.search(r"mFocusedApp=ActivityRecord\{[^\s]+ [^\s]+ ([^/\s]+)", res.stdout)
            if match:
                return match.group(1)
            # Fallback check
            match_focus = re.search(r"mCurrentFocus=Window\{[^\s]+ [^\s]+ ([^/\s]+)", res.stdout)
            if match_focus:
                return match_focus.group(1)
        except Exception:
            pass
        return None

    def auto_detect_game_profile(self) -> Tuple[str, str]:
        """
        Auto-detects the active game and maps to the optimal GamePilot profile.
        Returns: (profile_id, package_name)
        """
        pkg = self.detect_foreground_package()
        if not pkg:
            return "mobile_universal", "Unknown"

        if pkg in PACKAGE_PROFILE_MAP:
            return PACKAGE_PROFILE_MAP[pkg], pkg

        # Heuristic keywords
        pkg_lower = pkg.lower()
        if "surf" in pkg_lower or "runner" in pkg_lower or "temple" in pkg_lower or "dash" in pkg_lower:
            return "runner_3lane", pkg
        elif "fruit" in pkg_lower or "ninja" in pkg_lower or "slice" in pkg_lower or "cut" in pkg_lower:
            return "mobile_fruit_ninja", pkg
        elif "flap" in pkg_lower or "bird" in pkg_lower or "copter" in pkg_lower:
            return "flappy_tap", pkg
        elif "drive" in pkg_lower or "hill" in pkg_lower or "race" in pkg_lower or "car" in pkg_lower:
            return "mobile_earntodie2", pkg
        elif "clash" in pkg_lower or "royale" in pkg_lower or "arena" in pkg_lower or "brawl" in pkg_lower or "tower" in pkg_lower:
            return "mobile_clash_royale", pkg
        elif "fifa" in pkg_lower or "football" in pkg_lower or "soccer" in pkg_lower or "nba" in pkg_lower or "pes" in pkg_lower:
            return "mobile_fifa", pkg
        elif "smash" in pkg_lower or "solar" in pkg_lower or "sandbox" in pkg_lower or "destroy" in pkg_lower:
            return "mobile_solarsmash", pkg
        elif "bitlife" in pkg_lower or "life" in pkg_lower or "sim" in pkg_lower or "choice" in pkg_lower:
            return "mobile_bitlife", pkg

        return "mobile_universal", pkg

    def query_screen_size(self) -> Tuple[int, int]:
        """Queries physical display resolution via adb shell wm size."""
        try:
            cmd = self._cmd_prefix() + ["shell", "wm", "size"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            for line in res.stdout.splitlines():
                if "size:" in line.lower():
                    dim = line.split(":")[-1].strip()
                    w, h = dim.split("x")
                    self.screen_width = int(w)
                    self.screen_height = int(h)
                    return self.screen_width, self.screen_height
        except Exception:
            pass
        return self.screen_width, self.screen_height

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Captures raw screenshot directly from phone into OpenCV BGR numpy array.
        Uses binary pipe for lowest latency:
        1. Fast raw RGBA framebuffer (fastest, zero CPU compression overhead).
        2. Fallback PNG screencap with 5s timeout and CRLF normalization.
        """
        # 1. Fast Raw RGBA framebuffer stream
        try:
            cmd = self._cmd_prefix() + ["exec-out", "screencap"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            raw_bytes, _ = proc.communicate(timeout=4.5)
            if len(raw_bytes) >= 16:
                w = int.from_bytes(raw_bytes[0:4], byteorder="little")
                h = int.from_bytes(raw_bytes[4:8], byteorder="little")
                expected = 16 + w * h * 4
                if len(raw_bytes) >= expected and w > 0 and h > 0:
                    rgba = np.frombuffer(raw_bytes[16:expected], dtype=np.uint8).reshape((h, w, 4))
                    frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._frame_id += 1
                    self._new_frame_event.set()
                    return frame
        except Exception:
            pass

        # 2. Fallback PNG stream
        try:
            cmd = self._cmd_prefix() + ["exec-out", "screencap", "-p"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            raw_bytes, _ = proc.communicate(timeout=5.0)
            if raw_bytes:
                img_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                if frame is None and b"\r\n" in raw_bytes:
                    clean = raw_bytes.replace(b"\r\n", b"\n")
                    frame = cv2.imdecode(np.frombuffer(clean, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._frame_id += 1
                    self._new_frame_event.set()
                    return frame
        except Exception:
            pass

        return None

    def start_background_stream(self):
        """Starts continuous non-blocking frame capture worker."""
        if self._streaming:
            return
        self._streaming = True

        def worker():
            while self._streaming:
                self.capture_frame()
                time.sleep(0.01)

        self._stream_thread = threading.Thread(target=worker, daemon=True, name="AdbFrameStreamer")
        self._stream_thread.start()

    def stop_background_stream(self):
        self._streaming = False

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame

    def get_fresh_frame(self, timeout: float = 2.5) -> Optional[np.ndarray]:
        """
        Blocks until a brand-new frame is captured from the phone.
        Guarantees the pilot NEVER runs twice on the same screenshot!
        """
        if self._new_frame_event.wait(timeout):
            self._new_frame_event.clear()
            with self._frame_lock:
                return self._latest_frame
        # Fallback to direct capture if stream paused
        return self.capture_frame()

    # --- Touch & Gesture Dispatchers ---
    def tap(self, x: int, y: int):
        """Sends immediate tap to phone screen coordinates."""
        cmd = self._cmd_prefix() + ["shell", "input", "tap", str(int(x)), str(int(y))]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 100):
        """Sends swipe gesture across screen coordinates."""
        cmd = self._cmd_prefix() + [
            "shell",
            "input",
            "swipe",
            str(int(x1)),
            str(int(y1)),
            str(int(x2)),
            str(int(y2)),
            str(int(duration_ms)),
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def swipe_up(self, duration_ms: int = 80):
        """Vault / Jump gesture (swipes upward from lower center)."""
        cx = self.screen_width // 2
        y1 = int(self.screen_height * 0.68)
        y2 = int(self.screen_height * 0.32)
        self.swipe(cx, y1, cx, y2, duration_ms)

    def swipe_down(self, duration_ms: int = 80):
        """Slide / Duck gesture (swipes downward from upper center)."""
        cx = self.screen_width // 2
        y1 = int(self.screen_height * 0.32)
        y2 = int(self.screen_height * 0.68)
        self.swipe(cx, y1, cx, y2, duration_ms)

    def swipe_left(self, duration_ms: int = 70):
        """Dodge left gesture (swipes right to left)."""
        y = int(self.screen_height * 0.55)
        x1 = int(self.screen_width * 0.82)
        x2 = int(self.screen_width * 0.18)
        self.swipe(x1, y1=y, x2=x2, y2=y, duration_ms=duration_ms)

    def swipe_right(self, duration_ms: int = 70):
        """Dodge right gesture (swipes left to right)."""
        y = int(self.screen_height * 0.55)
        x1 = int(self.screen_width * 0.18)
        x2 = int(self.screen_width * 0.82)
        self.swipe(x1, y1=y, x2=x2, y2=y, duration_ms=duration_ms)

    def slice_target(self, target_x: int, target_y: int):
        """Fruit Ninja fast diagonal slice through target centroid."""
        x1 = max(20, target_x - 120)
        y1 = min(self.screen_height - 20, target_y + 100)
        x2 = min(self.screen_width - 20, target_x + 120)
        y2 = max(20, target_y - 100)
        self.swipe(x1, y1, x2, y2, duration_ms=45)

    def deploy_clash_card(self, card_x: int, card_y: int, target_x: int, target_y: int):
        """Native two-tap deployment: select card in tray then deploy at arena tile."""
        shell_script = f"input tap {int(card_x)} {int(card_y)} && sleep 0.08 && input tap {int(target_x)} {int(target_y)}"
        cmd = self._cmd_prefix() + ["shell", shell_script]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def trigger_auto_restart(self):
        """Taps the center or lower-center screen to dismiss Game Over / Play Again dialogs."""
        cx = self.screen_width // 2
        cy = int(self.screen_height * 0.75)
        self.tap(cx, cy)

    def dispatch_action(
        self,
        action_name: str,
        target_coords: Optional[Tuple[int, int]] = None,
    ):
        """Maps game action names into physical Android touch gestures."""
        act = action_name.lower().strip()

        # 1. 3-Lane Runner actions (Subway Surfers, Temple Run)
        if "left" in act and "tilt" not in act and "card" not in act:
            self.swipe_left()
        elif "right" in act and "tilt" not in act and "card" not in act:
            self.swipe_right()
        elif "jump" in act or "vault" in act or act == "up":
            self.swipe_up()
        elif "slide" in act or "duck" in act or act == "down":
            self.swipe_down()

        # 2. Fruit Ninja Slice actions
        elif "slice" in act:
            if target_coords:
                self.slice_target(target_coords[0], target_coords[1])
            else:
                self.slice_target(self.screen_width // 2, self.screen_height // 2)

        # 3. Precision tap-to-click target (Aim trainers, Fruit Ninja, Solar Smash)
        elif "tap_target" in act or "click_target" in act:
            tx = target_coords[0] if target_coords else self.screen_width // 2
            ty = target_coords[1] if target_coords else self.screen_height // 2
            self.tap(tx, ty)

        # 4. Tap / One-tap actions (Flappy Bird, Geometry Dash, Universal Taps)
        elif "flap" in act or "tap" in act or "jump_tap" in act:
            cx = self.screen_width // 2
            cy = int(self.screen_height * 0.65)
            self.tap(cx, cy)

        # 5. Clash Royale / RTS card deployment & auto-match queue
        elif "start_battle" in act:
            # Tap yellow Battle button on main menu
            self.tap(int(self.screen_width * 0.505), int(self.screen_height * 0.814))
        elif "confirm_ok" in act or "open_chest" in act:
            # Tap OK button or center to dismiss post-game screen or collect rewards
            self.tap(int(self.screen_width * 0.50), int(self.screen_height * 0.83))
            self.tap(int(self.screen_width * 0.50), int(self.screen_height * 0.50))
        elif "deploy_card_left" in act or "deploy_card_right" in act or "deploy_spell_center" in act:
            card_slots_x = [
                int(self.screen_width * 0.287),  # Slot 1: ~310
                int(self.screen_width * 0.463),  # Slot 2: ~500
                int(self.screen_width * 0.634),  # Slot 3: ~685
                int(self.screen_width * 0.806),  # Slot 4: ~870
            ]
            self._clash_card_idx = (getattr(self, "_clash_card_idx", 0) + 1) % 4
            card_x = card_slots_x[self._clash_card_idx]
            card_y = int(self.screen_height * 0.915)  # ~2140 (exact vertical card center)

            if "deploy_card_left" in act:
                target_x = int(self.screen_width * 0.194)  # ~210 left bridge mouth
                target_y = int(self.screen_height * 0.490)  # ~1150
            elif "deploy_card_right" in act:
                target_x = int(self.screen_width * 0.731)  # ~790 right bridge mouth
                target_y = int(self.screen_height * 0.490)  # ~1150
            else:  # deploy_spell_center
                # Direct damage spells (Fireball / Arrows) onto enemy Princess Towers
                self._clash_spell_idx = (getattr(self, "_clash_spell_idx", 0) + 1) % 2
                target_x = int(self.screen_width * (0.194 if self._clash_spell_idx == 0 else 0.741))  # 210 or 800
                target_y = int(self.screen_height * 0.235)  # ~550 (enemy Princess Tower)

            # Execute fast chained native card deployment
            self.deploy_clash_card(card_x, card_y, target_x, target_y)

        # 6. Vehicle Driver actions (Earn to Die 2)
        elif "accelerate" in act or "gas" in act:
            gx = int(self.screen_width * 0.84)
            gy = int(self.screen_height * 0.82)
            self.tap(gx, gy)
        elif "boost" in act or "nitro" in act:
            bx = int(self.screen_width * 0.16)
            by = int(self.screen_height * 0.82)
            self.tap(bx, by)
        elif "tilt_forward" in act:
            tx = int(self.screen_width * 0.70)
            ty = int(self.screen_height * 0.82)
            self.tap(tx, ty)
        elif "tilt_back" in act:
            tx = int(self.screen_width * 0.30)
            ty = int(self.screen_height * 0.82)
            self.tap(tx, ty)

        # 7. EA Sports FC / FIFA Mobile
        elif "sprint_tackle" in act:
            sx = int(self.screen_width * 0.88)
            sy = int(self.screen_height * 0.84)
            self.tap(sx, sy)
        elif "through_pass" in act:
            tx = int(self.screen_width * 0.76)
            ty = int(self.screen_height * 0.72)
            self.tap(tx, ty)
        elif "shoot_goal" in act:
            sx = int(self.screen_width * 0.88)
            sy = int(self.screen_height * 0.70)
            self.tap(sx, sy)
        elif "pass" in act:
            px = int(self.screen_width * 0.76)
            py = int(self.screen_height * 0.88)
            self.tap(px, py)

        # 7. Solar Smash
        elif "fire_laser" in act or "orbital_strike" in act or "launch_meteor" in act:
            # Tap weapon drawer on right edge, then tap planet center
            wx = int(self.screen_width * 0.93)
            wy = int(self.screen_height * 0.35)
            self.tap(wx, wy)
            time.sleep(0.03)
            cx = self.screen_width // 2
            cy = self.screen_height // 2
            self.tap(cx, cy)
        elif "rotate_planet" in act:
            y = self.screen_height // 2
            x1 = int(self.screen_width * 0.75)
            x2 = int(self.screen_width * 0.25)
            self.swipe(x1, y, x2, y, duration_ms=120)

        # 8. BitLife Simulation
        elif "age_up" in act:
            ax = int(self.screen_width * 0.50)
            ay = int(self.screen_height * 0.69)
            self.tap(ax, ay)
        elif "primary_choice" in act:
            cx = int(self.screen_width * 0.50)
            cy = int(self.screen_height * 0.58)
            self.tap(cx, cy)
        elif "secondary_choice" in act:
            cx = int(self.screen_width * 0.50)
            cy = int(self.screen_height * 0.66)
            self.tap(cx, cy)
        elif "dismiss_popup" in act:
            cx = int(self.screen_width * 0.50)
            cy = int(self.screen_height * 0.72)
            self.tap(cx, cy)

        # 9. Precision target clicker
        elif "click" in act or "target" in act:
            if target_coords:
                self.tap(target_coords[0], target_coords[1])
            else:
                cx = self.screen_width // 2
                cy = self.screen_height // 2
                self.tap(cx, cy)

        # 10. Auto-revive / restart / continue
        elif "restart" in act or "revive" in act or "continue" in act:
            self.trigger_auto_restart()
