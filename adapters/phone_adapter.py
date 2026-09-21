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
    # Mobile Card Battlers, TCGs & Deckbuilders
    "com.nvsgames.snap": "mobile_card_battler",
    "jp.pokemon.pokemontcgp": "mobile_card_battler",
    "com.pokemon.tcgl": "mobile_card_battler",
    "com.blizzard.wtcg.hearthstone": "mobile_card_battler",
    "jp.konami.masterduel": "mobile_card_battler",
    "jp.konami.duellinks": "mobile_card_battler",
    "com.wizards.mtga": "mobile_card_battler",
    "com.riotgames.legendsofruneterra": "mobile_card_battler",
    "com.humble.SlayTheSpire": "mobile_card_battler",
    "com.localthunk.balatro": "mobile_card_battler",
    "com.mattel163.uno": "mobile_card_battler",
    # Solitaire & Card Puzzles
    "com.mobilityware.solitaire": "mobile_solitaire",
    "com.pie.solitaire": "mobile_solitaire",
    "com.microsoft.solitaire.collection": "mobile_solitaire",
    "com.brainium.solitaire": "mobile_solitaire",
    "com.tripledot.solitaire": "mobile_solitaire",
    "com.zynga.solitaire": "mobile_solitaire",
    "com.me2star.solitaire": "mobile_solitaire",
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
        self.is_landscape = False
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
        elif "snake" in pkg_lower or "slither" in pkg_lower or "worm" in pkg_lower or "nibbles" in pkg_lower:
            return "mobile_snake", pkg
        elif (
            "solitaire" in pkg_lower
            or "klondike" in pkg_lower
            or "spider" in pkg_lower
            or "freecell" in pkg_lower
            or "patience" in pkg_lower
        ):
            return "mobile_solitaire", pkg
        elif (
            "snap" in pkg_lower
            or "card" in pkg_lower
            or "tcg" in pkg_lower
            or "hearthstone" in pkg_lower
            or "duel" in pkg_lower
            or "magic" in pkg_lower
            or "mtg" in pkg_lower
            or "deck" in pkg_lower
            or "spire" in pkg_lower
            or "balatro" in pkg_lower
            or "uno" in pkg_lower
            or "poker" in pkg_lower
        ):
            return "mobile_card_battler", pkg

        return "mobile_universal", pkg

    def _sync_screen_dimensions(self, frame: np.ndarray):
        """Dynamically syncs logical touch coordinate boundaries to active video frame orientation."""
        fh, fw = frame.shape[:2]
        if fw != self.screen_width or fh != self.screen_height:
            self.screen_width = fw
            self.screen_height = fh
            self.is_landscape = (fw > fh)

    def query_screen_size(self) -> Tuple[int, int]:
        """Queries active display resolution and orientation via dumpsys display & wm size."""
        try:
            # 1. Inspect active override display info (reports real logical dimensions in landscape/portrait)
            cmd = self._cmd_prefix() + ["shell", "dumpsys", "display"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            match_override = re.search(r"mOverrideDisplayInfo=DisplayInfo\{.*?real\s+(\d+)\s+x\s+(\d+)", res.stdout)
            if match_override:
                w, h = int(match_override.group(1)), int(match_override.group(2))
                self.screen_width = w
                self.screen_height = h
                self.is_landscape = (w > h)
                return self.screen_width, self.screen_height

            # 2. Check input viewports
            cmd_in = self._cmd_prefix() + ["shell", "dumpsys", "input"]
            res_in = subprocess.run(cmd_in, capture_output=True, text=True, timeout=2)
            match_vp = re.search(r"DisplayViewport\[id=0\]\s+Width=(\d+),\s+Height=(\d+)", res_in.stdout)
            if match_vp:
                w, h = int(match_vp.group(1)), int(match_vp.group(2))
                self.screen_width = w
                self.screen_height = h
                self.is_landscape = (w > h)
                return self.screen_width, self.screen_height

            # 3. Fallback to wm size
            cmd_wm = self._cmd_prefix() + ["shell", "wm", "size"]
            res_wm = subprocess.run(cmd_wm, capture_output=True, text=True, timeout=2)
            for line in res_wm.stdout.splitlines():
                if "size:" in line.lower():
                    dim = line.split(":")[-1].strip()
                    pw, ph = dim.split("x")
                    self.screen_width = int(pw)
                    self.screen_height = int(ph)
                    self.is_landscape = (self.screen_width > self.screen_height)
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
        Automatically synchronizes touch screen coordinates to active orientation!
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
                    self._sync_screen_dimensions(frame)
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
                    self._sync_screen_dimensions(frame)
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
    def _run_shell(self, shell_cmd: str, timeout: float = 3.5):
        """Executes shell command synchronously to guarantee atomic, collision-free touch events on device."""
        try:
            cmd = self._cmd_prefix() + ["shell", shell_cmd]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"ADB input failed: {exc}") from exc
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
            raise RuntimeError(f"ADB input failed: {detail}")

    def tap(self, x: int, y: int):
        """Sends immediate tap to phone screen coordinates."""
        self._run_shell(f"input tap {int(x)} {int(y)}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 120):
        """Sends swipe gesture across screen coordinates."""
        self._run_shell(f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(duration_ms)}")

    def hold(self, x: int, y: int, duration_ms: int = 350):
        """Sends press-and-hold touch gesture for sustained gas throttle, power shots, or continuous beams."""
        self.swipe(x, y, x, y, duration_ms=duration_ms)

    def double_tap(self, x: int, y: int, delay_sec: float = 0.08):
        """Chains rapid double-tap (activates hoverboard shield, nitro bursts, revives)."""
        shell_script = f"input tap {int(x)} {int(y)} && sleep {delay_sec} && input tap {int(x)} {int(y)}"
        self._run_shell(shell_script)

    def flick(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 65):
        """Rapid directional skill flick (finesse curl, chip shot, 5-star skill moves)."""
        self.swipe(x1, y1, x2, y2, duration_ms=duration_ms)

    def chained_taps(self, points: List[Tuple[int, int]], delay_sec: float = 0.05):
        """Executes a cluster of precision taps across target points in a single kernel batch."""
        parts = [f"input tap {int(pt[0])} {int(pt[1])}" for pt in points]
        shell_script = f" && sleep {delay_sec} && ".join(parts)
        self._run_shell(shell_script)

    def swipe_up(self, duration_ms: int = 80):
        """Vault / Jump gesture (swipes upward from lower center)."""
        cx = self.screen_width // 2
        if self.is_landscape:
            y1 = int(self.screen_height * 0.72)
            y2 = int(self.screen_height * 0.28)
        else:
            y1 = int(self.screen_height * 0.68)
            y2 = int(self.screen_height * 0.32)
        self.swipe(cx, y1, cx, y2, duration_ms)

    def swipe_down(self, duration_ms: int = 80):
        """Slide / Duck gesture (swipes downward from upper center)."""
        cx = self.screen_width // 2
        if self.is_landscape:
            y1 = int(self.screen_height * 0.28)
            y2 = int(self.screen_height * 0.72)
        else:
            y1 = int(self.screen_height * 0.32)
            y2 = int(self.screen_height * 0.68)
        self.swipe(cx, y1, cx, y2, duration_ms)

    def swipe_left(self, duration_ms: int = 70):
        """Dodge left gesture (swipes right to left)."""
        y = int(self.screen_height * 0.50) if self.is_landscape else int(self.screen_height * 0.55)
        x1 = int(self.screen_width * 0.75) if self.is_landscape else int(self.screen_width * 0.82)
        x2 = int(self.screen_width * 0.25) if self.is_landscape else int(self.screen_width * 0.18)
        self.swipe(x1, y1=y, x2=x2, y2=y, duration_ms=duration_ms)

    def swipe_right(self, duration_ms: int = 70):
        """Dodge right gesture (swipes left to right)."""
        y = int(self.screen_height * 0.50) if self.is_landscape else int(self.screen_height * 0.55)
        x1 = int(self.screen_width * 0.25) if self.is_landscape else int(self.screen_width * 0.18)
        x2 = int(self.screen_width * 0.75) if self.is_landscape else int(self.screen_width * 0.82)
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
        self._run_shell(shell_script)

    def play_card(self, hand_slot: int = 0, target_lane: str = "center"):
        """
        Executes native drag-to-play gesture from hand tray onto the battlefield or lane.
        Dynamically adapts geometry for Portrait (Marvel SNAP, Pokémon Pocket) and Landscape (Hearthstone, MTG, Balatro).
        """
        if self.is_landscape:
            # Landscape Card Games (Hearthstone, MTG Arena, Yu-Gi-Oh, Balatro, Slay the Spire)
            slot_ratios = [0.32, 0.40, 0.48, 0.56, 0.64, 0.72]
            slot_x = slot_ratios[hand_slot % len(slot_ratios)]
            start_x = int(self.screen_width * slot_x)
            start_y = int(self.screen_height * 0.93)

            if target_lane == "left":
                dest_x = int(self.screen_width * 0.38)
                dest_y = int(self.screen_height * 0.58)
            elif target_lane == "right":
                dest_x = int(self.screen_width * 0.62)
                dest_y = int(self.screen_height * 0.58)
            else:  # center
                dest_x = int(self.screen_width * 0.50)
                dest_y = int(self.screen_height * 0.58)
        else:
            # Portrait Card Games (Marvel SNAP, Pokémon TCG Pocket, Solitaire)
            slot_ratios = [0.22, 0.40, 0.60, 0.78]
            slot_x = slot_ratios[hand_slot % len(slot_ratios)]
            start_x = int(self.screen_width * slot_x)
            start_y = int(self.screen_height * 0.88)

            if target_lane == "left":
                dest_x = int(self.screen_width * 0.23)
                dest_y = int(self.screen_height * 0.52)
            elif target_lane == "right":
                dest_x = int(self.screen_width * 0.77)
                dest_y = int(self.screen_height * 0.52)
            else:  # center
                dest_x = int(self.screen_width * 0.50)
                dest_y = int(self.screen_height * 0.52)

        # Smooth drag from hand into lane + tap confirm
        self.swipe(start_x, start_y, dest_x, dest_y, duration_ms=320)

    def attack_target(self, target_type: str = "face"):
        """
        Executes minion combat arrow: drags from friendly minion position to target.
        """
        if self.is_landscape:
            fx = int(self.screen_width * 0.50)
            fy = int(self.screen_height * 0.60)
            if target_type == "face":
                tx = int(self.screen_width * 0.50)
                ty = int(self.screen_height * 0.18)
            else:
                tx = int(self.screen_width * 0.50)
                ty = int(self.screen_height * 0.38)
            self.swipe(fx, fy, tx, ty, duration_ms=260)
        else:
            fx = int(self.screen_width * 0.50)
            fy = int(self.screen_height * 0.60)
            tx = int(self.screen_width * 0.50)
            ty = int(self.screen_height * 0.28)
            self.swipe(fx, fy, tx, ty, duration_ms=260)

    def tap_solitaire_stock(self):
        """Taps top stock pile to deal next card(s)."""
        if self.is_landscape:
            sx = int(self.screen_width * 0.08)
            sy = int(self.screen_height * 0.20)
        else:
            sx = int(self.screen_width * 0.12)
            sy = int(self.screen_height * 0.125)
        self.tap(sx, sy)

    def tap_solitaire_waste(self):
        """Taps waste pile card to auto-move to foundation or build onto tableau."""
        if self.is_landscape:
            wx = int(self.screen_width * 0.08)
            wy = int(self.screen_height * 0.45)
        else:
            wx = int(self.screen_width * 0.25)
            wy = int(self.screen_height * 0.125)
        self.tap(wx, wy)

    def tap_solitaire_column(self, col_idx: int = 0, y_ratio: float = 0.50):
        """
        Taps exposed card in tableau column (0-6).
        Mobile solitaire has built-in tap-to-move which automatically sends valid cards to foundations or tableau!
        Executes an atomic 2-depth cascade tap to reliably hit short (1-2 cards) or deep (5-8 cards) stacks.
        """
        col_idx = max(0, min(6, col_idx))
        if self.is_landscape:
            col_ratios = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
            cx = int(self.screen_width * col_ratios[col_idx])
            y1 = int(self.screen_height * 0.45)
            y2 = int(self.screen_height * 0.65)
        else:
            col_ratios = [0.10, 0.23, 0.37, 0.50, 0.63, 0.77, 0.90]
            cx = int(self.screen_width * col_ratios[col_idx])
            y1 = int(self.screen_height * 0.38)
            y2 = int(self.screen_height * 0.56)
        shell_script = f"input tap {cx} {y1} && sleep 0.05 && input tap {cx} {y2}"
        self._run_shell(shell_script)

    def drag_solitaire_transfer(self, from_col: int = 0, to_col: int = 1):
        """Drags card/sequence between two tableau columns."""
        from_col = max(0, min(6, from_col))
        to_col = max(0, min(6, to_col))
        if self.is_landscape:
            col_ratios = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
            fx = int(self.screen_width * col_ratios[from_col])
            tx = int(self.screen_width * col_ratios[to_col])
            y = int(self.screen_height * 0.55)
        else:
            col_ratios = [0.10, 0.23, 0.37, 0.50, 0.63, 0.77, 0.90]
            fx = int(self.screen_width * col_ratios[from_col])
            tx = int(self.screen_width * col_ratios[to_col])
            y = int(self.screen_height * 0.52)
        self.swipe(fx, y, tx, y, duration_ms=260)

    def tap_solitaire_foundation(self, found_idx: int = 0):
        """Taps one of the 4 foundation piles (Aces to Kings)."""
        found_idx = max(0, min(3, found_idx))
        if self.is_landscape:
            fx = int(self.screen_width * 0.92)
            y_ratios = [0.18, 0.38, 0.58, 0.78]
            fy = int(self.screen_height * y_ratios[found_idx])
        else:
            x_ratios = [0.53, 0.66, 0.79, 0.92]
            fx = int(self.screen_width * x_ratios[found_idx])
            fy = int(self.screen_height * 0.125)
        self.tap(fx, fy)

    def sweep_solitaire_tableau(self):
        """
        Executes a swift, chained multi-tap sweep across all 7 tableau columns.
        Triggers instant cascade of available auto-moves!
        """
        if self.is_landscape:
            col_ratios = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
            y = int(self.screen_height * 0.55)
        else:
            col_ratios = [0.10, 0.23, 0.37, 0.50, 0.63, 0.77, 0.90]
            y = int(self.screen_height * 0.52)
        pts = [(int(self.screen_width * r), y) for r in col_ratios]
        self.chained_taps(pts, delay_sec=0.06)

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
        if "hoverboard" in act or "shield" in act:
            # Double tap screen center to activate invincibility hoverboard
            self.double_tap(self.screen_width // 2, int(self.screen_height * 0.50))
        elif "fast_fall" in act or "cancel_jump" in act:
            # Immediate downward swipe to cancel jump arc and slide
            self.swipe_down(duration_ms=50)
        elif "left" in act and "tilt" not in act and "card" not in act:
            self.swipe_left()
        elif "right" in act and "tilt" not in act and "card" not in act:
            self.swipe_right()
        elif act in {"jump", "vault", "up", "swipe_up", "turn_up", "snake_up"}:
            self.swipe_up()
        elif "slide" in act or "duck" in act or "down" in act:
            self.swipe_down()

        # 2. Fruit Ninja Slice actions
        elif "combo_slice" in act or "wide_slice" in act:
            x1 = int(self.screen_width * 0.15)
            y1 = int(self.screen_height * 0.75)
            x2 = int(self.screen_width * 0.85)
            y2 = int(self.screen_height * 0.35)
            self.flick(x1, y1, x2, y2, duration_ms=55)
        elif "slice" in act:
            if target_coords:
                self.slice_target(target_coords[0], target_coords[1])
            else:
                x1 = int(self.screen_width * 0.20)
                y1 = int(self.screen_height * 0.72)
                x2 = int(self.screen_width * 0.80)
                y2 = int(self.screen_height * 0.42)
                self.flick(x1, y1, x2, y2, duration_ms=60)

        # 3. Precision tap-to-click target (Aim trainers, Fruit Ninja, Solar Smash)
        elif "tap_target" in act or "click_target" in act:
            tx = target_coords[0] if target_coords else self.screen_width // 2
            ty = target_coords[1] if target_coords else self.screen_height // 2
            self.tap(tx, ty)

        # 4. Tap / One-tap actions (Flappy Bird, Geometry Dash, Universal Taps)
        elif "double_flap" in act or "double_jump" in act:
            cx = self.screen_width // 2
            cy = int(self.screen_height * 0.65)
            self.double_tap(cx, cy, delay_sec=0.08)
        elif act in {"flap", "tap", "jump_tap"}:
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
        elif "deploy_clash" in act or "deploy_defense_center" in act or "deploy_card_left" in act or "deploy_card_right" in act or "deploy_spell_center" in act:
            if target_coords and len(target_coords) == 4:
                card_x, card_y, target_x, target_y = target_coords
                self.deploy_clash_card(card_x, card_y, target_x, target_y)
            elif "deploy_defense_center" in act:
                # Golden pocket plant between both Princess Towers to lure both lanes
                card_slots_x = [int(self.screen_width * x) for x in [0.287, 0.463, 0.634, 0.806]]
                self._clash_card_idx = (getattr(self, "_clash_card_idx", 0) + 1) % 4
                self.deploy_clash_card(card_slots_x[self._clash_card_idx], int(self.screen_height * 0.915), int(self.screen_width * 0.505), int(self.screen_height * 0.575))
            else:
                card_slots_x = [
                    int(self.screen_width * 0.287),  # Slot 1: ~310
                    int(self.screen_width * 0.463),  # Slot 2: ~500
                    int(self.screen_width * 0.634),  # Slot 3: ~685
                    int(self.screen_width * 0.806),  # Slot 4: ~870
                ]
                self._clash_card_idx = (getattr(self, "_clash_card_idx", 0) + 1) % 4
                card_x = card_slots_x[self._clash_card_idx]
                card_y = int(self.screen_height * 0.915)  # ~2140

                if "deploy_card_left" in act:
                    target_x = int(self.screen_width * 0.194)  # Left bridge mouth
                    target_y = int(self.screen_height * 0.490)
                elif "deploy_card_right" in act:
                    target_x = int(self.screen_width * 0.731)  # Right bridge mouth
                    target_y = int(self.screen_height * 0.490)
                elif "deploy_spell_center" in act:
                    self._clash_spell_idx = (getattr(self, "_clash_spell_idx", 0) + 1) % 2
                    target_x = int(self.screen_width * (0.194 if self._clash_spell_idx == 0 else 0.741))
                    target_y = int(self.screen_height * 0.235)
                elif target_coords and len(target_coords) == 2:
                    target_x, target_y = target_coords
                else:
                    target_x = int(self.screen_width * 0.50)
                    target_y = int(self.screen_height * 0.50)

                self.deploy_clash_card(card_x, card_y, target_x, target_y)

        # 6. Vehicle Driver actions (Earn to Die 2 - Landscape Optimized)
        elif "accelerate" in act or "gas" in act:
            gx = int(self.screen_width * 0.86)
            gy = int(self.screen_height * 0.82)
            # Sustained hold on gas pedal to maintain powerful engine RPM
            self.hold(gx, gy, duration_ms=360)
        elif "boost" in act or "nitro" in act:
            bx = int(self.screen_width * 0.14)
            by = int(self.screen_height * 0.82)
            # Sustained nitro thruster burst
            self.hold(bx, by, duration_ms=240)
        elif "tilt_forward" in act:
            tx = int(self.screen_width * 0.75)
            ty = int(self.screen_height * 0.82)
            self.hold(tx, ty, duration_ms=160)
        elif "tilt_back" in act:
            tx = int(self.screen_width * 0.25)
            ty = int(self.screen_height * 0.82)
            self.hold(tx, ty, duration_ms=160)

        # 7. EA Sports FC / FIFA Mobile (Landscape Pro Controls)
        elif "finesse_shot" in act:
            # Swipe down on Shoot button for curling far-post finesse finish
            sx = int(self.screen_width * 0.88)
            sy = int(self.screen_height * 0.68)
            self.flick(sx, sy, sx, sy + 110, duration_ms=65)
        elif "chip_shot" in act:
            # Swipe up on Shoot button to lob the oncoming goalkeeper
            sx = int(self.screen_width * 0.88)
            sy = int(self.screen_height * 0.68)
            self.flick(sx, sy, sx, sy - 110, duration_ms=65)
        elif "power_shot" in act:
            # Swipe right across Shoot button for high velocity rocket strike
            sx = int(self.screen_width * 0.88)
            sy = int(self.screen_height * 0.68)
            self.flick(sx, sy, sx + 120, sy, duration_ms=75)
        elif "skill_move" in act or "roulette" in act:
            # Flick upward on Sprint & Skill button to trigger 5-star skill move
            sx = int(self.screen_width * 0.88)
            sy = int(self.screen_height * 0.84)
            self.flick(sx, sy, sx, sy - 130, duration_ms=70)
        elif "sprint_tackle" in act:
            # Sustained hold on sprint/tackle to lock on and press opposing attacker
            sx = int(self.screen_width * 0.88)
            sy = int(self.screen_height * 0.84)
            self.hold(sx, sy, duration_ms=320)
        elif "shoot_goal" in act:
            # Calibrated power strike (~65% power bar charge)
            sx = int(self.screen_width * 0.88)
            sy = int(self.screen_height * 0.68)
            self.hold(sx, sy, duration_ms=180)
        elif "through_pass" in act:
            # Measured through ball into running channel
            tx = int(self.screen_width * 0.77)
            ty = int(self.screen_height * 0.70)
            self.hold(tx, ty, duration_ms=140)
        elif "pass" in act:
            px = int(self.screen_width * 0.77)
            py = int(self.screen_height * 0.87)
            self.tap(px, py)
        elif "dribble_cut_inside" in act or "cut_inside" in act:
            # Virtual joystick diagonal cut inside towards the penalty box
            jx = int(self.screen_width * 0.16)
            jy = int(self.screen_height * 0.74)
            self.swipe(jx, jy, jx + 75, jy - 75, duration_ms=260)
        elif "dribble" in act or "drive_forward" in act:
            # Analog stick drive forward down the wing
            jx1 = int(self.screen_width * 0.16)
            jy = int(self.screen_height * 0.74)
            jx2 = int(self.screen_width * 0.25)
            self.swipe(jx1, jy, jx2, jy, duration_ms=260)

        # 8. Solar Smash (Landscape & Portrait Planetary Superweapons)
        elif "fire_laser" in act:
            cx = self.screen_width // 2
            cy = self.screen_height // 2
            if self.is_landscape:
                cat_x = int(self.screen_width * 0.94)
                cat_y = int(self.screen_height * 0.56)
                sub_x = int(self.screen_width * 0.85)
                sub_y = int(self.screen_height * 0.56)
            else:
                cat_x = int(self.screen_width * 0.40)
                cat_y = int(self.screen_height * 0.863)
                sub_x = int(self.screen_width * 0.21)
                sub_y = int(self.screen_height * 0.803)
            shell_script = f"input tap {cat_x} {cat_y} && sleep 0.12 && input tap {sub_x} {sub_y} && sleep 0.12 && input swipe {cx} {cy} {cx} {cy} 850"
            self._run_shell(shell_script)

        elif "orbital_strike" in act:
            cx = self.screen_width // 2
            cy = self.screen_height // 2
            if self.is_landscape:
                cat_x = int(self.screen_width * 0.94)
                cat_y = int(self.screen_height * 0.38)
                sub_x = int(self.screen_width * 0.85)
                sub_y = int(self.screen_height * 0.38)
            else:
                cat_x = int(self.screen_width * 0.27)
                cat_y = int(self.screen_height * 0.863)
                sub_x = int(self.screen_width * 0.21)
                sub_y = int(self.screen_height * 0.803)
            shell_script = (
                f"input tap {cat_x} {cat_y} && sleep 0.10 && input tap {sub_x} {sub_y} && sleep 0.10 && "
                f"input tap {cx - 80} {cy - 50} && sleep 0.08 && input tap {cx + 70} {cy + 60} && sleep 0.08 && input tap {cx} {cy + 80}"
            )
            self._run_shell(shell_script)

        elif "launch_meteor" in act:
            cx = self.screen_width // 2
            cy = self.screen_height // 2
            if self.is_landscape:
                cat_x = int(self.screen_width * 0.94)
                cat_y = int(self.screen_height * 0.20)
                sub_x = int(self.screen_width * 0.85)
                sub_y = int(self.screen_height * 0.20)
            else:
                cat_x = int(self.screen_width * 0.15)
                cat_y = int(self.screen_height * 0.863)
                sub_x = int(self.screen_width * 0.08)
                sub_y = int(self.screen_height * 0.803)
            shell_script = f"input tap {cat_x} {cat_y} && sleep 0.10 && input tap {sub_x} {sub_y} && sleep 0.10 && input tap {cx} {cy}"
            self._run_shell(shell_script)

        elif "rotate_planet" in act:
            cy = self.screen_height // 2
            x1 = int(self.screen_width * 0.75)
            x2 = int(self.screen_width * 0.25)
            self.swipe(x1, cy, x2, cy, duration_ms=180)

        # 9. BitLife & Choice Simulation
        elif "age_up" in act:
            ax = int(self.screen_width * 0.50)
            ay = int(self.screen_height * 0.69)
            self.tap(ax, ay)
        elif "primary_choice" in act or "choice_1" in act:
            cx = int(self.screen_width * 0.50)
            cy = int(self.screen_height * 0.54)
            self.tap(cx, cy)
        elif "secondary_choice" in act or "choice_2" in act:
            cx = int(self.screen_width * 0.50)
            cy = int(self.screen_height * 0.62)
            self.tap(cx, cy)
        elif "choice_3" in act:
            cx = int(self.screen_width * 0.50)
            cy = int(self.screen_height * 0.70)
            self.tap(cx, cy)
        elif "dismiss_popup" in act:
            cx = int(self.screen_width * 0.50)
            cy = int(self.screen_height * 0.75)
            self.tap(cx, cy)
            time.sleep(0.04)
            self.tap(cx, int(self.screen_height * 0.82))

        # 10. Mobile Card Battlers, TCGs & Deckbuilders (Marvel SNAP, Hearthstone, Pokémon, Balatro)
        elif "play_card" in act:
            self._card_slot_idx = (getattr(self, "_card_slot_idx", 0) + 1) % (6 if self.is_landscape else 4)
            lane = "center"
            if "left" in act:
                lane = "left"
            elif "right" in act:
                lane = "right"
            self.play_card(hand_slot=self._card_slot_idx, target_lane=lane)
        elif "attack_face" in act:
            self.attack_target(target_type="face")
        elif "attack_minion" in act or "trade_minion" in act:
            self.attack_target(target_type="minion")
        elif "hero_power" in act or "snap_cube" in act:
            if self.is_landscape:
                hx = int(self.screen_width * 0.64)
                hy = int(self.screen_height * 0.76)
            else:
                hx = int(self.screen_width * 0.50)
                hy = int(self.screen_height * 0.08)
            self.tap(hx, hy)
        elif "end_turn" in act or "pass_turn" in act or "done" in act:
            if self.is_landscape:
                ex = int(self.screen_width * 0.89)
                ey = int(self.screen_height * 0.49)
            else:
                ex = int(self.screen_width * 0.85)
                ey = int(self.screen_height * 0.92)
            self.tap(ex, ey)
        elif "balatro_play_hand" in act or "play_hand" in act:
            bx = int(self.screen_width * 0.38)
            by = int(self.screen_height * 0.76)
            self.tap(bx, by)
        elif "balatro_discard" in act or "discard" in act:
            dx = int(self.screen_width * 0.62)
            dy = int(self.screen_height * 0.76)
            self.tap(dx, dy)
        elif "select_card" in act:
            self._card_slot_idx = (getattr(self, "_card_slot_idx", 0) + 1) % (6 if self.is_landscape else 4)
            if self.is_landscape:
                slot_ratios = [0.32, 0.40, 0.48, 0.56, 0.64, 0.72]
                sx = int(self.screen_width * slot_ratios[self._card_slot_idx])
                sy = int(self.screen_height * 0.93)
            else:
                slot_ratios = [0.22, 0.40, 0.60, 0.78]
                sx = int(self.screen_width * slot_ratios[self._card_slot_idx])
                sy = int(self.screen_height * 0.88)
            self.tap(sx, sy)
        elif "confirm_choice" in act or "claim" in act:
            cx = int(self.screen_width * 0.50)
            cy = int(self.screen_height * (0.80 if self.is_landscape else 0.88))
            self.tap(cx, cy)

        # 11. Solitaire & Classic Card Puzzles (Klondike, Spider, FreeCell)
        elif "draw_stock" in act or "stock" in act:
            self.tap_solitaire_stock()
        elif "tap_waste_card" in act or "waste" in act:
            self.tap_solitaire_waste()
        elif "sweep_all_columns" in act or "sweep_tableau" in act:
            self.sweep_solitaire_tableau()
        elif "tap_tableau_column" in act or "tap_col" in act:
            # Check if specific column 1-7 requested
            col_match = re.search(r"col[_\s]?(\d+)", act)
            if col_match:
                col_idx = int(col_match.group(1)) - 1
            else:
                self._solitaire_col_idx = (getattr(self, "_solitaire_col_idx", 0) + 1) % 7
                col_idx = self._solitaire_col_idx

            if target_coords:
                self.tap(target_coords[0], target_coords[1])
            else:
                self.tap_solitaire_column(col_idx, y_ratio=0.55)
        elif "drag_column_transfer" in act or "column_transfer" in act:
            self._sol_xfer_step = (getattr(self, "_sol_xfer_step", 0) + 1) % 6
            # Try transferring between adjacent or valid columns
            from_c = self._sol_xfer_step
            to_c = (self._sol_xfer_step + 1) % 7
            self.drag_solitaire_transfer(from_col=from_c, to_col=to_c)
        elif "auto_complete" in act or "auto_finish" in act:
            # Tap prominent Auto-Complete button or center victory banner
            cx = self.screen_width // 2
            cy = int(self.screen_height * (0.86 if not self.is_landscape else 0.88))
            shell_script = f"input tap {cx} {cy} && sleep 0.10 && input tap {cx} {int(self.screen_height * 0.80)}"
            self._run_shell(shell_script)
        elif "tap_foundation" in act or "foundation" in act:
            self._solitaire_found_idx = (getattr(self, "_solitaire_found_idx", 0) + 1) % 4
            self.tap_solitaire_foundation(self._solitaire_found_idx)
        elif "new_deal" in act or "new_game" in act:
            self.trigger_auto_restart()

        # 12. Precision target clicker
        elif "click" in act or "target" in act:
            if target_coords:
                self.tap(target_coords[0], target_coords[1])
            else:
                cx = self.screen_width // 2
                cy = self.screen_height // 2
                self.tap(cx, cy)

        # 13. Auto-revive / restart / continue
        elif "restart" in act or "revive" in act or "continue" in act:
            self.trigger_auto_restart()

        # 14. Snake & Grid Arcades
        elif "turn_up" in act or "snake_up" in act:
            self.swipe_up(duration_ms=60)
        elif "turn_down" in act or "snake_down" in act:
            self.swipe_down(duration_ms=60)
        elif "turn_left" in act or "snake_left" in act:
            self.swipe_left(duration_ms=60)
        elif "turn_right" in act or "snake_right" in act:
            self.swipe_right(duration_ms=60)
