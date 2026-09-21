"""Window-selection regressions without interacting with the desktop."""
import unittest
from unittest.mock import Mock, patch

from pc_pilot import PcGamePilot
from profile_manager import ProfileManager
from vision_engine import VisionEngine


def window(title):
    return Mock(title=title, width=800, height=600, left=50, top=60, isMinimized=False)


class TargetingTests(unittest.TestCase):
    def setUp(self):
        self.pilot = PcGamePilot.__new__(PcGamePilot)
        self.pilot.profile_mgr = ProfileManager()
        self.pilot.vision = Mock()
        self.pilot.target_window = None
        self.pilot.requested_profile_id = "auto"
        self.pilot.window_keyword = ""

    def test_ordinary_chrome_window_is_not_dino(self):
        with patch("pc_pilot.attach_input_desktop"), \
                patch("pc_pilot.gw.getAllWindows", return_value=[window("Email - Google Chrome")]):
            profile, target = self.pilot._auto_detect_pc_game()
        self.assertEqual(profile.id, "pc_universal")
        self.assertIsNone(target)

    def test_preview_is_not_selected_as_game(self):
        preview = window("Jev-GamePilot // Solitaire Preview")
        game = window("Solitaire")
        with patch("pc_pilot.attach_input_desktop"), \
                patch("pc_pilot.gw.getAllWindows", return_value=[preview, game]):
            _, target = self.pilot._auto_detect_pc_game()
        self.assertIs(target, game)

    def test_explicit_window_is_honored_in_auto_mode(self):
        game = window("My Custom Game")
        self.pilot.window_keyword = "My Custom Game"
        with patch.object(self.pilot, "_find_matching_window", return_value=game) as find, \
                patch.object(self.pilot, "_auto_detect_pc_game", return_value=(self.pilot.profile_mgr.get_profile("runner_dino"), None)):
            profile = self.pilot._resolve_game_profile()
        find.assert_called_once_with("My Custom Game")
        self.assertIs(self.pilot.target_window, game)
        self.assertEqual(profile.id, "pc_universal")

    def test_negative_monitor_coordinates_are_preserved(self):
        vision = VisionEngine.__new__(VisionEngine)
        vision.set_region(-1920, -100, 800, 600)
        self.assertEqual(vision.region, {"left": -1920, "top": -100, "width": 800, "height": 600})
