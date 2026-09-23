"""CoC + Brawl Stars adapters, package map, profiles, vision phases, brain branches."""
import unittest

import numpy as np

from adapters.phone_adapter import PACKAGE_PROFILE_MAP, AdbController
from profile_manager import ProfileManager


def blank(h=2340, w=1080):
    return np.zeros((h, w, 3), dtype=np.uint8)


def noise(h=1080, w=2340, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


class PackageMapTests(unittest.TestCase):
    def test_coc_package_maps(self):
        self.assertEqual(PACKAGE_PROFILE_MAP["com.supercell.clashofclans"], "mobile_coc")

    def test_brawl_package_maps(self):
        self.assertEqual(PACKAGE_PROFILE_MAP["com.supercell.brawlstars"], "mobile_brawlstars")

    def test_clash_royale_unchanged(self):
        self.assertEqual(PACKAGE_PROFILE_MAP["com.supercell.clashroyale"], "mobile_clash_royale")


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.pm = ProfileManager()

    def test_coc_profile_exists(self):
        p = self.pm.get_profile("mobile_coc")
        self.assertIsNotNone(p)
        names = {a.name for a in p.actions}
        self.assertIn("find_match", names)
        self.assertIn("deploy_troop", names)
        self.assertIn("end_battle", names)
        self.assertIn("confirm_ok", names)
        self.assertIn("wait", names)

    def test_brawl_profile_exists(self):
        p = self.pm.get_profile("mobile_brawlstars")
        self.assertIsNotNone(p)
        names = {a.name for a in p.actions}
        self.assertIn("move_to", names)
        self.assertIn("attack", names)
        self.assertIn("use_super", names)
        self.assertIn("start_battle", names)
        self.assertIn("confirm_ok", names)
        self.assertIn("wait", names)


class CocAdapterTests(unittest.TestCase):
    def setUp(self):
        from adapters.coc_adapter import CocRaidAdapter, detect_coc_phase

        self.cls = CocRaidAdapter
        self.detect = detect_coc_phase
        self.adapter = CocRaidAdapter(1080, 2340)

    def test_blank_frame_unknown(self):
        self.assertEqual(self.detect(blank()), "unknown")

    def test_none_frame_safe(self):
        self.assertEqual(self.detect(None), "unknown")

    def test_decide_home_waits(self):
        d = self.adapter.decide(blank(), phase="home_village")
        self.assertEqual(d["action"], "wait")
        self.assertEqual(d["source"], "coc_raid")

    def test_search_waits(self):
        self.adapter.note_raid_start()
        d = self.adapter.decide(blank(), phase="attack_search")
        self.assertEqual(d["action"], "wait")

    def test_in_raid_deploys_with_4_tuple(self):
        frame = noise(2340, 1080, seed=1)
        d = self.adapter.decide(frame, phase="in_raid")
        self.assertEqual(d["action"], "deploy_troop")
        self.assertEqual(len(d["target_coords"]), 4)
        self.assertEqual(self.adapter.deploy_count, 1)
        self.assertTrue(self.adapter.raid_open)

    def test_raid_timeout_ends(self):
        self.adapter.note_raid_start()
        self.adapter.start_time -= 200  # force past MAX_RAID_SECONDS
        d = self.adapter.decide(blank(), phase="in_raid")
        self.assertEqual(d["action"], "end_battle")

    def test_results_dismiss_and_close(self):
        self.adapter.note_raid_start()
        d = self.adapter.decide(blank(), phase="results")
        self.assertEqual(d["action"], "confirm_ok")
        self.assertFalse(self.adapter.raid_open)

    def test_note_end_idempotent(self):
        self.assertIsNone(self.adapter.note_raid_end(None))
        self.adapter.note_raid_start()
        rec = self.adapter.note_raid_end(None)
        self.assertIsNotNone(rec)
        self.assertIsNone(self.adapter.note_raid_end(None))

    def test_perimeter_points_normalized(self):
        from adapters.coc_adapter import perimeter_points

        pts = perimeter_points((100, 200, 500, 800), 1080, 2340)
        self.assertTrue(pts)
        for rx, ry in pts:
            self.assertGreaterEqual(rx, 0.0)
            self.assertLessEqual(rx, 1.0)
            self.assertGreaterEqual(ry, 0.0)
            self.assertLessEqual(ry, 1.0)


class BrawlAdapterTests(unittest.TestCase):
    def setUp(self):
        from adapters.brawl_adapter import BrawlMatchAdapter, detect_brawl_phase

        self.detect = detect_brawl_phase
        self.adapter = BrawlMatchAdapter(2340, 1080)

    def test_blank_unknown(self):
        self.assertEqual(self.detect(blank(1080, 2340)), "unknown")

    def test_none_safe(self):
        self.assertEqual(self.detect(None), "unknown")

    def test_menu_waits(self):
        d = self.adapter.decide(blank(1080, 2340), phase="menu")
        self.assertEqual(d["action"], "wait")
        self.assertFalse(self.adapter.match_open)

    def test_matchmaking_waits(self):
        d = self.adapter.decide(blank(1080, 2340), phase="matchmaking")
        self.assertEqual(d["action"], "wait")

    def test_results_confirm(self):
        self.adapter.note_match_start()
        d = self.adapter.decide(blank(1080, 2340), phase="results")
        self.assertEqual(d["action"], "confirm_ok")
        self.assertFalse(self.adapter.match_open)

    def test_in_match_initial_move_or_attack(self):
        frame = noise(1080, 2340, seed=2)
        d = self.adapter.decide(frame, phase="in_match")
        self.assertIn(d["action"], ("move_to", "attack", "use_super", "wait"))
        self.assertTrue(self.adapter.match_open)

    def test_move_coords_4_tuple(self):
        from adapters.brawl_adapter import BrawlMatchAdapter

        ad = BrawlMatchAdapter(2340, 1080)
        # Force roam path: no enemy, no super.
        frame = blank(1080, 2340)
        d = ad.decide(frame, phase="in_match")
        if d["action"] == "move_to":
            self.assertEqual(len(d["target_coords"]), 4)

    def test_note_match_end_idempotent(self):
        self.assertIsNone(self.adapter.note_match_end(None))
        self.adapter.note_match_start()
        self.assertIsNotNone(self.adapter.note_match_end(None))
        self.assertIsNone(self.adapter.note_match_end(None))


class AutoDetectHeuristicTests(unittest.TestCase):
    """brawl/clashofclans keywords must not fall through to Clash Royale."""

    def test_heuristic_prefers_brawl(self):
        # Exact map wins; also check keyword path for unknown brawl* packages.
        adb = AdbController.__new__(AdbController)  # skip __init__ (no device)
        # Monkeypatch detect_foreground_package
        adb.detect_foreground_package = lambda: "com.supercell.brawlstars.beta"
        prof, pkg = adb.auto_detect_game_profile()
        self.assertEqual(prof, "mobile_brawlstars")

    def test_heuristic_prefers_coc(self):
        adb = AdbController.__new__(AdbController)
        adb.detect_foreground_package = lambda: "com.supercell.clashofclans.test"
        prof, pkg = adb.auto_detect_game_profile()
        self.assertEqual(prof, "mobile_coc")

    def test_clash_royale_keyword_still_works(self):
        adb = AdbController.__new__(AdbController)
        adb.detect_foreground_package = lambda: "com.supercell.clashroyale.lite"
        prof, pkg = adb.auto_detect_game_profile()
        self.assertEqual(prof, "mobile_clash_royale")


class DispatchTests(unittest.TestCase):
    """dispatch_action must route new action names without touching a device."""

    def _ctrl(self):
        adb = AdbController.__new__(AdbController)
        adb.screen_width = 1080
        adb.screen_height = 2340
        adb.is_landscape = False
        adb._clash_card_idx = 0
        adb._clash_spell_idx = 0
        adb.taps = []
        adb.swipes = []
        adb.tap = lambda x, y: adb.taps.append((x, y))
        adb.swipe = lambda x1, y1, x2, y2, duration_ms=120: adb.swipes.append((x1, y1, x2, y2, duration_ms))
        return adb

    def test_find_match_taps(self):
        adb = self._ctrl()
        AdbController.dispatch_action(adb, "find_match")
        self.assertEqual(len(adb.taps), 1)

    def test_deploy_troop_4tuple(self):
        adb = self._ctrl()
        AdbController.dispatch_action(adb, "deploy_troop", (167, 2118, 540, 1200))
        self.assertEqual(len(adb.taps), 2)
        self.assertEqual(adb.taps[0], (167, 2118))
        self.assertEqual(adb.taps[1], (540, 1200))

    def test_end_battle_taps(self):
        adb = self._ctrl()
        AdbController.dispatch_action(adb, "end_battle")
        self.assertEqual(len(adb.taps), 2)

    def test_move_to_swipes(self):
        adb = self._ctrl()
        adb.screen_width = 2340
        adb.screen_height = 1080
        AdbController.dispatch_action(adb, "move_to", (327, 777, 1170, 432))
        self.assertEqual(len(adb.swipes), 1)

    def test_use_super_tap(self):
        adb = self._ctrl()
        adb.screen_width = 2340
        adb.screen_height = 1080
        AdbController.dispatch_action(adb, "use_super", (1825, 950))
        self.assertEqual(adb.taps, [(1825, 950)])

    def test_attack_flick_and_fire(self):
        adb = self._ctrl()
        adb.screen_width = 2340
        adb.screen_height = 1080
        AdbController.dispatch_action(adb, "attack", (327, 777, 1000, 400, 2059, 842))
        self.assertEqual(len(adb.swipes), 1)
        self.assertEqual(len(adb.taps), 1)

    def test_wait_is_silent(self):
        adb = self._ctrl()
        AdbController.dispatch_action(adb, "wait")
        self.assertEqual(adb.taps, [])
        self.assertEqual(adb.swipes, [])


class BrainBranchSmokeTests(unittest.TestCase):
    def test_brain_constructs_new_adapters(self):
        from universal_brain import UniversalBrain

        ub = UniversalBrain()
        self.assertIsNotNone(ub.coc_adapter)
        self.assertIsNotNone(ub.brawl_adapter)

    def test_coc_home_returns_find_match(self):
        from universal_brain import UniversalBrain
        from profile_manager import ProfileManager
        from universal_vision import UniversalSceneState

        ub = UniversalBrain()
        prof = ProfileManager().get_profile("mobile_coc")
        scene = UniversalSceneState()
        scene.game_phase = "main_menu"
        scene.raw_frame = blank()
        d = ub.decide(scene, prof)
        self.assertEqual(d["action"], "find_match")

    def test_brawl_menu_returns_start_battle(self):
        from universal_brain import UniversalBrain
        from profile_manager import ProfileManager
        from universal_vision import UniversalSceneState

        ub = UniversalBrain()
        prof = ProfileManager().get_profile("mobile_brawlstars")
        scene = UniversalSceneState()
        scene.game_phase = "main_menu"
        scene.raw_frame = blank(1080, 2340)
        d = ub.decide(scene, prof)
        self.assertEqual(d["action"], "start_battle")

    def test_coc_results_confirm(self):
        from universal_brain import UniversalBrain
        from profile_manager import ProfileManager
        from universal_vision import UniversalSceneState

        ub = UniversalBrain()
        prof = ProfileManager().get_profile("mobile_coc")
        scene = UniversalSceneState()
        scene.game_phase = "game_over"
        scene.coc_phase = "results"
        scene.raw_frame = blank()
        d = ub.decide(scene, prof)
        self.assertEqual(d["action"], "confirm_ok")


if __name__ == "__main__":
    unittest.main()
