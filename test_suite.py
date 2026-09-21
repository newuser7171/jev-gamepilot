"""
Verification test suite for Jev-GamePilot 8 Ball Pool & Game Catalog.
"""
import sys
import numpy as np

def run_tests():
    print("1. Testing Profile Manager loading 28+ profiles...")
    from profile_manager import ProfileManager
    pm = ProfileManager()
    profiles = pm.list_profiles()
    print(f"   Loaded {len(profiles)} total game profiles.")
    assert len(profiles) >= 28, f"Expected >= 28 profiles, found {len(profiles)}"

    p8_mob = pm.get_profile("mobile_8ball_pool")
    p8_pc = pm.get_profile("pc_8ball_pool")
    assert p8_mob is not None, "mobile_8ball_pool profile missing"
    assert p8_pc is not None, "pc_8ball_pool profile missing"
    print("   [PASS] 8 Ball Pool profiles (mobile and PC) loaded successfully.")

    print("\n2. Testing Universal Vision 8 Ball Pool perception...")
    from universal_vision import UniversalVision
    uv = UniversalVision()
    # Create synthetic pool table frame (green felt with white cue ball)
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Green felt table
    test_frame[100:620, 150:1130] = [30, 120, 40]
    # White cue ball at center
    import cv2
    cv2.circle(test_frame, (640, 360), 12, (255, 255, 255), -1)
    # Red target ball
    cv2.circle(test_frame, (800, 360), 12, (0, 0, 220), -1)

    scene = uv.analyze_frame(test_frame, p8_mob)
    print(f"   Vision analyzed: table_detected={scene.table_detected}, cue_ball={scene.cue_ball}, pockets={len(scene.pockets)}")
    assert scene.table_detected is True, "Expected table_detected to be True"
    assert scene.cue_ball is not None, "Expected cue ball to be detected"
    assert len(scene.pockets) == 6, f"Expected 6 pockets, found {len(scene.pockets)}"

    overlay = uv.render_debug_overlay(test_frame, scene, p8_mob)
    assert overlay is not None and overlay.shape == test_frame.shape
    print("   [PASS] 8 Ball Pool perception and laser aim overlay rendered successfully.")

    print("\n3. Testing Universal Brain 8 Ball Pool decision reasoning...")
    from universal_brain import UniversalBrain
    ub = UniversalBrain()
    decision = ub.decide(scene, p8_mob)
    print(f"   Brain decision: action={decision.get('action')}, confidence={decision.get('confidence')}, source={decision.get('source')}")
    assert "action" in decision, "Expected action in decision"
    print("   [PASS] 8 Ball Pool decision engine produced valid action.")

    print("\n4. Testing Phone Adapter 8 Ball Pool actions...")
    from adapters.phone_adapter import PACKAGE_PROFILE_MAP
    assert "com.miniclip.eightballpool" in PACKAGE_PROFILE_MAP, "com.miniclip.eightballpool missing from package map"
    print("   [PASS] Android package mapping for 8 Ball Pool verified.")

    print("\nALL 4 TESTS PASSED CLEANLY!")

if __name__ == "__main__":
    run_tests()
