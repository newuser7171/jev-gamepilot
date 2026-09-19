import json
import time
import urllib.request

def test_gamepilot_decision():
    print("=== TEST JEV-GAMEPILOT TACTICAL DECISION VIA CLASSIFIER.DEV ===")
    
    # Dino profile actions
    labels = ["jump", "duck", "wait"]
    
    situations = [
        "Chrome Dino Runner: Player is running at speed 12. A low cactus obstacle is approaching fast at distance 65px directly ahead on ground level.",
        "Chrome Dino Runner: Player is running. A high flying pterodactyl bird is approaching at eye level (height 50px).",
        "Chrome Dino Runner: The ground ahead is completely clear. No obstacles for the next 400px.",
        "Subway Surfers: Middle lane is blocked by an oncoming subway train at 80px. Left lane is blocked by a barricade. Right lane is wide open with gold coins."
    ]
    
    payload = {
        "labels": ["jump", "duck", "wait", "lane_left", "lane_right"],
        "inputs": situations,
        "instructions": (
            "Select the best reflex action to survive and avoid collisions in this game. "
            "For low ground obstacles jump. For high flying obstacles duck. "
            "For blocked lanes shift to an open lane. For clear ground wait."
        )
    }
    
    req = urllib.request.Request(
        "https://classifier.dev",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "jev-gamepilot/2.0 (TypeSafe Jev System One)"
        }
    )
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
    latency = (time.perf_counter() - t0) * 1000
    
    print(f"Batch latency: {latency:.1f}ms (model: {data.get('model')})")
    for sit, res in zip(situations, data.get("results", [])):
        print(f"\nSituation: {sit[:70]}...")
        print(f"  -> Decision: [{res['label'].upper()}] (confidence: {res['confidence']:.2f})")

def test_profile_detection():
    print("\n=== TEST AUTO GAME PROFILE DETECTION VIA CLASSIFIER.DEV ===")
    profiles = [
        "dino_runner",
        "subway_surfers_3lane",
        "flappy_bird",
        "aim_trainer_clicker",
        "chess_copilot",
        "2d_platformer_arcade"
    ]
    
    window_titles = [
        "Chrome: chrome://dino - No internet runner game",
        "BlueStacks App Player - Subway Surfers Mobile Game",
        "Chess.com: Play Chess Online - Free Games (Grandmaster Tactics)",
        "Aim Lab / Kovaak's Target Practice Simulation",
        "Flappy Bird HTML5 Canvas Retro Arcade"
    ]
    
    payload = {
        "labels": profiles,
        "inputs": window_titles,
        "instructions": "Classify the active window or screen title into the exact corresponding GamePilot profile."
    }
    
    req = urllib.request.Request(
        "https://classifier.dev",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "jev-gamepilot/2.0"
        }
    )
    
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
        
    for title, res in zip(window_titles, data.get("results", [])):
        print(f"Window: '{title}'")
        print(f"  -> Auto-Detected Profile: [{res['label']}] (conf: {res['confidence']:.2f})")

if __name__ == "__main__":
    test_gamepilot_decision()
    test_profile_detection()
