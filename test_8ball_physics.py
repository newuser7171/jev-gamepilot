import cv2
import numpy as np
import math

def analyze_8ball_screen(img_path="pool_screen.png"):
    img = cv2.imread(img_path)
    if img is None:
        print("Failed to load", img_path)
        return

    h, w, _ = img.shape
    print(f"Analyzing {img_path} ({w}x{h})...")

    # 1. Cue power bar detection on left edge
    x1, x2 = int(w * 0.070), int(w * 0.095)
    y1, y2 = int(h * 0.35), int(h * 0.85)
    crop_cue = img[y1:y2, x1:x2]
    wood = (crop_cue[:, :, 2] > 180) & (crop_cue[:, :, 1] > 130) & (crop_cue[:, :, 0] < 150)
    cue_ready = wood.sum() > 300
    print(f"1. Cue ready: {cue_ready} (wood pixels: {wood.sum()})")

    # 2. Pocket definitions
    pockets = [
        ("Top-Left", int(w * 0.109), int(h * 0.171)),
        ("Top-Mid", int(w * 0.500), int(h * 0.139)),
        ("Top-Right", int(w * 0.891), int(h * 0.171)),
        ("Bot-Left", int(w * 0.109), int(h * 0.829)),
        ("Bot-Mid", int(w * 0.500), int(h * 0.861)),
        ("Bot-Right", int(w * 0.891), int(h * 0.829)),
    ]

    # 3. Detect Balls on Table
    table_x1, table_x2 = int(w * 0.11), int(w * 0.89)
    table_y1, table_y2 = int(h * 0.16), int(h * 0.84)
    table_bounds = img[table_y1:table_y2, table_x1:table_x2]
    gray = cv2.cvtColor(table_bounds, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=32,
        param1=50, param2=22, minRadius=16, maxRadius=28
    )

    balls = []
    cue_ball = None

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for cx, cy, r in circles:
            rx, ry = cx + table_x1, cy + table_y1
            bgr = img[ry, rx].astype(int)
            # Cue ball check: pure bright white
            is_white = (bgr[0] > 220) and (bgr[1] > 220) and (bgr[2] > 220)
            # Check 3x3 surrounding region
            patch = img[ry-4:ry+4, rx-4:rx+4]
            if patch.size > 0:
                is_solid_white = (patch[:, :, 0] > 200).all() and (patch[:, :, 1] > 200).all() and (patch[:, :, 2] > 200).all()
            else:
                is_solid_white = False

            # In 8 Ball Pool, the in-game ghost ball guide is also white circle, but cue ball has cue stick pointing at it
            balls.append({
                "pos": (rx, ry),
                "radius": r,
                "color": bgr,
                "is_white": is_white or is_solid_white
            })

    print(f"2. Detected {len(balls)} ball candidates on felt.")

    # Locate cue ball:
    # In table_crop, the cue ball was around (1170, 620)
    white_balls = [b for b in balls if b["is_white"]]
    # If multiple white, pick the one with lowest variance or nearest center
    if white_balls:
        cue_ball = min(white_balls, key=lambda b: math.hypot(b["pos"][0] - w*0.5, b["pos"][1] - h*0.5))
    else:
        cue_ball = {"pos": (int(w * 0.5), int(h * 0.58)), "radius": 24}

    print(f"3. Cue ball locked at: {cue_ball['pos']}")

    # 4. Filter object balls
    object_balls = [b for b in balls if b != cue_ball and math.hypot(b["pos"][0] - cue_ball["pos"][0], b["pos"][1] - cue_ball["pos"][1]) > 50]
    print(f"4. Object balls found: {len(object_balls)}")

    # 5. Evaluate best pot option
    best_shot = None
    best_score = -9999.0

    cx, cy = cue_ball["pos"]
    ball_r = 24

    for ob in object_balls:
        bx, by = ob["pos"]
        dist_cue_to_ball = math.hypot(bx - cx, by - cy)
        if dist_cue_to_ball < 40:
            continue

        for p_name, px, py in pockets:
            dist_ball_to_pocket = math.hypot(px - bx, py - by)
            if dist_ball_to_pocket < 30:
                continue

            # Ghost ball calculation: 2*R behind target ball along ball->pocket line
            ux = (px - bx) / dist_ball_to_pocket
            uy = (py - by) / dist_ball_to_pocket
            gx = int(bx - ux * (ball_r * 2))
            gy = int(by - uy * (ball_r * 2))

            # Aim vector from cue ball to ghost ball
            aim_dx = gx - cx
            aim_dy = gy - cy
            aim_dist = math.hypot(aim_dx, aim_dy)
            if aim_dist < 10:
                continue

            # Angle between (cue->ghost) and (ball->pocket)
            dot = (aim_dx * ux + aim_dy * uy) / (aim_dist)
            dot = max(-1.0, min(1.0, dot))
            cut_angle_deg = math.degrees(math.acos(dot))

            # Cut angle must be < 75 degrees for a realistic pot
            if cut_angle_deg > 75:
                continue

            # Score: High when cut angle is small and distances are short
            score = 100.0 - (cut_angle_deg * 1.2) - (dist_ball_to_pocket * 0.03) - (dist_cue_to_ball * 0.02)
            if score > best_score:
                best_score = score
                # Power calculation
                total_dist = dist_cue_to_ball + dist_ball_to_pocket
                if total_dist < 600:
                    power = 0.40
                elif total_dist < 1100:
                    power = 0.60
                else:
                    power = 0.80

                best_shot = {
                    "target_ball": (bx, by),
                    "ghost_ball": (gx, gy),
                    "pocket": (p_name, px, py),
                    "cut_angle_deg": round(cut_angle_deg, 1),
                    "power": power,
                    "score": round(score, 1),
                }

    if best_shot:
        print("\n=== OPTIMAL SHOT FOUND ===")
        print(f"Target Ball: {best_shot['target_ball']}")
        print(f"Target Ghost: {best_shot['ghost_ball']}")
        print(f"Pocket: {best_shot['pocket'][0]} ({best_shot['pocket'][1]}, {best_shot['pocket'][2]})")
        print(f"Cut Angle: {best_shot['cut_angle_deg']} deg")
        print(f"Recommended Power: {int(best_shot['power']*100)}%")
        print(f"Pot Quality Score: {best_shot['score']}")
    else:
        print("No direct cut shot found (break shot or safety recommended)")

if __name__ == "__main__":
    analyze_8ball_screen()
