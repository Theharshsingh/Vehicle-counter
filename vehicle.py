import cv2 as cv
import numpy as np
import time
from collections import deque

# ─── CONFIG ───────────────────────────────────────────────────────────────────
VIDEO_SOURCE       = 'video.mp4'
MIN_WIDTH          = 80
MIN_HEIGHT         = 80
CALIBRATION_DIST   = 10       # real-world meters between line1 and line2
PIXELS_PER_METER   = 8.0      # tune: pixels per meter for your camera
SPEED_LIMIT        = 60       # km/h
LINE1_Y            = 400      # first detection line (yellow)
LINE2_Y            = 550      # second detection line / counting line (magenta)
MATCH_THRESHOLD    = 80       # max pixel distance to match same vehicle
LINE_OFFSET        = 8        # crossing tolerance in pixels
ALERT_DURATION     = 3        # seconds overspeed alert stays on screen
SPEED_SMOOTH       = 5        # rolling average window for live speed
HUD_WIDTH          = 280      # right-side info panel width
# ──────────────────────────────────────────────────────────────────────────────

cap = cv.VideoCapture(VIDEO_SOURCE)
FPS = cap.get(cv.CAP_PROP_FPS) or 30

algo = cv.bgsegm.createBackgroundSubtractorMOG()

tracked   = {}          # id -> vehicle data dict
vid_count = 0
counter   = 0
overspeed_count = 0
speed_log = []          # confirmed speeds for avg calculation
overspeed_alerts = []   # (message, expiry_time)
speed_history = {}      # id -> deque of recent speeds for smoothing


def get_center(x, y, w, h):
    return x + w // 2, y + h // 2


def calc_speed_lines(t1, t2):
    dt = t2 - t1
    return round((CALIBRATION_DIST / dt) * 3.6, 1) if dt > 0 else 0


def draw_hud(canvas, total, overspd, avg_spd, log):
    """Draw right-side info panel."""
    h = canvas.shape[0]
    panel = np.zeros((h, HUD_WIDTH, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    cv.rectangle(panel, (0, 0), (HUD_WIDTH - 1, h - 1), (60, 60, 60), 1)

    def txt(text, y, color=(200, 200, 200), scale=0.55, thick=1):
        cv.putText(panel, text, (10, y), cv.FONT_HERSHEY_SIMPLEX, scale, color, thick)

    txt("── VEHICLE MONITOR ──", 30, (0, 220, 255), 0.55, 1)
    txt(f"Counted   : {total}",  65, (0, 255, 100))
    txt(f"Overspeed : {overspd}", 95, (0, 80, 255) if overspd > 0 else (0, 255, 100))
    txt(f"Avg Speed : {avg_spd} km/h", 125, (255, 200, 0))
    txt(f"Limit     : {SPEED_LIMIT} km/h", 155, (0, 200, 255))

    txt("── Speed Log ──", 190, (180, 180, 180), 0.5)
    for i, (vid, spd, flag) in enumerate(reversed(log[-12:])):
        color = (0, 80, 255) if flag else (0, 255, 150)
        marker = " !" if flag else "  "
        txt(f"{marker} #{vid:>3}  {spd:>6.1f} km/h", 215 + i * 22, color, 0.48)

    return np.hstack([canvas, panel])


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    current_time = time.time()

    # ── Preprocessing ──
    gray     = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur     = cv.GaussianBlur(gray, (3, 3), 5)
    mask     = algo.apply(blur)
    dilated  = cv.dilate(mask, np.ones((5, 5)))
    kernel   = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    closed   = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)
    closed   = cv.morphologyEx(closed,  cv.MORPH_CLOSE, kernel)
    contours, _ = cv.findContours(closed, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # ── Detection lines ──
    cv.line(frame, (0, LINE1_Y), (frame_w, LINE1_Y), (0, 255, 255), 2)
    cv.line(frame, (0, LINE2_Y), (frame_w, LINE2_Y), (255, 0, 255), 2)
    cv.putText(frame, "L1", (5, LINE1_Y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv.putText(frame, "L2", (5, LINE2_Y - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    matched_ids = set()

    for c in contours:
        x, y, w, h = cv.boundingRect(c)
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        cx, cy = get_center(x, y, w, h)

        # ── Match to existing vehicle ──
        best_id, best_dist = None, MATCH_THRESHOLD
        for vid, data in tracked.items():
            d = np.hypot(cx - data['cx'], cy - data['cy'])
            if d < best_dist:
                best_dist, best_id = d, vid

        if best_id is None:
            best_id = vid_count
            tracked[best_id] = {
                'cx': cx, 'cy': cy, 'prev_cy': cy,
                'line1_time': None, 'line2_time': None,
                'speed': 0, 'live_speed': 0,
                'counted': False, 'overspeed': False,
                'direction': 'down'
            }
            speed_history[best_id] = deque(maxlen=SPEED_SMOOTH)
            vid_count += 1

        matched_ids.add(best_id)
        vd = tracked[best_id]

        # ── Update position & direction ──
        vd['direction'] = 'down' if cy >= vd['cy'] else 'up'
        vd['prev_cy']   = vd['cy']
        vd['cx'], vd['cy'] = cx, cy

        # ── Live speed (pixel displacement) ──
        pixel_diff = abs(cy - vd['prev_cy'])
        if pixel_diff > 0:
            raw_spd = round((pixel_diff / PIXELS_PER_METER) * FPS * 3.6, 1)
            speed_history[best_id].append(raw_spd)
            vd['live_speed'] = round(sum(speed_history[best_id]) / len(speed_history[best_id]), 1)

        # ── Line 1 crossing ──
        if vd['line1_time'] is None and abs(cy - LINE1_Y) <= LINE_OFFSET:
            vd['line1_time'] = current_time

        # ── Line 2 crossing → confirmed speed ──
        if (vd['line2_time'] is None and vd['line1_time'] is not None
                and abs(cy - LINE2_Y) <= LINE_OFFSET):
            vd['line2_time'] = current_time
            vd['speed'] = calc_speed_lines(vd['line1_time'], vd['line2_time'])

            if vd['speed'] > SPEED_LIMIT:
                vd['overspeed'] = True

        # ── Count at Line 2 ──
        if not vd['counted'] and abs(cy - LINE2_Y) <= LINE_OFFSET:
            counter += 1
            vd['counted'] = True
            confirmed = vd['speed'] if vd['speed'] > 0 else vd['live_speed']
            speed_log.append((best_id, confirmed, vd['overspeed']))

            if vd['overspeed']:
                overspeed_count += 1
                msg = f"OVERSPEED! #{best_id}  {vd['speed']} km/h"
                overspeed_alerts.append((msg, current_time + ALERT_DURATION))
                print(f"[OVERSPEED] Vehicle #{best_id} — {vd['speed']} km/h")
            else:
                print(f"[COUNT] Vehicle #{best_id} — {confirmed} km/h | Total: {counter}")

        # ── Draw bounding box ──
        display_speed = vd['speed'] if vd['speed'] > 0 else vd['live_speed']
        is_fast       = display_speed > SPEED_LIMIT
        box_color     = (0, 0, 255) if is_fast else (0, 255, 0)
        lbl_color     = (0, 0, 255) if is_fast else (255, 255, 0)

        cv.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        cv.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        direction_arrow = "v" if vd['direction'] == 'down' else "^"
        label = f"#{best_id} {display_speed}km/h {direction_arrow}"
        cv.putText(frame, label, (x, max(y - 8, 15)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, lbl_color, 2)

        # Overspeed badge on box
        if is_fast:
            cv.putText(frame, "OVERSPEED", (x, y + h + 16),
                       cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # ── Cleanup stale vehicles ──
    for vid in list(tracked):
        if vid not in matched_ids and tracked[vid]['cy'] > frame_h + 20:
            del tracked[vid]
            speed_history.pop(vid, None)

    # ── Overspeed alert banner (flashing) ──
    overspeed_alerts = [(m, e) for m, e in overspeed_alerts if e > current_time]
    for i, (msg, exp) in enumerate(overspeed_alerts):
        flash = int(current_time * 4) % 2 == 0
        if flash:
            cv.rectangle(frame, (0, frame_h - 50 - i * 45),
                         (frame_w, frame_h - 10 - i * 45), (0, 0, 180), -1)
        cv.putText(frame, msg, (20, frame_h - 20 - i * 45),
                   cv.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # ── Top HUD on frame ──
    avg_speed = round(sum(s for _, s, _ in speed_log) / len(speed_log), 1) if speed_log else 0.0
    cv.rectangle(frame, (0, 0), (360, 50), (20, 20, 20), -1)
    cv.putText(frame, f"COUNT: {counter}  |  OVERSPEED: {overspeed_count}",
               (10, 32), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2)

    # ── Attach side HUD panel ──
    output = draw_hud(frame, counter, overspeed_count, avg_speed, speed_log)

    cv.imshow('Vehicle Speed Detection', output)
    if cv.waitKey(1) == 27:
        break

cap.release()
cv.destroyAllWindows()

print("\n" + "=" * 40)
print(f"  Total Vehicles Counted : {counter}")
print(f"  Overspeed Violations   : {overspeed_count}")
avg = round(sum(s for _, s, _ in speed_log) / len(speed_log), 1) if speed_log else 0
print(f"  Average Speed          : {avg} km/h")
print("=" * 40)
