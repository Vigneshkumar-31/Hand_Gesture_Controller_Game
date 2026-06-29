import cv2
import mediapipe as mp
import pyautogui
import time
import urllib.request
import os

# ============================================================
#  FIX 1 — kill pyautogui's hidden 0.1s delay (biggest culprit)
# ============================================================
pyautogui.PAUSE     = 0        # default is 0.1 s added after EVERY call
pyautogui.FAILSAFE  = False    # don't raise exception if mouse hits corner

# ========== CONFIG ==========
COOLDOWN   = 0.05   # 50 ms — tight enough for rapid jumps, loose enough to avoid doubles
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"                        
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
# ============================

TIP = [4, 8, 12, 16, 20]
PIP = [3, 7, 11, 15, 19]

if not os.path.exists(MODEL_PATH):
    print("Downloading HandLandmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Done.\n")

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    # FIX 2 — lower confidence = faster detection response
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# FIX 3 — request the highest frame rate the camera supports
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS,          60)   # ask for 60 fps; camera will give max it can

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ============================================================
#  GESTURE HELPERS
# ============================================================

def fingers_up(lm):
    up = [lm[TIP[0]].x < lm[PIP[0]].x]          # thumb (x-axis)
    for i in range(1, 5):
        up.append(lm[TIP[i]].y < lm[PIP[i]].y)  # other fingers (y-axis)
    return up  # [thumb, index, middle, ring, pinky]


def detect_gesture(lm):
    _, index, middle, ring, pinky = fingers_up(lm)

    if index and not middle and not ring and not pinky:
        return "INDEX_UP"    # ☝ jump

    if not index and not middle and not ring and not pinky:
        return "FIST"        # ✊ duck

    if index and middle and not ring and not pinky:
        return "PEACE"       # ✌ restart

    # All fingers up
    if index and middle and ring and pinky:
        return "OPEN_PALM"   # 🖐 jump (alternative)

    return "NONE"


# ============================================================
#  STATE
# ============================================================
last_jump_time = 0
duck_active    = False
prev_gesture   = "NONE"

GESTURE_LABELS = {
    "OPEN_PALM": ("JUMP",    (0, 255, 100)),
    "FIST":      ("NEUTRAL", (180, 180, 180)),
    "PEACE":     ("RESTART", (255, 200, 0)),
    "INDEX_UP":  ("---",     (180, 180, 180)),
    "NONE":      ("---",     (180, 180, 180)),
}

print("Make sure the game window is focused!\nStarting in 3 seconds...")
time.sleep(3)

# ============================================================
#  MAIN LOOP
# ============================================================
with HandLandmarker.create_from_options(options) as landmarker:

    # FIX 4 — use a monotonic counter instead of CAP_PROP_POS_MSEC
    # (CAP_PROP_POS_MSEC can return 0 repeatedly, causing detection to be skipped)
    frame_ts_ms = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_ts_ms += 1           # always strictly increasing — never 0 twice
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result    = landmarker.detect_for_video(mp_image, frame_ts_ms)

        gesture = "NONE"

        if result.hand_landmarks:
            lm  = result.hand_landmarks[0]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]

            # Draw skeleton
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt,
                           6 if i in TIP else 4,
                           (0, 0, 255) if i in TIP else (255, 255, 255), -1)

            gesture = detect_gesture(lm)
            now     = time.time()

            # ---- JUMP ----
            if gesture == "OPEN_PALM":
                if prev_gesture != "OPEN_PALM":
                    pyautogui.press("space")
                    last_jump_time = now
                    print("[OPEN_PALM] -> JUMP")
                elif now - last_jump_time > COOLDOWN:
                    pyautogui.press("space")
                    last_jump_time = now
                if duck_active:
                    pyautogui.keyUp("down")
                    duck_active = False

            # ---- NEUTRAL (fist = do nothing, release duck if held) ----
            elif gesture == "FIST":
                if duck_active:
                    pyautogui.keyUp("down")
                    duck_active = False

            # ---- RESTART ----
            elif gesture == "PEACE":
                if now - last_jump_time > COOLDOWN * 10:
                    pyautogui.press("enter")
                    last_jump_time = now
                    print("[PEACE] -> RESTART")
                if duck_active:
                    pyautogui.keyUp("down")
                    duck_active = False

            # ---- NEUTRAL ----
            else:
                if duck_active:
                    pyautogui.keyUp("down")
                    duck_active = False

        else:
            if duck_active:
                pyautogui.keyUp("down")
                duck_active = False

        prev_gesture = gesture   # store for rising-edge detection next frame

        # ---- HUD ----
        label, color = GESTURE_LABELS.get(gesture, ("---", (180,180,180)))
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 115), (310, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Gesture: {label}", (10, h - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        legend = [
            "Open palm        ->  JUMP",
            "Fist             ->  NEUTRAL",
            "Peace sign       ->  RESTART",
        ]
        for i, line in enumerate(legend):
            cv2.putText(frame, line, (10, h - 55 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

        cv2.putText(frame, "Q = quit", (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        cv2.imshow("Dino Gesture Controller", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if duck_active:
    pyautogui.keyUp("down")
cap.release()
cv2.destroyAllWindows()
                