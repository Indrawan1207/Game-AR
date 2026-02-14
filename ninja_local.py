import cv2
import mediapipe as mp
import numpy as np
import time
import random
import math
from collections import deque
import threading

# =========================
# ADVANCED SOUND ENGINE
# =========================
try:
    import winsound
    def play_sound(kind):
        def _play():
            try:
                if kind == "slice": winsound.Beep(880, 35)
                elif kind == "bomb": winsound.Beep(200, 180)
                elif kind == "combo": winsound.Beep(1200, 45)
                elif kind == "start":
                    for f in [440, 880, 1320]:
                        winsound.Beep(f, 55)
            except:
                pass
        threading.Thread(target=_play, daemon=True).start()
except:
    def play_sound(kind): pass

# =========================
# BRANDING & COLORS
# =========================
GAME_NAME = "AETHER BLADES"
COLOR_P1 = (255, 230, 0)     # Neon Cyan-ish
COLOR_P2 = (0, 140, 255)     # Neon Ember
COLOR_BOMB = (40, 40, 45)
COLOR_UI = (255, 255, 255)

WIDTH, HEIGHT = 1280, 720
NET_X = WIDTH // 2

# =========================
# PERFORMANCE TUNING
# =========================
TRACK_W, TRACK_H = 480, 270      # tracking kecil = lebih cepat
TARGET_FPS = 60.0

cv2.setUseOptimized(True)
cv2.setNumThreads(0)

# =========================
# CAMERA (LOW LATENCY)
# =========================
class Camera:
    def __init__(self, src=0, width=1280, height=720, backend=cv2.CAP_DSHOW):
        self.cap = cv2.VideoCapture(src, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.ok = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _loop(self):
        while self.running:
            if not self.cap.isOpened():
                self.ok = False
                time.sleep(0.01)
                continue

            # grab latest frame (reduce latency)
            grabbed = self.cap.grab()
            if not grabbed:
                self.ok = False
                time.sleep(0.005)
                continue

            ok, fr = self.cap.retrieve()
            if not ok:
                self.ok = False
                continue

            with self.lock:
                self.frame = fr
                self.ok = True

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        try:
            self.t.join(timeout=0.2)
        except:
            pass
        self.cap.release()

def window_closed(name):
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except:
        return True

# =========================
# HELPERS
# =========================
def clamp(v, a, b):
    return max(a, min(b, v))

def put_text(img, text, org, scale=1.0, color=(255,255,255), th=2, font=cv2.FONT_HERSHEY_DUPLEX):
    cv2.putText(img, text, org, font, scale, (0,0,0), th+3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, th, cv2.LINE_AA)

def glass_panel(img, x, y, w, h, title="", val=""):
    # lightweight "glass": blur + tint
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x+w), min(img.shape[0], y+h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(roi, (21, 21), 0)
    tint = np.full_like(blur, (20, 20, 25), dtype=np.uint8)
    panel = cv2.addWeighted(blur, 0.88, tint, 0.12, 0)
    img[y0:y1, x0:x1] = cv2.addWeighted(panel, 0.82, roi, 0.18, 0)

    cv2.rectangle(img, (x0, y0), (x1, y1), (255,255,255), 1, cv2.LINE_AA)
    if title:
        cv2.putText(img, title, (x+10, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(img, val, (x+10, y+70), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255,255,255), 2, cv2.LINE_AA)

def adaptive_smooth(old, new):
    """Fast movement => more responsive, slow => smoother."""
    if new is None:
        return old
    ox, oy = old
    nx, ny = new
    speed = math.hypot(nx-ox, ny-oy)
    a = 0.35 if speed < 18 else 0.78
    return (int((1-a)*ox + a*nx), int((1-a)*oy + a*ny))

# =========================
# CORE CLASSES
# =========================
class Particle:
    def __init__(self, x, y, color, is_splatter=False):
        self.x, self.y = float(x), float(y)
        self.color = color
        self.is_splatter = is_splatter
        self.vx = random.uniform(-10, 10)
        self.vy = random.uniform(-15, 5)
        self.life = 1.0
        self.size = random.randint(10, 22) if is_splatter else random.randint(3, 7)
        self.gravity = 0.35 if not is_splatter else 0.10

    def update(self, dt):
        step = dt * 60.0
        self.x += self.vx * step
        self.y += self.vy * step
        self.vy += self.gravity * step
        self.life -= 0.040 * step

class AetherFruit:
    def __init__(self, side):
        self.side = side
        self.radius = 42
        self.active = True
        self.is_bomb = random.random() < 0.15
        self.color = COLOR_P1 if side == 'p1' else COLOR_P2
        if self.is_bomb:
            self.color = COLOR_BOMB

        margin = 150
        if side == 'p1':
            self.x = random.randint(margin, WIDTH//2 - margin)
            self.vx = random.uniform(2.0, 5.0)
        else:
            self.x = random.randint(WIDTH//2 + margin, WIDTH - margin)
            self.vx = random.uniform(-5.0, -2.0)

        self.y = HEIGHT + 50
        self.vy = random.uniform(-23, -17)
        self.rot = 0.0
        self.rot_v = random.uniform(-5, 5)

    def update(self, dt):
        step = dt * 60.0
        self.x += self.vx * step
        self.y += self.vy * step
        self.vy += 0.55 * step
        self.rot += self.rot_v * step
        return self.y < HEIGHT + 100 and -200 < self.x < WIDTH + 200

    def draw(self, img):
        if not self.active:
            return
        cx, cy = int(self.x), int(self.y)

        # glow ring
        glow = img.copy()
        cv2.circle(glow, (cx, cy), self.radius+12, tuple(int(c*0.25) for c in self.color), 3, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.25, img, 0.75, 0, img)

        cv2.circle(img, (cx, cy), self.radius, self.color, -1, cv2.LINE_AA)
        cv2.circle(img, (cx-12, cy-12), 9, (255,255,255), -1, cv2.LINE_AA)

        if self.is_bomb:
            put_text(img, "X", (cx-14, cy+12), 1.1, (0,0,255), 3, cv2.FONT_HERSHEY_SIMPLEX)

# =========================
# HAND TRACKING (FAST + STABLE)
# =========================
def palm_center(hand_lms, sx, sy):
    # Palm center average: 0,5,9,13,17
    idxs = [0, 5, 9, 13, 17]
    xs = [hand_lms.landmark[i].x for i in idxs]
    ys = [hand_lms.landmark[i].y for i in idxs]
    x = int(np.mean(xs) * TRACK_W * sx)
    y = int(np.mean(ys) * TRACK_H * sy)
    return x, y

def index_tip(hand_lms, sx, sy):
    x = int(hand_lms.landmark[8].x * TRACK_W * sx)
    y = int(hand_lms.landmark[8].y * TRACK_H * sy)
    return x, y

def pick_best_per_side(landmarks, sx, sy):
    """
    Return tips dict: {'p1':(x,y) or None, 'p2':(x,y) or None}
    Strategy:
    - Compute palm center for each detected hand
    - Assign by palm center x < NET => p1 else p2
    - If multiple hands fall in same side, pick the one closest to that side center
    """
    cand = {'p1': [], 'p2': []}
    for hlm in landmarks:
        px, py = palm_center(hlm, sx, sy)
        tx, ty = index_tip(hlm, sx, sy)
        side = 'p1' if px < NET_X else 'p2'
        cand[side].append(((tx, ty), (px, py)))

    tips = {'p1': None, 'p2': None}
    if cand['p1']:
        target = (WIDTH//4, HEIGHT//2)
        tips['p1'] = min(cand['p1'], key=lambda item: (item[1][0]-target[0])**2 + (item[1][1]-target[1])**2)[0]
    if cand['p2']:
        target = (3*WIDTH//4, HEIGHT//2)
        tips['p2'] = min(cand['p2'], key=lambda item: (item[1][0]-target[0])**2 + (item[1][1]-target[1])**2)[0]
    return tips

# =========================
# GAME ENGINE
# =========================
class AetherEngine:
    def __init__(self):
        # Low-latency cam for laptop webcam
        self.cam = Camera(0, width=WIDTH, height=HEIGHT, backend=cv2.CAP_DSHOW)

        # MediaPipe lightweight
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.state = "MENU"  # MENU, PLAY, GAMEOVER
        self.reset_data()

    def reset_data(self):
        self.score = {'p1': 0, 'p2': 0}
        self.combo = {'p1': 0, 'p2': 0}
        self.last_hit = {'p1': 0.0, 'p2': 0.0}
        self.fruits = []
        self.particles = []
        self.trails = {'p1': deque(maxlen=15), 'p2': deque(maxlen=15)}
        self.shake = 0
        self.timer = 60
        self.start_time = 0.0
        self._spawn_cd = 0.0

    def apply_camera_shake(self, img):
        if self.shake > 0:
            dx = random.randint(-self.shake, self.shake)
            dy = random.randint(-self.shake, self.shake)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img = cv2.warpAffine(img, M, (WIDTH, HEIGHT))
            self.shake = max(0, self.shake - 2)
        return img

    def run(self):
        cv2.namedWindow(GAME_NAME, cv2.WINDOW_NORMAL)
        prev = time.perf_counter()

        while True:
            if window_closed(GAME_NAME):
                break

            ok, frame = self.cam.read()
            if not ok or frame is None:
                continue

            frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            dt = now - prev
            prev = now
            dt = clamp(dt, 0.0, 0.05)

            # 1) MediaPipe on SMALL frame (speed)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            small = cv2.resize(rgb, (TRACK_W, TRACK_H))
            results = self.hands.process(small)

            sx = WIDTH / TRACK_W
            sy = HEIGHT / TRACK_H

            # 2) Base render
            display = frame.copy()
            display = self.apply_camera_shake(display)

            # net line
            cv2.line(display, (NET_X, 0), (NET_X, HEIGHT), (255,255,255), 1, cv2.LINE_AA)

            # 3) Hands -> stable tips per side
            tips = {'p1': None, 'p2': None}
            if results.multi_hand_landmarks:
                tips = pick_best_per_side(results.multi_hand_landmarks, sx, sy)

            # 4) Update trails (with smoothing)
            for side in ['p1', 'p2']:
                if tips[side] is not None:
                    if len(self.trails[side]) == 0:
                        self.trails[side].append(tips[side])
                    else:
                        sm = adaptive_smooth(self.trails[side][-1], tips[side])
                        self.trails[side].append(sm)

            # draw trail
            for side in ['p1', 'p2']:
                pts = list(self.trails[side])
                if len(pts) > 1:
                    color = COLOR_P1 if side == 'p1' else COLOR_P2
                    for i in range(1, len(pts)):
                        cv2.line(display, pts[i-1], pts[i], color, i, cv2.LINE_AA)

            # 5) State
            if self.state == "MENU":
                self.draw_menu(display, tips)
            elif self.state == "PLAY":
                self.update_play(display, tips, dt)
            elif self.state == "GAMEOVER":
                self.draw_game_over(display)

            # 6) Input
            cv2.imshow(GAME_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            if key == ord('r'):
                self.state = "MENU"
                self.reset_data()

        self.cam.release()
        cv2.destroyAllWindows()

    def draw_menu(self, img, tips):
        img[:] = cv2.GaussianBlur(img, (25, 25), 0)
        put_text(img, GAME_NAME, (WIDTH//2-300, HEIGHT//2-120), 2.6, (255,255,255), 5, cv2.FONT_HERSHEY_TRIPLEX)

        cx, cy = WIDTH//2, HEIGHT//2 + 60
        cv2.circle(img, (cx, cy), 64, (255,255,255), 2, cv2.LINE_AA)
        put_text(img, "SLICE TO START", (cx-120, cy+110), 0.95, (255,255,255), 2, cv2.FONT_HERSHEY_SIMPLEX)
        put_text(img, "R: RESET    Q/ESC: QUIT", (cx-150, cy+150), 0.75, (200,200,200), 2, cv2.FONT_HERSHEY_SIMPLEX)

        # detect slice hit start button
        for side in ['p1', 'p2']:
            if tips[side]:
                if math.hypot(tips[side][0]-cx, tips[side][1]-cy) < 64:
                    play_sound("start")
                    self.state = "PLAY"
                    self.start_time = time.time()
                    return

    def update_play(self, img, tips, dt):
        elapsed = time.time() - self.start_time
        self.timer = max(0, 60 - int(elapsed))
        if self.timer <= 0:
            self.state = "GAMEOVER"
            return

        # Spawn with cooldown (lebih stabil daripada random murni)
        self._spawn_cd -= dt
        if self._spawn_cd <= 0 and len(self.fruits) < 10:
            self.fruits.append(AetherFruit('p1'))
            self.fruits.append(AetherFruit('p2'))
            self._spawn_cd = random.uniform(0.25, 0.40)  # rate spawn

        # Update Fruits + collision
        for f in self.fruits[:]:
            if not f.update(dt):
                self.fruits.remove(f)
                continue

            f.draw(img)

            for side in ['p1', 'p2']:
                if tips[side] is None:
                    continue

                # territory rule
                if (side == 'p1' and f.x >= NET_X) or (side == 'p2' and f.x <= NET_X):
                    continue

                if math.hypot(tips[side][0]-f.x, tips[side][1]-f.y) < f.radius + 18:
                    self.handle_slice(f, side)

        # Update Particles (lebih ringan)
        new_particles = []
        for p in self.particles:
            p.update(dt)
            if p.life > 0:
                # draw
                overlay = img.copy()
                cv2.circle(overlay, (int(p.x), int(p.y)), p.size, p.color, -1, cv2.LINE_AA)
                cv2.addWeighted(overlay, clamp(p.life, 0, 1), img, 1-clamp(p.life, 0, 1), 0, img)
                new_particles.append(p)
        self.particles = new_particles

        # HUD
        glass_panel(img, 50, 20, 220, 100, "PLAYER 1", str(self.score['p1']))
        glass_panel(img, WIDTH-270, 20, 220, 100, "PLAYER 2", str(self.score['p2']))
        put_text(img, f"{self.timer}s", (WIDTH//2-35, 55), 1.6, (255,255,255), 2)

        # Combo label
        for s in ['p1', 'p2']:
            if self.combo[s] > 3:
                txt = "AETHER!!" if self.combo[s] > 8 else "COMBO!"
                x_pos = 90 if s == 'p1' else WIDTH-290
                put_text(img, f"{txt} x{self.combo[s]}", (x_pos, 155), 0.95, (0,255,255), 2, cv2.FONT_HERSHEY_TRIPLEX)

    def handle_slice(self, f, side):
        if not f.active:
            return
        f.active = False
        if f in self.fruits:
            self.fruits.remove(f)

        if f.is_bomb:
            play_sound("bomb")
            self.score[side] = max(0, self.score[side] - 50)
            self.shake = 26
            for _ in range(14):
                self.particles.append(Particle(f.x, f.y, (50,50,50), is_splatter=False))
        else:
            play_sound("slice")
            now = time.time()
            if now - self.last_hit[side] < 0.8:
                self.combo[side] += 1
                if self.combo[side] in (4, 7, 10):
                    play_sound("combo")
            else:
                self.combo[side] = 1
            self.last_hit[side] = now

            mult = 1.0 + (self.combo[side] * 0.10)
            self.score[side] += int(10 * mult)

            for _ in range(10):
                self.particles.append(Particle(f.x, f.y, f.color, is_splatter=True))

    def draw_game_over(self, img):
        img[:] = cv2.GaussianBlur(img, (41, 41), 0)
        winner = "DRAW"
        if self.score['p1'] > self.score['p2']:
            winner = "PLAYER 1 DOMINATES"
        elif self.score['p2'] > self.score['p1']:
            winner = "PLAYER 2 DOMINATES"

        put_text(img, "MATCH CONCLUDED", (WIDTH//2-270, HEIGHT//2-70), 1.7, (255,255,255), 3, cv2.FONT_HERSHEY_TRIPLEX)
        put_text(img, winner, (WIDTH//2-240, HEIGHT//2+25), 1.25, (0,255,255), 2, cv2.FONT_HERSHEY_SIMPLEX)
        put_text(img, "PRESS 'R' TO REMATCH", (WIDTH//2-210, HEIGHT//2+135), 0.85, (200,200,200), 2, cv2.FONT_HERSHEY_SIMPLEX)

if __name__ == "__main__":
    game = AetherEngine()
    game.run()