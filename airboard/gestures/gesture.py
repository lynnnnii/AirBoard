"""
AirBoard gesture recognition module (contributor: Kim Hyunbo)

Pure recognition logic operating on landmark arrays; testable without MediaPipe or a webcam.
Hand detection is handled by airboard.tracking.HandTracker.

Other modules should consume only GestureRecognizer.update(hands, frame_w, frame_h, t).
  - result.pen_point   : Writing position (camera pixels), passed to coordinate mapping
  - result.pen_down    : True while writing; False when the pen is lifted
  - result.pose        : Current hand pose ("PINCH", "FIST", "POINTER", "TWO_FINGERS", "OPEN", "NONE")
  - result.events      : One-shot events triggered in this frame ("DOUBLE_CLAP", "SWIPE_LEFT", "SWIPE_RIGHT")
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Landmark indices (21 MediaPipe hand joints)
#   0: wrist
#   1~4  : thumb (CMC, MCP, IP, TIP)
#   5~8  : index finger (MCP, PIP, DIP, TIP)
#   9~12 : middle finger / 13–16: ring finger / 17–20: little finger
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP = 9
PINKY_MCP = 17
FINGERS = {  # Finger name: (MCP, TIP)
    "index": (5, 8),
    "middle": (9, 12),
    "ring": (13, 16),
    "pinky": (17, 20),
}

# ---------------------------------------------------------------------------
# Tuning parameters (calibrate through hand gesture experiments)
# ---------------------------------------------------------------------------
CFG = {
    # Pinch hysteresis: ratio = thumb–index tip distance / palm size
    "pinch_on": 0.25,        # Start writing below this threshold
    "pinch_off": 0.35,       # Stop writing above this threshold
    "pinch_frames": 2,       # Require this many consecutive frames for a state transition

    # 손가락 폄: 굽힘 비율 = wrist-손가락끝 거리 / wrist-손가락MCP 거리
    "curl_extended": 1.55,   # Extended when above this threshold
    # Folding ratio = fingertip-to-palm-center distance / palm size
    #   Fingertips move toward the palm in a fist, reducing this ratio.
    #   (Extended finger ~1.6+, pinching index ~0.9–1.2, fist ~0.3–0.7)
    "fold_tip": 0.8,         # Considered folded below this threshold
    "fist_min_folded": 5,    # Require all five fingers, including the thumb, to be folded
    "fist_thumb_fold": 0.9,  # Thumb is folded if its tip-to-palm distance / palm size is below this value
    # 검지 뻗음: 검지끝이 검지MCP보다 손가락 방향(wrist→중지MCP)으로 얼마나 나가 있는지 / 손바닥 크기
    #   주먹은 검지 끝이 손바닥 안으로 말려 MCP보다 wrist 쪽에 있고 (0 이하),
    #   while a pinch leaves the C-shaped index tip beyond its MCP (positive reach).
    "fist_index_reach": 0.15,# Fist requires reach below this value to distinguish it from pinch
    "fist_frames": 2,        # Require this many consecutive fist frames before erasing

    # Clap: palm-center distance / average palm size
    "clap_close": 1.6,       # Treat palms as touching below this ratio
    "clap_open": 2.0,        # Require separation beyond this ratio before counting another clap
    "clap_near": 3.0,        # If hands disappear within this distance, assume temporary overlap/occlusion
    "clap_grace": 0.4,       # Treat missing nearby hands as touching for this many seconds
    "clap_window": 1.3,      # Require two claps within this many seconds for DOUBLE_CLAP
    "clap_min_gap": 0.12,    # Ignore too-short gaps that might count one clap twice
    "clap_cooldown": 1.0,    # Ignore claps for this many seconds after DOUBLE_CLAP

    # 스와이프: 손을 편 상태에서 wrist이 빠르게 좌우로 이동
    "swipe_window": 0.5,     # Measure movement over this many seconds
    "swipe_dist": 0.18,      # Swipe if displacement exceeds this fraction of frame width
    "swipe_max_slope": 0.7,  # Reject when vertical/horizontal movement ratio exceeds this threshold
    "swipe_open_fingers": 3, # Open hand requires at least this many of the four fingers extended
    "swipe_open_curl": 1.4,  # Extension threshold for swipes (looser than normal)
    "swipe_open_ratio": 0.6, # Require this fraction of window frames to show an open hand (allowing jitter)
    "swipe_lost_grace": 0.25,# Keep history during brief tracking loss up to this many seconds
    "swipe_cooldown": 0.8,
    "mode_flash": 0.8,       # NEXT PAGE / PREV PAGE indicator duration (seconds)
}


# ---------------------------------------------------------------------------
# Basic geometry helpers
# ---------------------------------------------------------------------------
def dist(a, b):
    return float(np.linalg.norm(a - b))


def palm_size(lm):
    """손바닥 크기 = wrist(0) ~ 중지 MCP(9) 거리. 카메라와의 거리 보정에 사용."""
    return dist(lm[WRIST, :2], lm[MIDDLE_MCP, :2]) + 1e-6


def palm_center(lm):
    return lm[[0, 5, 9, 13, 17], :2].mean(axis=0)


def pinch_ratio(lm):
    """Normalize the 2D thumb-to-index tip distance by the 3D palm size.
    Use 2D numerator: fingertip z is noisy and may prevent pinch detection even with touching fingertips.
    분모는 3D: 2D 손바닥 크기는 손을 눕히면 짧아져 비율이 커진다. wrist·MCP의 z는 비교적 안정적이다."""
    return dist(lm[THUMB_TIP, :2], lm[INDEX_TIP, :2]) / (dist(lm[WRIST], lm[MIDDLE_MCP]) + 1e-6)


def finger_curls(lm):
    """Finger curl ratio: larger indicates extension; use distance ratios rather than y coordinates for tilt robustness."""
    w = lm[WRIST, :2]
    return {name: dist(w, lm[tip, :2]) / (dist(w, lm[mcp, :2]) + 1e-6)
            for name, (mcp, tip) in FINGERS.items()}


def tip_to_palm(lm):
    """Normalized distance of each fingertip from palm center; smaller means more folded."""
    c = palm_center(lm)
    s = palm_size(lm)
    return {name: dist(lm[tip, :2], c) / s for name, (mcp, tip) in FINGERS.items()}


def index_reach(lm):
    """검지끝이 검지MCP보다 손가락 방향(wrist→중지MCP)으로 나가 있는 정도. 음수면 손바닥 안으로 말림.
    Project onto the palm axis rather than screen y for tilt invariance."""
    axis = lm[MIDDLE_MCP, :2] - lm[WRIST, :2]
    s = palm_size(lm)
    return float(np.dot(lm[INDEX_TIP, :2] - lm[INDEX_MCP, :2], axis / s)) / s


def classify_pose(lm, curls, tips=None, config=None):
    """Classify static single-hand pose; PinchDetector handles pinches separately."""
    tips = tip_to_palm(lm) if tips is None else tips
    folded = {k: v < CFG["fold_tip"] for k, v in tips.items()}
    extended = {k: v > CFG["curl_extended"] for k, v in curls.items()}

    cfg = {**CFG, **(config or {})}
    # Undo/redo needs a clearly extended, straight thumb pointed sideways.
    thumb = lm[THUMB_TIP, :2] - lm[2, :2]
    thumb_x, thumb_y = float(thumb[0]) / palm_size(lm), float(thumb[1]) / palm_size(lm)
    thumb_reach = dist(lm[THUMB_TIP, :2], lm[INDEX_MCP, :2]) / palm_size(lm)
    thumb_straight = dist(lm[2, :2], lm[THUMB_TIP, :2]) / (
        dist(lm[2, :2], lm[3, :2]) + dist(lm[3, :2], lm[THUMB_TIP, :2]) + 1e-6)
    # Redo is designed for the user's right hand: the thumb alone points right,
    # since the curled fingers are naturally hidden from the camera in that pose.
    # Undo keeps the curled-finger check to stay distinct from the eraser fist.
    thumb_sideways = (thumb_reach > cfg.get('history_thumb_reach', .8)
            and abs(thumb_x) > cfg.get('history_thumb_x_min', .5)
            and abs(thumb_x) > cfg.get('history_thumb_side_ratio', 1.3) * abs(thumb_y)
            and thumb_straight > cfg.get('history_thumb_straight', .85))
    if (thumb_sideways and thumb_x > 0
            and sum(folded.values()) >= cfg.get('history_redo_min_folded_fingers', 0)):
        return "THUMB_RIGHT"
    if (thumb_sideways and thumb_x < 0
            and sum(folded.values()) >= cfg.get('history_min_folded_fingers', 3)):
        return "THUMB_LEFT"

    # 1) Fist: index curled and at least fist_min_folded fingers, including the thumb, folded.
    #    Check fist before pinch because the thumb/index fingertips can also be close.
    #    When pinching with other fingers folded, the index tip also approaches palm center, so folding alone is insufficient.
    #    Also check index_reach to distinguish a truly curled index tip from a pinch.
    thumb_tip_to_palm = dist(lm[THUMB_TIP, :2], palm_center(lm)) / palm_size(lm)
    thumb_folded = thumb_tip_to_palm <= cfg.get("fist_thumb_fold", CFG["fist_thumb_fold"])
    fist_folded_count = sum(folded.values()) + int(thumb_folded)
    if folded["index"] and fist_folded_count >= cfg.get("fist_min_folded", CFG["fist_min_folded"]) \
            and index_reach(lm) < CFG["fist_index_reach"]:
        return "FIST"
    # 2) Pointer: only index extended, other fingers folded, and thumb separated from index
    if extended["index"] and folded["middle"] and folded["ring"] and folded["pinky"] \
            and pinch_ratio(lm) > CFG["pinch_off"]:
        return "POINTER"
    # 3) Open palm: all four fingers extended
    if all(extended.values()):
        return "OPEN"
    return "NONE"


# ---------------------------------------------------------------------------
# State detectors
# ---------------------------------------------------------------------------
class PinchDetector:
    """Detect pen-down/up using hysteresis and frame debouncing."""

    def __init__(self, config=None):
        self.config = {**CFG, **(config or {})}
        self.down = False
        self.count = 0
        self.armed = True

    def update(self, ratio, blocked=False, observe_release=False):
        if ratio is None or not np.isfinite(ratio):
            self.down, self.count = False, 0
            self.armed = False
            return False
        if blocked:
            self.down, self.count = False, 0
            if observe_release:
                self.armed = self.armed or ratio>self.config['pinch_off']
            else:
                self.armed = False
            return False
        if not self.armed:
            if ratio > self.config['pinch_off']:
                self.armed = True
            return False
        if self.down and ratio>=self.config.get('pinch_immediate_off',float('inf')):
            self.down,self.count=False,0
            return False
        want = (ratio < self.config['pinch_on']) if not self.down else not (ratio > self.config['pinch_off'])
        if want != self.down:
            self.count += 1
            key = 'pinch_on_frames' if want else 'pinch_off_frames'
            if self.count >= self.config.get(key,self.config['pinch_frames']):
                self.down, self.count = want, 0
        else:
            self.count = 0
        return self.down


class ClapDetector:
    """양손 손바닥 중심 거리가 가까워졌다 벌어지는 것을 박수 1회로 세고, 짧은 시간 안에 2회면 DOUBLE_CLAP.

    실제 박수에서는 두 손이 맞닿는 순간 MediaPipe가 한 손 또는 두 손 모두를 놓치는 경우가 많다.
    그래서 '직전에 두 손이 가까이 있었는데 갑자기 손이 사라짐'도 맞닿음으로 처리한다.
    """

    def __init__(self, config=None):
        self.config = {**CFG, **(config or {})}
        self.touching = False
        self.last_d = None
        self.last_near_t = -1e9
        self.claps = deque()
        self.cooldown_until = 0.
        self.armed = False
        self.open_distance = 0.
        self.approaching = False

    def _register_touch(self,t):
        if not self.armed or self.touching:
            return
        self.armed=False
        self.touching=True
        if not self.claps or t-self.claps[-1]>self.config['clap_min_gap']:
            self.claps.append(t)

    def update(self,hands,t):
        cfg=self.config
        while self.claps and t-self.claps[0]>cfg['clap_window']:
            self.claps.popleft()
        if t<self.cooldown_until:
            return False
        if len(hands)>=2:
            a,b=hands[:2]
            scale=(palm_size(a)+palm_size(b))/2
            d=dist(palm_center(a),palm_center(b))/scale
            self.last_d=d
            if d>cfg['clap_open']:
                if not self.armed:
                    self.open_distance=d
                else:
                    self.open_distance=max(self.open_distance,d)
                self.armed=True
                self.touching=False
                self.approaching=self.open_distance-d>=cfg.get('clap_approach_drop',.35)
            elif self.armed:
                self.approaching=self.open_distance-d>=cfg.get('clap_approach_drop',.35)
                if d<cfg['clap_close'] and self.approaching:
                    self._register_touch(t)
            if d<cfg['clap_near']:
                self.last_near_t=t
        elif (t-self.last_near_t<cfg['clap_grace'] and self.armed
              and (self.approaching or (self.last_d is not None
                   and self.last_d<cfg.get('clap_occlusion_close',2.5)))):
            # 양손이 가까워지다 가려진 경우만 접촉으로 인정한다.
            self._register_touch(t)
        elif t-self.last_near_t>=cfg.get('clap_reacquire_seconds',.85):
            self.armed=False
            self.approaching=False
            self.last_d=None
        if len(self.claps)>=2:
            self.claps.clear()
            self.armed=False
            self.touching=False
            self.cooldown_until=t+cfg['clap_cooldown']
            return True
        return False


class SwipeDetector:
    """손바닥을 편 상태에서 손이 빠르게 좌우로 이동하면 스와이프.

    인식률을 높이기 위한 처리
      - wrist 대신 손바닥 중심을 추적 (더 안정적)
      - 펼친 한 손의 이동을 판정하며 도구 동작/손 소실 시 이력을 초기화
      - 짧은 손가락 검출 흔들림은 허용하지만 해당 프레임 좌표는 기록하지 않음
      - 재입력 방지 시간의 이동은 다음 스와이프 이력에 포함하지 않음
      - 세로 이동은 절대값이 아니라 가로 이동 대비 비율로 제한 (약간 대각선이어도 인정)
    """

    def __init__(self, config=None):
        self.config = {**CFG, **(config or {})}
        self.hist = deque()          # (t, x, y, is_open)
        self.last_seen = -1e9
        self.cooldown_until = 0.0
        self._last_swipe = None      # (방향, 시각) 반대 방향 되돌림 무시용

    def update(self, point, is_open, frame_w, frame_h, t):
        # 도구 동작·양손·손 소실 구간의 이동을 스와이프 이력에 섞지 않는다.
        if point is None or t < self.cooldown_until:
            self.hist.clear()
            return None
        if not is_open:
            # 잠깐의 손가락 검출 흔들림은 넘기되 도구 입력은 호출자가 None으로 차단한다.
            if t-self.last_seen > self.config.get('swipe_open_grace', .1):
                self.hist.clear()
            return None
        self.last_seen = t
        self.hist.append((t, point[0] / frame_w, point[1] / frame_h, bool(is_open)))
        while self.hist and t - self.hist[0][0] > self.config["swipe_window"]:
            self.hist.popleft()
        if t < self.cooldown_until or len(self.hist) < 3:
            return None

        open_ratio = sum(h[3] for h in self.hist) / len(self.hist)
        if open_ratio < self.config["swipe_open_ratio"]:
            return None
        dx = self.hist[-1][1] - self.hist[0][1]
        dy = abs(self.hist[-1][2] - self.hist[0][2])
        if abs(dx) > self.config["swipe_dist"] and dy < self.config["swipe_max_slope"] * abs(dx):
            self.hist.clear()
            event = "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
            # 스와이프 뒤 손을 제자리로 되돌리는 동작이 반대 방향 스와이프로 잡히지 않게,
            # swipe_reverse_guard 초 동안은 반대 방향을 무시한다 (같은 방향 연속 스와이프는 그대로 허용).
            if (self._last_swipe is not None and event != self._last_swipe[0]
                    and t - self._last_swipe[1] < self.config.get('swipe_reverse_guard', 0.)):
                return None
            self.cooldown_until = t + self.config["swipe_cooldown"]
            self._last_swipe = (event, t)
            return event
        return None


# ---------------------------------------------------------------------------
# 결과 구조체와 메인 인식기
# ---------------------------------------------------------------------------
@dataclass
class GestureResult:
    hands: list = field(default_factory=list)   # 각 손의 (21,3) 픽셀 좌표 배열
    main_hand: np.ndarray = None                 # 필기에 쓰는 손 (가장 크게 보이는 손)
    pose: str = "NONE"                           # PINCH / FIST / POINTER / TWO_FINGERS / OPEN / NONE
    pen_down: bool = False                       # True = 쓰는 중 (화면에는 WRITE ON으로 표시)
    pen_point: tuple = None                      # 필기 좌표 (엄지끝과 검지끝의 중간점)
    pointer_point: tuple = None                  # 검지 끝 좌표 (포인터·지우개 위치)
    events: list = field(default_factory=list)   # DOUBLE_CLAP, SWIPE_LEFT, SWIPE_RIGHT
    debug: dict = field(default_factory=dict)    # 튜닝용 수치 (핀치 비율, 굽힘 비율 등)


class GestureRecognizer:
    """랜드마크 리스트를 받아 프레임마다 GestureResult를 낸다."""

    def __init__(self, config=None):
        self.pinch = PinchDetector(config)
        self.config = {**CFG, **(config or {})}
        self.clap = ClapDetector(config)
        self._swipe_blocked_until = 0.
        self._ink_blocked_until = 0.
        self.swipe = SwipeDetector(config)
        self.fist_count = 0
        self._last_ink_point = None
        self._prev_pose = 'NONE'     # 직전 프레임 자세 (검지 자세 유지 판정용)
        self._prev_history_pose = None
        self._history_cooldown_until = float('-inf')

    def update(self, hands, frame_w, frame_h, t):
        """hands: HandTracker.detect() 결과 (손마다 (21,3) 픽셀 좌표). t: 초 단위 시각."""
        r = GestureResult(hands=hands)

        # 박수는 양손 정보로 먼저 판단
        if self.clap.update(hands, t):
            r.events.append("DOUBLE_CLAP")
        if len(hands)>=2 or 'DOUBLE_CLAP' in r.events:
            self._swipe_blocked_until=t+self.config.get('clap_swipe_guard',.7)
            self._ink_blocked_until=t+self.config.get('clap_ink_guard',.12)
        ink_busy=(len(hands)>=2 or t<self._ink_blocked_until)
        clap_busy=(len(hands)>=2 or bool(self.clap.claps)
                   or t<self.clap.cooldown_until or t<self._swipe_blocked_until)

        if not hands:
            self.pinch.update(None)
            self._prev_pose = 'NONE'
            self._prev_history_pose = None
            self._last_ink_point = None
            self.fist_count = 0
            self.swipe.update(None, False, frame_w, frame_h, t)
            r.debug = {"hand_count": 0, "clap_touch": self.clap.touching, "clap_count": len(self.clap.claps)}
            return r

        # 필기 손 = 가장 크게 보이는 손 (카메라에 가장 가까운 발표자 손)
        lm = max(hands, key=palm_size)
        r.main_hand = lm
        curls = finger_curls(lm)
        tips = tip_to_palm(lm)
        pose = classify_pose(lm, curls, tips, self.config)
        if pose in ('THUMB_LEFT', 'THUMB_RIGHT'):
            if pose != self._prev_history_pose and t >= self._history_cooldown_until:
                r.events.append(pose)
                self._history_cooldown_until = t + self.config.get('history_gesture_cooldown', .55)
            self._prev_history_pose = pose
        else:
            self._prev_history_pose = None
        ratio = pinch_ratio(lm)
        # 아래 두 판정은 UI가 매핑을 보고 켤 때만 쓴다. 핀치로 쓰는 기본 설정에서는 원래 판정과 똑같이 동작해야 한다.
        #  - index_writing: 검지로 쓰는 설정 (검지 자세 유지 + ✌️ 인식)
        #  - two_fingers: ✌️를 다른 동작에 연결한 설정 (✌️ 인식만)
        # 두 판정 모두 엄지가 검지에 붙은 핀치 시작 구간(pinch_on 미만)과 핀치 중에는 끼어들지 않고,
        # 핀치 판정기를 막지도 않는다. 그래서 검지로 쓰다가도 엄지를 붙이면 바로 핀치로 넘어간다.
        index_mode = self.config.get('index_writing', False)
        pinch_free = not self.pinch.down and ratio > self.config['pinch_on']
        # 검지·중지 두 손가락 펴기(✌️): 검지로 쓰다가 펜을 드는 '멈춤' 제스처. 약지·새끼는 접혀 있어야 한다.
        # 두 손가락 사이 간격은 보지 않는다.
        two = self.config.get('two_finger_curl', CFG['curl_extended'])
        if ((index_mode or self.config.get('two_fingers', False)) and pinch_free and pose == 'NONE'
                and curls['index'] > two and curls['middle'] > two
                and all(tips[k] < self.config.get('two_finger_fold', CFG['fold_tip']) for k in ('ring', 'pinky'))):
            pose = 'TWO_FINGERS'
        # 검지 자세가 이어지는 중에는 기준을 느슨하게 유지한다(히스테리시스). 글씨를 쓸 때는 검지가 살짝 굽고
        # 엄지가 검지 쪽에 붙기 쉬워서, 처음 판정 기준(검지 펴짐 1.55, 엄지 간격 0.38)을 계속 요구하면 자꾸 끊긴다.
        kept_pointer = (index_mode and pinch_free and pose == 'NONE' and self._prev_pose == 'POINTER'
                        and curls['index'] > self.config.get('pointer_keep_curl', CFG['curl_extended'])
                        and all(tips[k] < self.config.get('pointer_keep_fold', CFG['fold_tip'])
                                for k in ('middle', 'ring', 'pinky'))
                        and ratio > self.config.get('pointer_keep_pinch', CFG['pinch_off']))
        if kept_pointer:
            pose = 'POINTER'

        # 쓰는 중 닫힌 핀치는 유지한다. 지우개 전환은 pen-up 후에만 허용한다.
        if (pose=='FIST' and self.pinch.down
                and ratio<=self.pinch.config['pinch_off']):
            pose='NONE'
        # 핀치한 손의 나머지 손가락을 접는 자세와 주먹을 구분한다.
        if pose=='FIST' and tips['index']>self.config.get('fist_index_fold',.65):
            pose='NONE'
        fist_candidate = pose=="FIST"
        self.fist_count = self.fist_count + 1 if fist_candidate else 0
        is_fist = self.fist_count >= self.config.get('fist_frames',CFG['fist_frames'])
        if pose == "FIST" and not is_fist:
            pose = "NONE"

        # 양손이 보이면(박수 중일 수 있음) 필기를 막는다
        r.pen_down = self.pinch.update(
            ratio, blocked=(pose == "FIST" or ink_busy or (fist_candidate and not self.pinch.down)),
            observe_release=(ink_busy and len(hands)==1 and not fist_candidate))
        if r.pen_down:
            pose = "PINCH"
        if ink_busy:
            pose = 'NONE'
        r.pose = pose
        self._prev_pose = pose

        r.pen_point = tuple(((lm[THUMB_TIP, :2] + lm[INDEX_TIP, :2]) / 2).astype(int))
        # 해제 확인 중이거나, 손가락이 벌어지기 시작하면(비율 > pinch_hold) 손 이동을 잉크에 추가하지 않는다.
        # 펜이 실제로 들리기(pinch_off) 전에 손을 떼면서 움직인 길이 선으로 남지 않게 한다.
        # pinch_hold는 핀치로 쓸 때만 쓴다(pinch_writing, UI가 매핑을 보고 끈다). 핀치를 지우개 등에 연결했을 때
        # 손가락을 살짝 벌렸다고 지우개 위치까지 멈추면 안 되기 때문이다. 해제 확인 중 고정은 원래 동작 그대로 둔다.
        hold = self.config.get('pinch_hold') if self.config.get('pinch_writing', True) else None
        lifting = self.pinch.count>0 or (hold is not None and ratio>hold)
        if r.pen_down:
            if lifting and self._last_ink_point is not None:
                r.pen_point = self._last_ink_point
            else:
                self._last_ink_point = r.pen_point
        else:
            self._last_ink_point = None
        r.pointer_point = tuple(lm[INDEX_TIP, :2].astype(int)) if pose != "FIST" \
            else tuple(palm_center(lm).astype(int))

        # Page-swipe gestures are disabled; hand motion must not emit swipe events.
        r.debug = {"hand_count": len(hands),
                   "swipe_open_count": 0,
                   "clap_ink_blocked": ink_busy,
                   "clap_navigation_blocked": clap_busy,
                   "pinch_armed": self.pinch.armed,
                   "fist_confirm_frames": self.fist_count,
                   "clap_count": len(self.clap.claps),
                   "clap_armed": self.clap.armed,
                   "clap_approaching": self.clap.approaching,
                   "pinch_ratio": round(ratio, 2),
                   "pinch_3d": round(dist(lm[THUMB_TIP], lm[INDEX_TIP]) / (dist(lm[WRIST], lm[MIDDLE_MCP]) + 1e-6), 2),
                   "index_reach": round(index_reach(lm), 2),
                   "fold(tip-palm)": " ".join(f"{v:.2f}" for v in tips.values()),
                   "curl": " ".join(f"{v:.2f}" for v in curls.values()),
                   "clap_d": None if self.clap.last_d is None else round(self.clap.last_d, 2),
                   "clap_touch": self.clap.touching,
                   "clap_count": len(self.clap.claps)}
        return r
