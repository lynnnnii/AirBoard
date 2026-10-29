"""
AirBoard - 제스처 판정 모듈 (담당: Kim Hyunbo)

랜드마크 배열만 입력으로 받는 순수 판정 로직. MediaPipe·웹캠 없이 테스트할 수 있다.
손 인식은 airboard.tracking.HandTracker가 맡는다.

다른 팀원은 GestureRecognizer.update(hands, frame_w, frame_h, t) 결과만 받아서 쓰면 된다.
  - result.pen_point   : 필기 좌표 (카메라 픽셀 좌표) -> 좌표 변환 담당에게 전달
  - result.pen_down    : True면 쓰는 중, False면 펜 든 상태
  - result.pose        : 현재 손 모양 ("PINCH", "FIST", "POINTER", "OPEN", "NONE")
  - result.events      : 이번 프레임에 발생한 1회성 이벤트 ("DOUBLE_CLAP", "SWIPE_LEFT", "SWIPE_RIGHT")
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# 랜드마크 번호 (MediaPipe Hands 21개 관절)
#   0: 손목
#   1~4  : 엄지 (CMC, MCP, IP, TIP)
#   5~8  : 검지 (MCP, PIP, DIP, TIP)
#   9~12 : 중지 / 13~16 : 약지 / 17~20 : 새끼
# ---------------------------------------------------------------------------
WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_TIP = 5, 8
MIDDLE_MCP = 9
PINKY_MCP = 17
FINGERS = {  # 손가락 이름: (MCP, TIP)
    "index": (5, 8),
    "middle": (9, 12),
    "ring": (13, 16),
    "pinky": (17, 20),
}

# ---------------------------------------------------------------------------
# 튜닝 값 (직접 손으로 실험하면서 조정할 것)
# ---------------------------------------------------------------------------
CFG = {
    # 핀치(필기) 히스테리시스: 비율 = 엄지끝-검지끝 거리 / 손바닥 크기
    "pinch_on": 0.25,        # 이보다 작아지면 쓰기 시작
    "pinch_off": 0.35,       # 이보다 커져야 쓰기 멈춤
    "pinch_frames": 2,       # 이 프레임 수만큼 연속으로 조건을 만족해야 상태 전환

    # 손가락 폄: 굽힘 비율 = 손목-손가락끝 거리 / 손목-손가락MCP 거리
    "curl_extended": 1.55,   # 이보다 크면 편 손가락
    # 손가락 접힘: 손가락끝-손바닥중심 거리 / 손바닥 크기
    #   주먹을 쥐면 손가락 끝이 손바닥 안쪽으로 들어와 이 값이 작아진다.
    #   (편 손가락 약 1.6 이상, 핀치할 때 검지 약 0.9~1.2, 주먹 약 0.3~0.7)
    "fold_tip": 0.8,         # 이보다 작으면 접힌 손가락
    "fist_min_folded": 3,    # 검지 포함 네 손가락 중 이 개수 이상 접히면 주먹
    # 검지 뻗음: 검지끝이 검지MCP보다 손가락 방향(손목→중지MCP)으로 얼마나 나가 있는지 / 손바닥 크기
    #   주먹은 검지 끝이 손바닥 안으로 말려 MCP보다 손목 쪽에 있고 (0 이하),
    #   핀치(나머지 손가락 접음)는 검지가 C자라 끝이 MCP보다 바깥에 있다 (양수).
    "fist_index_reach": 0.15,# 이보다 작아야 주먹 (핀치와 구분)
    "fist_frames": 2,        # 주먹이 이 프레임 이상 유지돼야 지우개로 인정

    # 박수: 양 손바닥 중심 거리 / 평균 손바닥 크기
    "clap_close": 1.6,       # 이보다 가까우면 손이 맞닿은 것으로 봄
    "clap_open": 2.0,        # 이보다 멀어져야 다음 박수를 셀 수 있음
    "clap_near": 3.0,        # 이 거리 안에서 손이 사라지면 '겹쳐서 안 보이는 것'으로 간주
    "clap_grace": 0.4,       # 가까이 있던 손이 사라진 뒤 이 시간(초)까지는 맞닿은 것으로 처리
    "clap_window": 1.3,      # 두 번의 박수가 이 시간(초) 안에 있어야 DOUBLE_CLAP
    "clap_min_gap": 0.12,    # 너무 짧은 간격(같은 박수가 두 번 세지는 것) 무시
    "clap_cooldown": 1.0,    # DOUBLE_CLAP 직후 이 시간 동안 박수 무시

    # 스와이프: 손을 편 상태에서 손목이 빠르게 좌우로 이동
    "swipe_window": 0.5,     # 이 시간(초) 동안의 이동량을 봄
    "swipe_dist": 0.18,      # 이동량이 화면 너비의 이 비율 이상이면 스와이프
    "swipe_max_slope": 0.7,  # 세로 이동 / 가로 이동이 이보다 크면 스와이프 아님 (대각선 허용)
    "swipe_open_fingers": 3, # 네 손가락 중 이 개수 이상 펴져 있으면 '손바닥 편 상태'
    "swipe_open_curl": 1.4,  # 손가락 폄 판정 기준 (스와이프용, 일반 기준보다 느슨)
    "swipe_open_ratio": 0.6, # 시간 창 안 프레임 중 이 비율 이상이 '손바닥 편 상태'면 인정 (흔들림 허용)
    "swipe_lost_grace": 0.25,# 빠르게 움직여 손을 잠깐 놓쳐도 이 시간(초)까지는 기록 유지
    "swipe_cooldown": 0.8,
    "mode_flash": 0.8,       # NEXT PAGE / PREV PAGE 표시 유지 시간(초)
}


# ---------------------------------------------------------------------------
# 기본 계산 함수
# ---------------------------------------------------------------------------
def dist(a, b):
    return float(np.linalg.norm(a - b))


def palm_size(lm):
    """손바닥 크기 = 손목(0) ~ 중지 MCP(9) 거리. 카메라와의 거리 보정에 사용."""
    return dist(lm[WRIST, :2], lm[MIDDLE_MCP, :2]) + 1e-6


def palm_center(lm):
    return lm[[0, 5, 9, 13, 17], :2].mean(axis=0)


def pinch_ratio(lm):
    """엄지끝-검지끝 3D 거리를 3D 손바닥 크기로 정규화. z를 포함해 손이 옆으로 돌아간 경우의 오판을 줄인다.
    분모도 3D여야 한다: 2D 손바닥 크기는 손을 눕히면 짧아져 비율이 커지고 핀치가 잘 안 잡힌다."""
    return dist(lm[THUMB_TIP], lm[INDEX_TIP]) / (dist(lm[WRIST], lm[MIDDLE_MCP]) + 1e-6)


def finger_curls(lm):
    """각 손가락의 굽힘 비율. 클수록 펴짐. 손이 기울어져도 동작하도록 y좌표 비교 대신 거리 비율을 쓴다."""
    w = lm[WRIST, :2]
    return {name: dist(w, lm[tip, :2]) / (dist(w, lm[mcp, :2]) + 1e-6)
            for name, (mcp, tip) in FINGERS.items()}


def tip_to_palm(lm):
    """각 손가락 끝이 손바닥 중심에서 얼마나 떨어져 있는지 (손바닥 크기로 정규화). 작을수록 접힘."""
    c = palm_center(lm)
    s = palm_size(lm)
    return {name: dist(lm[tip, :2], c) / s for name, (mcp, tip) in FINGERS.items()}


def index_reach(lm):
    """검지끝이 검지MCP보다 손가락 방향(손목→중지MCP)으로 나가 있는 정도. 음수면 손바닥 안으로 말림.
    손이 기울어져도 동작하도록 화면 y축 대신 손바닥 축에 투영한다."""
    axis = lm[MIDDLE_MCP, :2] - lm[WRIST, :2]
    s = palm_size(lm)
    return float(np.dot(lm[INDEX_TIP, :2] - lm[INDEX_MCP, :2], axis / s)) / s


def classify_pose(lm, curls, tips=None):
    """한 손의 정적인 모양 분류 (핀치 여부는 PinchDetector가 따로 판단)."""
    tips = tip_to_palm(lm) if tips is None else tips
    folded = {k: v < CFG["fold_tip"] for k, v in tips.items()}
    extended = {k: v > CFG["curl_extended"] for k, v in curls.items()}

    # 1) 주먹: 검지가 접혀 있고, 네 손가락 중 fist_min_folded개 이상 접힘.
    #    MediaPipe는 주먹일 때 손가락 하나쯤은 덜 접힌 것으로 잡는 경우가 많아 4개 모두를 요구하지 않는다.
    #    주먹은 엄지·검지 끝도 가까워서 핀치보다 먼저 검사해야 한다.
    #    나머지 손가락을 접고 핀치하면 검지 끝도 손바닥 중심 가까이 오므로 접힘 조건만으로는 구분이 안 된다.
    #    그래서 검지 끝이 MCP보다 손바닥 안쪽으로 말려 들어갔는지(index_reach)를 추가로 본다.
    if folded["index"] and sum(folded.values()) >= CFG["fist_min_folded"] \
            and index_reach(lm) < CFG["fist_index_reach"]:
        return "FIST"
    # 2) 포인터: 검지만 펴고 나머지 접힘 + 엄지와 검지가 떨어져 있음
    if extended["index"] and folded["middle"] and folded["ring"] and folded["pinky"] \
            and pinch_ratio(lm) > CFG["pinch_off"]:
        return "POINTER"
    # 3) 손바닥 펴기: 네 손가락 모두 폄
    if all(extended.values()):
        return "OPEN"
    return "NONE"


# ---------------------------------------------------------------------------
# 상태 판정기들
# ---------------------------------------------------------------------------
class PinchDetector:
    """히스테리시스 + 프레임 디바운스로 Pen Down / Pen Up 판정."""

    def __init__(self):
        self.down = False
        self.count = 0

    def update(self, ratio, blocked=False):
        if ratio is None or blocked:          # 손이 사라지거나 주먹이면 즉시 펜 들기
            self.down, self.count = False, 0
            return self.down
        want = (ratio < CFG["pinch_on"]) if not self.down else not (ratio > CFG["pinch_off"])
        if want != self.down:
            self.count += 1
            if self.count >= CFG["pinch_frames"]:
                self.down, self.count = want, 0
        else:
            self.count = 0
        return self.down


class ClapDetector:
    """양손 손바닥 중심 거리가 가까워졌다 벌어지는 것을 박수 1회로 세고, 짧은 시간 안에 2회면 DOUBLE_CLAP.

    실제 박수에서는 두 손이 맞닿는 순간 MediaPipe가 한 손 또는 두 손 모두를 놓치는 경우가 많다.
    그래서 '직전에 두 손이 가까이 있었는데 갑자기 손이 사라짐'도 맞닿음으로 처리한다.
    """

    def __init__(self):
        self.touching = False
        self.last_d = None          # 마지막으로 두 손이 보였을 때의 거리
        self.last_near_t = -1e9     # 두 손이 clap_near 안에 있었던 마지막 시각
        self.claps = deque()
        self.cooldown_until = 0.0

    def _register_touch(self, t):
        if not self.touching:
            self.touching = True
            if not self.claps or t - self.claps[-1] > CFG["clap_min_gap"]:
                self.claps.append(t)

    def update(self, hands, t):
        if t < self.cooldown_until:
            self.touching = False
            return False

        if len(hands) >= 2:
            a, b = hands[0], hands[1]
            scale = (palm_size(a) + palm_size(b)) / 2
            d = dist(palm_center(a), palm_center(b)) / scale
            self.last_d = d
            if d < CFG["clap_near"]:
                self.last_near_t = t
            if d < CFG["clap_close"]:
                self._register_touch(t)
            elif d > CFG["clap_open"]:
                self.touching = False
        else:
            # 손이 1개 이하로 보임: 방금 전까지 두 손이 가까이 있었다면 겹쳐서 안 보이는 것
            if t - self.last_near_t < CFG["clap_grace"]:
                self._register_touch(t)
            else:
                self.touching = False
                self.last_d = None

        while self.claps and t - self.claps[0] > CFG["clap_window"]:
            self.claps.popleft()
        if len(self.claps) >= 2:
            self.claps.clear()
            self.touching = False
            self.cooldown_until = t + CFG["clap_cooldown"]
            return True
        return False


class SwipeDetector:
    """손바닥을 편 상태에서 손이 빠르게 좌우로 이동하면 스와이프.

    인식률을 높이기 위한 처리
      - 손목 대신 손바닥 중심을 추적 (더 안정적)
      - 빠르게 움직이면 손가락이 흐려져 '편 손'이 몇 프레임 깨지므로, 시간 창 안에서 60% 이상만 펴져 있으면 인정
      - 빠른 움직임 때문에 손을 잠깐 놓쳐도 0.25초까지는 기록을 지우지 않음
      - 세로 이동은 절대값이 아니라 가로 이동 대비 비율로 제한 (약간 대각선이어도 인정)
    """

    def __init__(self):
        self.hist = deque()          # (t, x, y, is_open)
        self.last_seen = -1e9
        self.cooldown_until = 0.0

    def update(self, point, is_open, frame_w, frame_h, t):
        if point is None:
            if t - self.last_seen > CFG["swipe_lost_grace"]:
                self.hist.clear()
            return None
        self.last_seen = t
        self.hist.append((t, point[0] / frame_w, point[1] / frame_h, bool(is_open)))
        while self.hist and t - self.hist[0][0] > CFG["swipe_window"]:
            self.hist.popleft()
        if t < self.cooldown_until or len(self.hist) < 3:
            return None

        open_ratio = sum(h[3] for h in self.hist) / len(self.hist)
        if open_ratio < CFG["swipe_open_ratio"]:
            return None
        dx = self.hist[-1][1] - self.hist[0][1]
        dy = abs(self.hist[-1][2] - self.hist[0][2])
        if abs(dx) > CFG["swipe_dist"] and dy < CFG["swipe_max_slope"] * abs(dx):
            self.hist.clear()
            self.cooldown_until = t + CFG["swipe_cooldown"]
            return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
        return None


# ---------------------------------------------------------------------------
# 결과 구조체와 메인 인식기
# ---------------------------------------------------------------------------
@dataclass
class GestureResult:
    hands: list = field(default_factory=list)   # 각 손의 (21,3) 픽셀 좌표 배열
    main_hand: np.ndarray = None                 # 필기에 쓰는 손 (가장 크게 보이는 손)
    pose: str = "NONE"                           # PINCH / FIST / POINTER / OPEN / NONE
    pen_down: bool = False                       # True = 쓰는 중 (화면에는 WRITE ON으로 표시)
    pen_point: tuple = None                      # 필기 좌표 (엄지끝과 검지끝의 중간점)
    pointer_point: tuple = None                  # 검지 끝 좌표 (포인터·지우개 위치)
    events: list = field(default_factory=list)   # DOUBLE_CLAP, SWIPE_LEFT, SWIPE_RIGHT
    debug: dict = field(default_factory=dict)    # 튜닝용 수치 (핀치 비율, 굽힘 비율 등)


class GestureRecognizer:
    """랜드마크 리스트를 받아 프레임마다 GestureResult를 낸다."""

    def __init__(self):
        self.pinch = PinchDetector()
        self.clap = ClapDetector()
        self.swipe = SwipeDetector()
        self.fist_count = 0

    def update(self, hands, frame_w, frame_h, t):
        """hands: HandTracker.detect() 결과 (손마다 (21,3) 픽셀 좌표). t: 초 단위 시각."""
        r = GestureResult(hands=hands)

        # 박수는 양손 정보로 먼저 판단
        if self.clap.update(hands, t):
            r.events.append("DOUBLE_CLAP")

        if not hands:
            self.pinch.update(None)
            self.fist_count = 0
            self.swipe.update(None, False, frame_w, frame_h, t)
            r.debug = {"hand_count": 0, "clap_touch": self.clap.touching, "clap_count": len(self.clap.claps)}
            return r

        # 필기 손 = 가장 크게 보이는 손 (카메라에 가장 가까운 발표자 손)
        lm = max(hands, key=palm_size)
        r.main_hand = lm
        curls = finger_curls(lm)
        tips = tip_to_palm(lm)
        pose = classify_pose(lm, curls, tips)
        ratio = pinch_ratio(lm)

        # 주먹은 몇 프레임 유지돼야 인정
        self.fist_count = self.fist_count + 1 if pose == "FIST" else 0
        is_fist = self.fist_count >= CFG["fist_frames"]
        if pose == "FIST" and not is_fist:
            pose = "NONE"

        # 양손이 보이면(박수 중일 수 있음) 필기를 막는다
        r.pen_down = self.pinch.update(ratio, blocked=(pose == "FIST" or len(hands) >= 2))
        if r.pen_down:
            pose = "PINCH"
        r.pose = pose

        r.pen_point = tuple(((lm[THUMB_TIP, :2] + lm[INDEX_TIP, :2]) / 2).astype(int))
        r.pointer_point = tuple(lm[INDEX_TIP, :2].astype(int)) if pose != "FIST" \
            else tuple(palm_center(lm).astype(int))

        # 양손이 보이면(박수 중일 수 있음) 스와이프도 막는다. 박수 때 손 위치가 튀면서 스와이프로 오인되는 것 방지
        n_open = sum(v > CFG["swipe_open_curl"] for v in curls.values())
        swipe_ok = (len(hands) == 1 and pose not in ("PINCH", "FIST", "POINTER")
                    and n_open >= CFG["swipe_open_fingers"])
        sw = self.swipe.update(palm_center(lm) if len(hands) == 1 else None,
                               swipe_ok, frame_w, frame_h, t)
        if sw:
            r.events.append(sw)

        r.debug = {"hand_count": len(hands),
                   "pinch_ratio": round(ratio, 2),
                   "index_reach": round(index_reach(lm), 2),
                   "fold(tip-palm)": " ".join(f"{v:.2f}" for v in tips.values()),
                   "curl": " ".join(f"{v:.2f}" for v in curls.values()),
                   "clap_d": None if self.clap.last_d is None else round(self.clap.last_d, 2),
                   "clap_touch": self.clap.touching,
                   "clap_count": len(self.clap.claps)}
        return r
