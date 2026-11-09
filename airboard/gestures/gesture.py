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

    # Finger extension: curl ratio = wrist-to-tip distance / wrist-to-MCP distance
    "curl_extended": 1.55,   # Extended when above this threshold
    # Folding ratio = fingertip-to-palm-center distance / palm size
    #   Fingertips move toward the palm in a fist, reducing this ratio.
    #   (Extended finger ~1.6+, pinching index ~0.9–1.2, fist ~0.3–0.7)
    "fold_tip": 0.8,         # Considered folded below this threshold
    "fist_min_folded": 5,    # Require all five fingers, including the thumb, to be folded
    "fist_thumb_fold": 0.9,  # Thumb is folded if its tip-to-palm distance / palm size is below this value
    # Index reach: index-tip projection past index MCP along wrist-to-middle-MCP axis, divided by palm size
    #   For a fist, the index tip curls toward the wrist behind the MCP (reach <= 0),
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

    # Swipe: the wrist moves rapidly sideways while the hand is open
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
    """Palm size = distance from wrist (0) to middle MCP (9), used for distance normalization."""
    return dist(lm[WRIST, :2], lm[MIDDLE_MCP, :2]) + 1e-6


def palm_center(lm):
    return lm[[0, 5, 9, 13, 17], :2].mean(axis=0)


def pinch_ratio(lm):
    """Normalize the 2D thumb-to-index tip distance by the 3D palm size.
    Use 2D numerator: fingertip z is noisy and may prevent pinch detection even with touching fingertips.
    The denominator is 3D because palm tilt shortens its 2D projection; wrist/MCP z values are relatively stable."""
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
    """Index-tip reach past the MCP along the wrist-to-middle-MCP axis; negative values mean inward curling.
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
    """Count one clap when palms approach and separate; two claps in a short window trigger DOUBLE_CLAP.

    When palms touch, MediaPipe often temporarily loses one or both hands.
    Therefore treat sudden tracking loss after the palms approach as contact.
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
            # Count contact only when hands become occluded while approaching.
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
    """Detect a swipe from fast lateral movement of an open hand.

    Measures to improve recognition reliability
      - Track the palm center rather than wrist (more stable)
      - Detect movement of one open hand; reset history on tool actions or tracking loss
      - Tolerate brief finger-pose jitter without recording those frame coordinates
      - Exclude cooldown movement from the next swipe history
      - Limit vertical movement as a fraction of horizontal movement to allow diagonal swipes
    """

    def __init__(self, config=None):
        self.config = {**CFG, **(config or {})}
        self.hist = deque()          # (t, x, y, is_open)
        self.last_seen = -1e9
        self.cooldown_until = 0.0
        self._last_swipe = None      # (direction, time) Used to ignore reverse movement when returning the hand

    def update(self, point, is_open, frame_w, frame_h, t):
        # Do not mix tool actions, two-hand frames, or tracking gaps into swipe history.
        if point is None or t < self.cooldown_until:
            self.hist.clear()
            return None
        if not is_open:
            # Allow brief finger-pose noise; the caller suppresses tool input by passing None.
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
            # To avoid interpreting hand return motion after a swipe as an opposite swipe,
            # ignore reverse movement for swipe_reverse_guard seconds (repeated same-direction swipes remain allowed).
            if (self._last_swipe is not None and event != self._last_swipe[0]
                    and t - self._last_swipe[1] < self.config.get('swipe_reverse_guard', 0.)):
                return None
            self.cooldown_until = t + self.config["swipe_cooldown"]
            self._last_swipe = (event, t)
            return event
        return None


# ---------------------------------------------------------------------------
# Result structure and main recognizer
# ---------------------------------------------------------------------------
@dataclass
class GestureResult:
    hands: list = field(default_factory=list)   # (21, 3) pixel coordinate array for each hand
    main_hand: np.ndarray = None                 # Drawing hand (the largest visible hand)
    pose: str = "NONE"                           # PINCH / FIST / POINTER / TWO_FINGERS / OPEN / NONE
    pen_down: bool = False                       # True = Writing (displayed as WRITE ON)
    pen_point: tuple = None                      # Drawing point (midpoint between thumb and index tips)
    pointer_point: tuple = None                  # Index fingertip coordinates (pointer/eraser position)
    events: list = field(default_factory=list)   # DOUBLE_CLAP, SWIPE_LEFT, SWIPE_RIGHT
    debug: dict = field(default_factory=dict)    # Tuning values (pinch ratio, curl ratio, etc.)


class GestureRecognizer:
    """Produce a GestureResult per frame from the landmark arrays."""

    def __init__(self, config=None):
        self.pinch = PinchDetector(config)
        self.config = {**CFG, **(config or {})}
        self.clap = ClapDetector(config)
        self._swipe_blocked_until = 0.
        self._ink_blocked_until = 0.
        self.swipe = SwipeDetector(config)
        self.fist_count = 0
        self._last_ink_point = None
        self._prev_pose = 'NONE'     # Previous frame pose (used for stable index-gesture recognition)
        self._prev_history_pose = None
        self._history_cooldown_until = float('-inf')

    def update(self, hands, frame_w, frame_h, t):
        """hands: output of HandTracker.detect() ((21, 3) pixels per hand); t: time in seconds."""
        r = GestureResult(hands=hands)

        # Detect claps first using both hands
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

        # Writing hand = largest visible hand (closest to camera)
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
        # Enable the two checks below only when configured by UI gesture mapping; preserve normal pinch behavior otherwise.
        #  - index_writing: Index drawing mode (held index pose + two-finger recognition)
        #  - two_fingers: Two-finger gesture mapped to another action (recognize two fingers only)
        # Neither check interrupts the start of a pinch (ratio below pinch_on) or an active pinch,
        # nor blocks the pinch detector. Pinching while using index drawing switches immediately to pinch.
        index_mode = self.config.get('index_writing', False)
        pinch_free = not self.pinch.down and ratio > self.config['pinch_on']
        # Two-finger extension (index and middle): stop writing in index drawing mode; ring and little fingers must be folded.
        # Finger separation distance is not checked.
        two = self.config.get('two_finger_curl', CFG['curl_extended'])
        if ((index_mode or self.config.get('two_fingers', False)) and pinch_free and pose == 'NONE'
                and curls['index'] > two and curls['middle'] > two
                and all(tips[k] < self.config.get('two_finger_fold', CFG['fold_tip']) for k in ('ring', 'pinky'))):
            pose = 'TWO_FINGERS'
        # Relax the recognition threshold while the index pose is held (hysteresis). During writing, the index bends slightly
        # and thumb approaches it; strict initial thresholds (extension 1.55, thumb gap 0.38) would cause interruptions.
        kept_pointer = (index_mode and pinch_free and pose == 'NONE' and self._prev_pose == 'POINTER'
                        and curls['index'] > self.config.get('pointer_keep_curl', CFG['curl_extended'])
                        and all(tips[k] < self.config.get('pointer_keep_fold', CFG['fold_tip'])
                                for k in ('middle', 'ring', 'pinky'))
                        and ratio > self.config.get('pointer_keep_pinch', CFG['pinch_off']))
        if kept_pointer:
            pose = 'POINTER'

        # Maintain an active pinch while writing; allow eraser switching only after pen-up.
        if (pose=='FIST' and self.pinch.down
                and ratio<=self.pinch.config['pinch_off']):
            pose='NONE'
        # Distinguish a fist from a pinch with other fingers folded.
        if pose=='FIST' and tips['index']>self.config.get('fist_index_fold',.65):
            pose='NONE'
        fist_candidate = pose=="FIST"
        self.fist_count = self.fist_count + 1 if fist_candidate else 0
        is_fist = self.fist_count >= self.config.get('fist_frames',CFG['fist_frames'])
        if pose == "FIST" and not is_fist:
            pose = "NONE"

        # Suppress drawing when both hands appear (possibly clapping)
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
        # During release confirmation or when fingers begin opening (ratio > pinch_hold), do not add hand movement to ink.
        # Prevent unwanted tails while releasing the pinch before the pen-up threshold (pinch_off).
        # Apply pinch_hold only when drawing with pinch (pinch_writing, configurable by UI). When pinch is mapped to eraser, 
        # slight finger opening must not freeze eraser motion. Keep the existing release-confirmation hold behavior.
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
