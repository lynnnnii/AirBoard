"""제스처 매핑과 UI 설정을 JSON에서만 읽는다."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ('draw','stop','toggle_highlighter','eraser','pointer','page_next','page_prev','none')

# 제스처 종류별로 의미 있는 동작만 고를 수 있다.
#  - 순간형(한 프레임짜리 이벤트): 쓰기·지우기·포인터처럼 '하는 동안' 동작은 한 프레임만 켜져 의미가 없다.
#  - 유지형(그 모양을 하는 동안 계속): 페이지 넘기기는 쓰기·지우기와 헷갈려 실수로 넘어가기 쉬워 뺀다.
ONE_SHOT_GESTURES = ('DOUBLE_CLAP','SWIPE_LEFT','SWIPE_RIGHT','SWIPE_UP','SWIPE_DOWN')
ONE_SHOT_ACTIONS = ('toggle_highlighter','page_next','page_prev','none')
HELD_ACTIONS = ('draw','stop','toggle_highlighter','eraser','pointer','none')


def allowed_actions(gesture: str) -> tuple[str,...]:
    """이 제스처에 연결할 수 있는 동작 목록 (설정 화면에서 이 안에서만 고른다)."""
    return ONE_SHOT_ACTIONS if gesture in ONE_SHOT_GESTURES else HELD_ACTIONS


def read_json(path: Path) -> dict:
    """UTF-8 JSON 객체를 읽는다."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,dict):
        raise ValueError('JSON 객체가 필요합니다.')
    return data


class Settings:
    """사용자 매핑은 기본 프리셋을 덮어쓴다."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = Path(root)
        self.ui = read_json(self.root/'config/default.json')['ui']
        self._ui_defaults = json.loads(json.dumps(self.ui))   # 저장할 때 바뀐 값만 고르기 위한 원본 복사
        self.mapping = read_json(self.root/'config/gestures.json')
        if not self.mapping:
            raise ValueError('기본 제스처 매핑이 비어 있습니다: config/gestures.json을 복원하세요.')
        defaults = dict(self.mapping)
        # 저장소 기본 위치에서는 AIRBOARD_USER_SETTINGS로 개인 설정 파일을 바꿀 수 있다
        # (테스트 파일들이 개발자의 개인 설정 없이 팀 기본값으로 돌도록 사용).
        override = os.environ.get('AIRBOARD_USER_SETTINGS') if self.root == ROOT else None
        self.user_path = Path(override) if override else self.root/'config/ui-user.json'
        if self.user_path.exists():
            user = read_json(self.user_path)
            mapping = dict(user['mapping'])
            for old, new in (('SWIPE_LEFT', 'SWIPE_UP'), ('SWIPE_RIGHT', 'SWIPE_DOWN')):
                if old in mapping:
                    mapping.setdefault(new, mapping.pop(old))
            # 개인 매핑에 없는 제스처(나중에 추가된 기본 제스처)는 기본값을 쓴다.
            self.mapping = {**self.mapping, **mapping}
            overrides = dict(user['ui'])
            if user.get('version', 1) < 2:
                # 옛 설정 화면이 통째로 저장한 팔레트/꼬리 제거 기본값을 이관한다.
                for key in ('colors', 'widths', 'highlighter_color'):
                    overrides.pop(key, None)
                smoothing = dict(overrides.get('handwriting_smoothing', {}))
                if smoothing.get('end_trim') == 0:
                    smoothing.pop('end_trim')
                overrides['handwriting_smoothing'] = smoothing
            smoothing = {**self.ui.get('handwriting_smoothing', {}),
                         **overrides.pop('handwriting_smoothing', {})}
            self.ui.update(overrides)
            self.ui['handwriting_smoothing'] = smoothing
            self._repair_mapping(defaults)
        self.validate()

    def _repair_mapping(self, defaults: dict) -> None:
        """예전에 저장한 개인 매핑에 지금은 고를 수 없는 조합이 있으면 기본값(안 되면 none)으로 바꾼다.

        draw가 여러 제스처에 있으면 첫 번째만 남긴다. 그래도 프로그램이 멈추지 않고 열리게 하기 위함이다.
        """
        for g,a in list(self.mapping.items()):
            if a in ACTIONS and a not in allowed_actions(g):
                fallback = defaults.get(g,'none')
                self.mapping[g] = fallback if fallback in allowed_actions(g) else 'none'
        drawing = [g for g,a in self.mapping.items() if a=='draw']
        for g in drawing[1:]:
            self.mapping[g] = 'none'

    def set_action(self, gesture: str, action: str) -> None:
        """제스처에 동작을 연결한다. draw는 한 제스처에만 둘 수 있어 다른 draw는 none이 된다."""
        if action not in allowed_actions(gesture):
            raise ValueError(f'{gesture}에는 {action}을 연결할 수 없습니다.')
        if action == 'draw':
            for g,a in self.mapping.items():
                if a == 'draw' and g != gesture:
                    self.mapping[g] = 'none'
        self.mapping[gesture] = action

    def validate(self) -> None:
        """실행 가능한 값만 허용한다."""
        if not isinstance(self.mapping,dict) or not self.mapping or any(not isinstance(k,str) or not k or v not in allowed_actions(k) for k,v in self.mapping.items()):
            raise ValueError('잘못된 제스처 매핑')
        if list(self.mapping.values()).count('draw') > 1:
            raise ValueError('draw는 한 제스처에만 연결할 수 있습니다.')
        if not self.ui['colors'] or any(len(c)!=7 or c[0]!='#' for c in self.ui['colors']):
            raise ValueError('색상은 #RRGGBB 형식입니다.')
        for c in self.ui['colors']:
            int(c[1:],16)
        if not self.ui['widths'] or any(type(w) is not int or not 1<=w<=40 for w in self.ui['widths']):
            raise ValueError('굵기는 1~40 정수입니다.')
        for key in ('dwell_seconds','notification_seconds','navigation_quiet_seconds','navigation_cooldown_seconds'):
            if not 0<float(self.ui.get(key,.5))<=10:
                raise ValueError('UI 시간은 0~10초 범위입니다.')
        if type(self.ui['tool_hold_frames']) is not int or not 1<=self.ui['tool_hold_frames']<=60:
            raise ValueError('도구 유지 프레임은 1~60입니다.')
        if not 0<float(self.ui['highlighter_opacity'])<=1:
            raise ValueError('불투명도는 0~1 범위입니다.')

    def save(self) -> None:
        """임시 파일 교체로 개인 설정을 저장한다.

        UI 값은 config/default.json과 다른 것만 저장한다. 전부 복사해 두면 나중에 팀 기본값을
        고쳐도 개인 파일의 옛 값이 덮어써서 반영되지 않는다.
        """
        self.validate()
        if 'draw' not in self.mapping.values():
            raise ValueError('쓰기(draw)를 연결한 제스처가 없습니다. 하나는 draw로 정해 주세요.')
        ui = {k:v for k,v in self.ui.items() if self._ui_defaults.get(k) != v}
        temp = self.user_path.with_suffix('.tmp')
        temp.write_text(json.dumps({'version':2,'mapping':self.mapping,'ui':ui},indent=2),encoding='utf-8')
        temp.replace(self.user_path)
