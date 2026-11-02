"""제스처 매핑과 UI 설정을 JSON에서만 읽는다."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ('draw','stop','toggle_highlighter','eraser','pointer','page_next','page_prev','none')


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
        # 저장소 기본 위치에서는 AIRBOARD_USER_SETTINGS로 개인 설정 파일을 바꿀 수 있다
        # (테스트 파일들이 개발자의 개인 설정 없이 팀 기본값으로 돌도록 사용).
        override = os.environ.get('AIRBOARD_USER_SETTINGS') if self.root == ROOT else None
        self.user_path = Path(override) if override else self.root/'config/ui-user.json'
        if self.user_path.exists():
            user = read_json(self.user_path)
            # 개인 매핑에 없는 제스처(나중에 추가된 기본 제스처)는 기본값을 쓴다.
            self.mapping = {**self.mapping, **user['mapping']}
            self.ui.update(user['ui'])
        self.validate()

    def validate(self) -> None:
        """실행 가능한 값만 허용한다."""
        if not isinstance(self.mapping,dict) or not self.mapping or any(not isinstance(k,str) or not k or v not in ACTIONS for k,v in self.mapping.items()):
            raise ValueError('잘못된 제스처 매핑')
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
        ui = {k:v for k,v in self.ui.items() if self._ui_defaults.get(k) != v}
        temp = self.user_path.with_suffix('.tmp')
        temp.write_text(json.dumps({'mapping':self.mapping,'ui':ui},indent=2),encoding='utf-8')
        temp.replace(self.user_path)
