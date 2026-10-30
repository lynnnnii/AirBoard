"""제스처 매핑과 UI 설정을 JSON에서만 읽는다."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ('draw','toggle_highlighter','eraser','pointer','page_next','page_prev','none')


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
        self.mapping = read_json(self.root/'config/gestures.json')
        self.user_path = self.root/'config/ui-user.json'
        if self.user_path.exists():
            user = read_json(self.user_path)
            self.mapping = user['mapping']
            self.ui.update(user['ui'])
        self.validate()

    def validate(self) -> None:
        """실행 가능한 값만 허용한다."""
        if not isinstance(self.mapping,dict) or any(not isinstance(k,str) or not k or v not in ACTIONS for k,v in self.mapping.items()):
            raise ValueError('잘못된 제스처 매핑')
        if not self.ui['colors'] or any(len(c)!=7 or c[0]!='#' for c in self.ui['colors']):
            raise ValueError('색상은 #RRGGBB 형식입니다.')
        for c in self.ui['colors']:
            int(c[1:],16)
        if not self.ui['widths'] or any(type(w) is not int or not 1<=w<=40 for w in self.ui['widths']):
            raise ValueError('굵기는 1~40 정수입니다.')
        for key in ('dwell_seconds','notification_seconds'):
            if not 0<float(self.ui[key])<=10:
                raise ValueError('UI 시간은 0~10초 범위입니다.')
        if type(self.ui['tool_hold_frames']) is not int or not 1<=self.ui['tool_hold_frames']<=60:
            raise ValueError('도구 유지 프레임은 1~60입니다.')
        if not 0<float(self.ui['highlighter_opacity'])<=1:
            raise ValueError('불투명도는 0~1 범위입니다.')

    def save(self) -> None:
        """임시 파일 교체로 개인 설정을 저장한다."""
        self.validate()
        temp = self.user_path.with_suffix('.tmp')
        temp.write_text(json.dumps({'mapping':self.mapping,'ui':self.ui},indent=2),encoding='utf-8')
        temp.replace(self.user_path)
