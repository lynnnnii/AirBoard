"""Read gesture mappings and UI settings exclusively from JSON."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ('draw','stop','toggle_highlighter','eraser','pointer','undo','redo','none')

# Restrict each gesture category to meaningful actions.
#  - One-shot gestures (single-frame events) cannot meaningfully activate continuous actions such as drawing, erasing, or pointing.
#  - Held gestures exclude page navigation because it can be confused with writing or erasing.
ONE_SHOT_GESTURES = ('DOUBLE_CLAP','SWIPE_LEFT','SWIPE_RIGHT','SWIPE_UP','SWIPE_DOWN','THUMB_LEFT','THUMB_RIGHT')
ONE_SHOT_ACTIONS = ('toggle_highlighter','undo','redo','none')
HELD_ACTIONS = ('draw','stop','toggle_highlighter','eraser','pointer','none')
DISABLED_GESTURES = ('SWIPE_LEFT','SWIPE_RIGHT','SWIPE_UP','SWIPE_DOWN')


def allowed_actions(gesture: str) -> tuple[str,...]:
    """List the actions permitted for this gesture in the settings menu."""
    if gesture in DISABLED_GESTURES:
        return ('none',)
    return ONE_SHOT_ACTIONS if gesture in ONE_SHOT_GESTURES else HELD_ACTIONS


def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON object."""
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data,dict):
        raise ValueError('Expected a JSON object.')
    return data


class Settings:
    """User mappings override the default preset."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = Path(root)
        self.ui = read_json(self.root/'config/default.json')['ui']
        self._ui_defaults = json.loads(json.dumps(self.ui))   # Keep an original copy to save only changed values
        self.mapping = read_json(self.root/'config/gestures.json')
        if not self.mapping:
            raise ValueError('The default gesture mapping is empty; restore config/gestures.json.')
        defaults = dict(self.mapping)
        # The AIRBOARD_USER_SETTINGS variable overrides the user settings file location.
        # (This lets tests run with team defaults instead of a developer's personal settings).
        override = os.environ.get('AIRBOARD_USER_SETTINGS') if self.root == ROOT else None
        self.user_path = Path(override) if override else self.root/'config/ui-user.json'
        if self.user_path.exists():
            user = read_json(self.user_path)
            # For gestures not configured by the user, fall back to defaults (including newly added gestures).
            self.mapping = {**self.mapping, **user['mapping']}
            self.ui.update(user['ui'])
            self._repair_mapping(defaults)
        self.validate()

    def _repair_mapping(self, defaults: dict) -> None:
        """Reset unsupported actions from older settings (such as page swipes) to none or defaults."""
        for g,a in list(self.mapping.items()):
            if a not in allowed_actions(g):
                fallback = defaults.get(g,'none')
                self.mapping[g] = fallback if fallback in allowed_actions(g) else 'none'

    def set_action(self, gesture: str, action: str) -> None:
        """Assign an action to a gesture; drawing may be assigned to multiple gestures."""
        if action not in allowed_actions(gesture):
            raise ValueError(f'{gesture} cannot be mapped to {action}.')
        self.mapping[gesture] = action

    def validate(self) -> None:
        """Accept only supported configuration values."""
        if not isinstance(self.mapping,dict) or not self.mapping or any(not isinstance(k,str) or not k or v not in allowed_actions(k) for k,v in self.mapping.items()):
            raise ValueError('Invalid gesture mapping')
        if not self.ui['colors'] or any(len(c)!=7 or c[0]!='#' for c in self.ui['colors']):
            raise ValueError('Color must use #RRGGBB format.')
        for c in self.ui['colors']:
            int(c[1:],16)
        if not self.ui['widths'] or any(type(w) is not int or not 1<=w<=40 for w in self.ui['widths']):
            raise ValueError('Line width must be an integer from 1 to 40.')
        for key in ('dwell_seconds','notification_seconds','navigation_quiet_seconds','navigation_cooldown_seconds'):
            if not 0<float(self.ui.get(key,.5))<=10:
                raise ValueError('UI timing must be between 0 and 10 seconds.')
        if type(self.ui['tool_hold_frames']) is not int or not 1<=self.ui['tool_hold_frames']<=60:
            raise ValueError('Tool hold frames must be between 1 and 60.')
        if not 0<float(self.ui['highlighter_opacity'])<=1:
            raise ValueError('Opacity must be between 0 and 1.')

    def save(self) -> None:
        """Save user settings using an atomic temporary-file replacement.

        Only save UI values that differ from config/default.json. Copying all values would
        cause outdated personal values to override future team default updates.
        """
        self.validate()
        ui = {k:v for k,v in self.ui.items() if self._ui_defaults.get(k) != v}
        temp = self.user_path.with_suffix('.tmp')
        temp.write_text(json.dumps({'mapping':self.mapping,'ui':ui},indent=2),encoding='utf-8')
        temp.replace(self.user_path)
