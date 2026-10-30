"""정규화 카메라 좌표를 슬라이드 픽셀로 변환한다."""
import json
from pathlib import Path
import cv2
import numpy as np


class Calibration:
    """TL, TR, BR, BL 순서의 볼록 사각형. 입력은 좌우 반전 후 0~1 좌표."""

    def __init__(self, corners: list, size: tuple[int, int] = (960, 540)) -> None:
        p = np.asarray(corners, dtype=np.float32)
        if p.shape != (4, 2) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('네 점은 유한한 0~1 정규화 좌표여야 합니다.')
        if len(size) != 2 or any(type(v) is not int or v < 2 for v in size):
            raise ValueError('슬라이드 크기는 2 이상의 정수 두 개입니다.')
        edges = np.roll(p, -1, axis=0) - p
        following = np.roll(edges, -1, axis=0)
        cross = edges[:, 0]*following[:, 1]-edges[:, 1]*following[:, 0]
        if not (cross > 1e-6).all():
            raise ValueError('좌상→우상→우하→좌하 순서의 볼록 사각형을 지정하세요.')
        w, h = size
        self.corners, self.size = p.copy(), tuple(size)
        self.matrix = cv2.getPerspectiveTransform(p, np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]))

    def contains(self, position: tuple) -> bool:
        """사각형 밖 또는 손 소실 좌표는 사용하지 않는다."""
        if position is None:
            return False
        p = np.asarray(position, dtype=float)
        return p.shape == (2,) and np.isfinite(p).all() and cv2.pointPolygonTest(self.corners, tuple(map(float,p)), False) >= 0

    def transform(self, position: tuple) -> tuple[float, float] | None:
        """영역 밖은 None; 영역 안은 슬라이드 픽셀로 반환한다."""
        if not self.contains(position):
            return None
        result = cv2.perspectiveTransform(np.float32([[position]]), self.matrix)[0,0]
        return tuple(map(float, result))

    def save(self, path: Path) -> None:
        """사용자 캘리브레이션을 저장한다."""
        Path(path).write_text(json.dumps({'version':1,'corners':self.corners.tolist(),'size':self.size},indent=2),encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'Calibration':
        """파일을 검증하여 복원한다."""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != 1:
            raise ValueError('지원하지 않는 캘리브레이션 버전')
        return cls(data['corners'],tuple(data['size']))
