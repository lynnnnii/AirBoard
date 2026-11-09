"""Convert normalized camera coordinates into slide pixels."""
import json
from pathlib import Path
import cv2
import numpy as np


class Calibration:
    """A convex quadrilateral ordered TL, TR, BR, BL, using mirrored coordinates in [0, 1]."""

    def __init__(self, corners: list, size: tuple[int, int] = (960, 540)) -> None:
        p = np.asarray(corners, dtype=np.float32)
        if p.shape != (4, 2) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('The four points must be finite normalized coordinates in [0, 1].')
        if len(size) != 2 or any(type(v) is not int or v < 2 for v in size):
            raise ValueError('Slide dimensions must be two integers of at least 2.')
        edges = np.roll(p, -1, axis=0) - p
        following = np.roll(edges, -1, axis=0)
        cross = edges[:, 0]*following[:, 1]-edges[:, 1]*following[:, 0]
        if not (cross > 1e-6).all():
            raise ValueError('Provide a convex quadrilateral ordered top-left, top-right, bottom-right, bottom-left.')
        w, h = size
        self.corners, self.size = p.copy(), tuple(size)
        self.matrix = cv2.getPerspectiveTransform(p, np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]]))

    def contains(self, position: tuple) -> bool:
        """Discard points outside the quadrilateral or when hand tracking is lost."""
        if position is None:
            return False
        p = np.asarray(position, dtype=float)
        return p.shape == (2,) and np.isfinite(p).all() and cv2.pointPolygonTest(self.corners, tuple(map(float,p)), False) >= 0

    def transform(self, position: tuple) -> tuple[float, float] | None:
        """Return None for out-of-bounds points; otherwise return the slide pixel coordinates."""
        if not self.contains(position):
            return None
        result = cv2.perspectiveTransform(np.float32([[position]]), self.matrix)[0,0]
        return tuple(map(float, result))

    def save(self, path: Path) -> None:
        """Save user calibration settings."""
        Path(path).write_text(json.dumps({'version':1,'corners':self.corners.tolist(),'size':self.size},indent=2),encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'Calibration':
        """Validate and restore calibration from a file."""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        if data.get('version') != 1:
            raise ValueError('Unsupported calibration version')
        return cls(data['corners'],tuple(data['size']))
