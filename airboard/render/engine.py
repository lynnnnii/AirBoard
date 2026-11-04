"""AirBoard의 PDF 표시, 필기 레이어, 페이지별 undo와 내보내기.

이 모듈은 창/웹캠을 열지 않는다. 입력은 보정된 슬라이드 픽셀 좌표,
출력은 UI를 포함하지 않는 BGR 이미지다. 같은 스레드에서 호출한다.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
import json
import hashlib
import math
import os
import tempfile

import cv2
import numpy as np
import pymupdf

Tool = Literal['pen', 'highlighter', 'eraser', 'pointer']
Point = tuple[float, float]


@dataclass
class Stroke:
    """안내서의 Stroke 형식. page는 0부터, color는 #RRGGBB, width는 지름(px)."""
    stroke_id: int
    page: int
    tool: Tool
    color: str
    width: float
    points: list[Point]


@dataclass(frozen=True)
class RenderConfig:
    """config/default.json의 render 항목에서 읽는 설정."""
    max_width: int
    max_height: int
    cache_pages: int
    pen_color: str
    pen_width: float
    highlighter_color: str
    highlighter_width: float
    highlighter_opacity: float
    eraser_width: float

    def __post_init__(self) -> None:
        """잘못된 크기/투명도 설정을 일찍 거부한다."""
        if min(self.max_width, self.max_height, self.cache_pages) < 1:
            raise ValueError('렌더링 크기와 cache_pages는 양수여야 합니다.')
        if not 0 < self.highlighter_opacity <= 1:
            raise ValueError('highlighter_opacity는 0 초과 1 이하여야 합니다.')
        for value in (self.pen_width, self.highlighter_width, self.eraser_width):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('도구 굵기는 유한한 양수여야 합니다.')
        color_bgr(self.pen_color)
        color_bgr(self.highlighter_color)

    @classmethod
    def from_json(cls, path: str | Path) -> RenderConfig:
        """기존 설정 파일의 다른 팀원 항목은 읽거나 수정하지 않는다."""
        return cls(**json.loads(Path(path).read_text(encoding='utf-8'))['render'])


def color_bgr(color: str) -> tuple[int, int, int]:
    """HEX 색상을 OpenCV BGR로 변환한다."""
    if len(color) != 7 or not color.startswith('#'):
        raise ValueError('color는 #RRGGBB 형식이어야 합니다.')
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


@dataclass
class _Operation:
    """작업 당시 투명도도 기록하여 나중 설정 변경에 영향받지 않게 한다."""
    stroke: Stroke | None = None
    opacity: float = 1.0
    clear: bool = False
    erased_ids: list[int] | None = None


@dataclass
class _Canvas:
    """원본과 premultiplied 색/alpha를 분리해 원본 삭제를 막는다."""
    background: np.ndarray
    color: np.ndarray
    alpha: np.ndarray
    composed: np.ndarray


@dataclass
class _Active:
    """한 번의 연속 필기 또는 지우개 작업. 마스크 중복은 누적 농도를 만들지 않는다."""
    stroke: Stroke
    opacity: float
    mask: np.ndarray
    bounds: tuple[int, int, int, int] | None = None


class AirBoardRenderer:
    """PDF와 필기 상태 소유자. 손 인식/제스처/UI 모듈을 참조하지 않는다."""

    def __init__(self, pdf_path: str | Path, config: RenderConfig) -> None:
        """PDF를 열고 첫 페이지를 캐시한다. 모든 페이지를 미리 렌더링하지 않는다."""
        self.pdf_path = Path(pdf_path).resolve()
        self.config = config
        self.doc = pymupdf.open(self.pdf_path)
        if self.doc.needs_pass or len(self.doc) == 0:
            self.doc.close()
            raise ValueError('암호가 없고 페이지가 있는 PDF가 필요합니다.')
        self.current_page = 0
        self._history: list[list[_Operation]] = [[] for _ in self.doc]
        self._redo: list[list[_Operation]] = [[] for _ in self.doc]
        self._cache: OrderedDict[int, _Canvas] = OrderedDict()
        self._active: _Active | None = None
        self._next_id = 0
        self._closed = False
        self._canvas(0)

    @property
    def page_count(self) -> int:
        """전체 페이지 수."""
        return len(self.doc)

    @property
    def slide_size(self) -> tuple[int, int]:
        """현재 슬라이드의 (너비, 높이). 좌표 변환 담당자에게 전달한다."""
        height, width = self._canvas(self.current_page).background.shape[:2]
        return width, height

    def _canvas(self, index: int) -> _Canvas:
        """페이지 이미지를 LRU 캐시에서 얻고, 없으면 한 번 렌더링한다."""
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        page = self.doc[index]
        scale = min(self.config.max_width / page.rect.width,
                    self.config.max_height / page.rect.height)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                              colorspace=pymupdf.csRGB, alpha=False)
        rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
        bg = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        canvas = _Canvas(bg, np.zeros(bg.shape, np.float32),
                         np.zeros((*bg.shape[:2], 1), np.float32), bg.copy())
        erased = {i for op in self._history[index] for i in (op.erased_ids or [])}
        for operation in self._history[index]:
            if operation.stroke is None or operation.stroke.stroke_id not in erased:
                self._replay(canvas, operation)
        self._cache[index] = canvas
        while len(self._cache) > self.config.cache_pages:
            self._cache.popitem(last=False)
        return canvas

    def _settings(self, tool: Tool) -> tuple[str, float, float]:
        """모드별 기본 색상, 지름, 투명도."""
        c = self.config
        if tool == 'pen':
            return c.pen_color, c.pen_width, 1.0
        if tool == 'highlighter':
            return c.highlighter_color, c.highlighter_width, c.highlighter_opacity
        return c.pen_color, c.eraser_width, 1.0

    @staticmethod
    def _paint_segment(active: _Active, point: Point) -> None:
        """이전 점과 현재 점 사이 마스크를 갱신한다. 한 점도 점으로 그린다."""
        stroke = active.stroke
        now = tuple(round(v) for v in point)
        previous = tuple(round(v) for v in stroke.points[-1]) if stroke.points else now
        thickness = max(1, round(stroke.width))
        cv2.line(active.mask, previous, now, 255, thickness, cv2.LINE_AA)
        cv2.circle(active.mask, now, max(1, thickness // 2), 255, -1, cv2.LINE_AA)
        if not stroke.points:
            cv2.circle(active.mask, previous, max(1, thickness // 2), 255, -1, cv2.LINE_AA)
        stroke.points.append(point)
        h, w = active.mask.shape
        pad = thickness + 2
        bounds = (max(0, min(previous[0], now[0]) - pad),
                  max(0, min(previous[1], now[1]) - pad),
                  min(w, max(previous[0], now[0]) + pad + 1),
                  min(h, max(previous[1], now[1]) + pad + 1))
        if active.bounds is None:
            active.bounds = bounds
        else:
            a, b, x, y = active.bounds
            active.bounds = min(a, bounds[0]), min(b, bounds[1]), max(x, bounds[2]), max(y, bounds[3])

    @staticmethod
    def _apply(canvas: _Canvas, active: _Active) -> None:
        """마스크의 범위만 합성한다. 지우개는 필기 alpha만 제거한다."""
        if active.bounds is None:
            return
        x0, y0, x1, y1 = active.bounds
        region = np.s_[y0:y1, x0:x1]
        a = active.mask[region].astype(np.float32)[..., None] / 255.0 * active.opacity
        inverse = 1.0 - a
        if active.stroke.tool == 'eraser':
            canvas.color[region] *= inverse
            canvas.alpha[region] *= inverse
        else:
            color = np.array(color_bgr(active.stroke.color), np.float32)
            canvas.color[region] = color * a + canvas.color[region] * inverse
            canvas.alpha[region] = a + canvas.alpha[region] * inverse
        output = canvas.color[region] + canvas.background[region] * (1.0 - canvas.alpha[region])
        canvas.composed[region] = np.rint(output).clip(0, 255).astype(np.uint8)

    def _replay(self, canvas: _Canvas, operation: _Operation) -> None:
        """undo 또는 캐시 재구성 시 작업을 원래 순서대로 재생한다."""
        if operation.erased_ids is not None:
            return
        if operation.clear:
            canvas.color.fill(0)
            canvas.alpha.fill(0)
            canvas.composed[:] = canvas.background
            return
        saved = operation.stroke
        assert saved is not None
        stroke = Stroke(saved.stroke_id, saved.page, saved.tool, saved.color, saved.width, [])
        active = _Active(stroke, operation.opacity, np.zeros(canvas.background.shape[:2], np.uint8))
        for point in saved.points:
            self._paint_segment(active, point)
        self._apply(canvas, active)

    def begin_stroke(self, tool: Tool = 'pen', *, color: str | None = None,
                     width: float | None = None) -> None:
        """새 작업 시작. eraser도 한 번의 연속 작업을 한 undo 단위로 기록한다."""
        if tool not in ('pen', 'highlighter', 'eraser', 'pointer'):
            raise ValueError(f'지원하지 않는 도구: {tool}')
        default_color, default_width, opacity = self._settings(tool)
        selected_color = color if color is not None else default_color
        selected_width = width if width is not None else default_width
        color_bgr(selected_color)
        if not math.isfinite(selected_width) or selected_width <= 0:
            raise ValueError('width는 유한한 양수여야 합니다.')
        self.end_stroke()
        if tool == 'pointer':
            return
        self._next_id += 1
        stroke = Stroke(self._next_id, self.current_page, tool, selected_color, selected_width, [])
        self._active = _Active(stroke, opacity,
                               np.zeros(self._canvas(self.current_page).background.shape[:2], np.uint8))

    def add_point(self, x: float, y: float) -> bool:
        """슬라이드 픽셀 좌표 추가. 범위 밖/NaN은 획을 종료하여 연결선을 막는다."""
        if self._active is None:
            return False
        w, h = self.slide_size
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < w and 0 <= y < h):
            self.end_stroke()
            return False
        self._paint_segment(self._active, (float(x), float(y)))
        return True

    def end_stroke(self) -> None:
        """현재 획 확정. 손 추적 실패/도구 변경/페이지 이동 전에도 호출한다."""
        active = self._active
        if active is None:
            return
        if active.stroke.points:
            self._apply(self._canvas(active.stroke.page), active)
            self._history[active.stroke.page].append(_Operation(active.stroke, active.opacity))
            self._redo[active.stroke.page].clear()
        self._active = None

    def discard_preview(self) -> None:
        """미리보기를 확정하지 않고 제거한다. 보정 모듈의 최종 획으로 대체할 때 사용한다."""
        self._active = None

    def set_preview(self, points: list[Point] | None, tool: Tool = 'pen', *,
                    color: str | None = None, width: float | None = None) -> None:
        """보정 모듈의 진행 중 획을 복사해 표시한다. 짧아지면 마스크도 다시 만든다.

        기존 점에 이어진 경우 새 구간만 그린다. pen_down 기반 update API와 혼용하지 않는다.
        """
        if points is None:
            self.discard_preview()
            return
        default_color, default_width, _ = self._settings(tool)
        chosen_color = color if color is not None else default_color
        chosen_width = width if width is not None else default_width
        copied = [(float(x), float(y)) for x, y in points]
        w, h = self.slide_size
        if any(not (math.isfinite(x) and math.isfinite(y) and 0 <= x < w and 0 <= y < h)
               for x, y in copied):
            raise ValueError('미리보기 좌표는 슬라이드 안의 유한한 픽셀 좌표여야 합니다.')
        active = self._active
        same_style = active is not None and (
            active.stroke.tool, active.stroke.color, active.stroke.width
        ) == (tool, chosen_color, chosen_width)
        prefix = (same_style and len(copied) >= len(active.stroke.points)
                  and copied[:len(active.stroke.points)] == active.stroke.points)
        if not prefix:
            self.discard_preview()
            self.begin_stroke(tool, color=chosen_color, width=chosen_width)
        if self._active is not None:
            for point in copied[len(self._active.stroke.points):]:
                self.add_point(*point)

    def commit_stroke(self, points: list[Point], tool: Tool = 'pen', *,
                      color: str | None = None, width: float | None = None) -> None:
        """꼬리 제거가 끝난 최종 획으로 미리보기를 교체하여 한 번만 확정한다."""
        self.discard_preview()
        if not points:
            return
        self.set_preview(points, tool, color=color, width=width)
        self.end_stroke()

    def update(self, cursor: Point | None, tool: Tool, pen_down: bool, *,
               eraser_active: bool = False, color: str | None = None,
               width: float | None = None) -> None:
        """통합용 입력 어댑터. 주먹 지우개는 pen_down과 별도 eraser_active를 쓴다.

        UI/메인 루프가 확정한 tool과 보정 좌표만 받는다. GestureState를 변경하지 않는다.
        """
        enabled = eraser_active if tool == 'eraser' else pen_down and tool in ('pen', 'highlighter')
        if cursor is None or not enabled:
            self.end_stroke()
            return
        default_color, default_width, _ = self._settings(tool)
        chosen_color = color if color is not None else default_color
        chosen_width = width if width is not None else default_width
        active = self._active
        if active is None or (active.stroke.tool, active.stroke.color, active.stroke.width) != (tool, chosen_color, chosen_width):
            self.begin_stroke(tool, color=chosen_color, width=chosen_width)
        self.add_point(*cursor)

    def erase_stroke_at(self, point: Point, radius: float = 15) -> bool:
        """접촉한 최상단 획 하나를 지운다. PDF 배경과 다른 획은 보존한다."""
        self.end_stroke()
        history = self._history[self.current_page]
        erased = {i for op in history for i in (op.erased_ids or [])}
        p = np.asarray(point, dtype=float)
        for op in reversed(history):
            if op.clear:
                break
            s = op.stroke
            if s is None or s.tool == 'eraser' or s.stroke_id in erased or not s.points:
                continue
            pts = np.asarray(s.points)
            a, b = pts[:-1], pts[1:]
            if len(a):
                v = b-a
                t = np.clip(((p-a)*v).sum(axis=1)/np.maximum((v*v).sum(axis=1),1e-12),0,1)
                distance = np.linalg.norm(a+t[:,None]*v-p,axis=1).min()
            else:
                distance = np.linalg.norm(pts[0]-p)
            if distance <= radius+s.width/2:
                history.append(_Operation(erased_ids=[s.stroke_id]))
                self._redo[self.current_page].clear()
                self._cache.pop(self.current_page,None)
                self._canvas(self.current_page)
                return True
        return False

    def render_page(self) -> np.ndarray:
        """UI 없는 BGR 이미지 복사본. 진행 중 획은 원본 상태에 한 번만 합성한다."""
        canvas = self._canvas(self.current_page)
        result = canvas.composed.copy()
        active = self._active
        if active is not None and active.bounds is not None:
            x0, y0, x1, y1 = active.bounds
            region = np.s_[y0:y1, x0:x1]
            a = active.mask[region].astype(np.float32)[..., None] / 255.0 * active.opacity
            if active.stroke.tool == 'eraser':
                value = canvas.color[region] * (1 - a) + canvas.background[region] * (1 - canvas.alpha[region] * (1 - a))
            else:
                value = np.array(color_bgr(active.stroke.color), np.float32) * a + canvas.composed[region] * (1 - a)
            result[region] = np.rint(value).clip(0, 255).astype(np.uint8)
        return result

    def goto_page(self, page: int) -> bool:
        """0부터 시작하는 페이지로 이동한다. 범위 밖이면 현재 페이지를 유지한다."""
        if not 0 <= page < self.page_count:
            return False
        self.end_stroke()
        self.current_page = page
        self._canvas(page)
        return True

    def change_page(self, offset: int) -> bool:
        """+1은 다음, -1은 이전 페이지. 현재 작업을 종료한 뒤 이동한다."""
        return self.goto_page(self.current_page + offset)

    def undo(self) -> bool:
        """현재 페이지의 마지막 작업을 취소한다. 지우개 작업도 복원한다."""
        self.end_stroke()
        history = self._history[self.current_page]
        if not history:
            return False
        self._redo[self.current_page].append(history.pop())
        self._cache.pop(self.current_page, None)
        self._canvas(self.current_page)
        return True

    def redo(self) -> bool:
        """현재 페이지에서 실행 취소한 마지막 작업을 다시 적용한다."""
        self.end_stroke()
        stack = self._redo[self.current_page]
        if not stack:
            return False
        self._history[self.current_page].append(stack.pop())
        self._cache.pop(self.current_page, None)
        self._canvas(self.current_page)
        return True

    def clear_page(self) -> None:
        """현재 페이지의 필기만 지우고 실행 취소가 가능한 작업으로 기록한다."""
        self.end_stroke()
        operation = _Operation(clear=True)
        self._history[self.current_page].append(operation)
        self._redo[self.current_page].clear()
        self._replay(self._canvas(self.current_page), operation)

    def operation_count(self, page: int | None = None) -> int:
        """확정된 작업 수를 반환한다. 내부 이력은 직접 공개하지 않는다."""
        return len(self._history[self.current_page if page is None else page])

    def save_png(self, path: str | Path) -> Path:
        """현재 페이지를 PNG로 저장한다. Windows 한글 경로도 지원한다."""
        path = Path(path).resolve()
        if path.suffix.lower() != '.png':
            raise ValueError('PNG 출력 확장자는 .png 이어야 합니다.')
        self.end_stroke()
        success, encoded = cv2.imencode('.png', self.render_page())
        if not success:
            raise OSError('PNG 인코딩 실패')
        self._atomic_write(path, encoded.tobytes())
        return path

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        """완성된 파일만 대상 경로로 이동한다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def save_pdf(self, path: str | Path) -> Path:
        """전체 페이지를 합성 이미지 PDF로 저장. 원본/현재 페이지는 보존한다.

        출력은 래스터 PDF이며 원본 텍스트 검색/링크/개별 획 편집은 보존하지 않는다.
        각 페이지의 표시 크기와 회전된 방향을 유지한다. 내보내기는 동기식이다.
        """
        path = Path(path).resolve()
        if path.suffix.lower() != '.pdf':
            raise ValueError('PDF 출력 확장자는 .pdf 이어야 합니다.')
        if path == self.pdf_path:
            raise ValueError('원본 PDF와 다른 이름으로 저장하세요.')
        self.end_stroke()
        output = pymupdf.open()
        try:
            for index in range(self.page_count):
                canvas = self._canvas(index)
                success, encoded = cv2.imencode('.png', canvas.composed)
                if not success:
                    raise OSError('페이지 PNG 인코딩 실패')
                rect = self.doc[index].rect
                page = output.new_page(width=rect.width, height=rect.height)
                page.insert_image(page.rect, stream=encoded.tobytes(), keep_proportion=False)
            self._atomic_write(path, output.tobytes(garbage=4, deflate=True))
        finally:
            output.close()
        return path

    @property
    def document_key(self) -> str:
        """PDF 내용으로 구분하므로 같은 이름의 다른 PDF와 상태가 섞이지 않는다."""
        if not hasattr(self, '_document_key'):
            digest = hashlib.sha256()
            with self.pdf_path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            self._document_key = digest.hexdigest()
        return self._document_key

    def save_session(self, path: str | Path) -> None:
        """현재 페이지와 편집 이력을 원자적으로 저장한다. 진행 획도 보존한다."""
        history = [[asdict(op) for op in page] for page in self._history]
        if self._active is not None and self._active.stroke.points:
            active = self._active
            history[active.stroke.page].append(asdict(_Operation(active.stroke, active.opacity)))
        data = {'version':1, 'document':self.document_key,
                'render_size':[self.config.max_width,self.config.max_height],
                'current_page':self.current_page, 'history':history}
        self._atomic_write(Path(path),json.dumps(data,allow_nan=False).encode('utf-8'))

    def load_session(self, path: str | Path) -> None:
        """검증된 JSON만 복원한다. 실패하면 현재 필기 상태를 변경하지 않는다."""
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            if (data['version']!=1 or data['document']!=self.document_key
                    or data['render_size']!=[self.config.max_width,self.config.max_height]):
                raise ValueError('PDF 또는 렌더링 크기가 저장 상태와 다릅니다.')
            page=data['current_page']
            if type(page) is not int or not 0<=page<self.page_count:
                raise ValueError('잘못된 페이지 번호')
            raw=data['history']
            if not isinstance(raw,list) or len(raw)!=self.page_count:
                raise ValueError('잘못된 페이지 이력')
            history=[]
            next_id=0
            for index,operations in enumerate(raw):
                if not isinstance(operations,list):
                    raise ValueError('잘못된 작업 목록')
                restored=[]
                if not operations:
                    history.append(restored)
                    continue
                w,h=self._canvas(index).background.shape[1::-1]
                for item in operations:
                    if item.get('erased_ids') is not None:
                        ids = item['erased_ids']
                        if not isinstance(ids,list) or any(type(i) is not int or i<1 for i in ids):
                            raise ValueError('Invalid erased stroke ids')
                        restored.append(_Operation(erased_ids=ids))
                        continue
                    if item.get('clear') is True:
                        restored.append(_Operation(clear=True))
                        continue
                    stroke=Stroke(**item['stroke'])
                    opacity=item['opacity']
                    if (stroke.page!=index or stroke.tool not in ('pen','highlighter','eraser')
                            or type(stroke.stroke_id) is not int or stroke.stroke_id<1
                            or not math.isfinite(stroke.width) or not 0<stroke.width<=10000
                            or not math.isfinite(opacity) or not 0<opacity<=1):
                        raise ValueError('잘못된 획 속성')
                    color_bgr(stroke.color)
                    points=[]
                    for x,y in stroke.points:
                        if not(math.isfinite(x) and math.isfinite(y) and 0<=x<w and 0<=y<h):
                            raise ValueError('잘못된 획 좌표')
                        points.append((float(x),float(y)))
                    stroke.points=points
                    next_id=max(next_id,stroke.stroke_id)
                    restored.append(_Operation(stroke,opacity))
                history.append(restored)
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            raise ValueError(f'필기 저장 파일을 복원할 수 없습니다: {path}') from exc
        self._active=None
        self._history=history
        self._redo=[[] for _ in self.doc]
        self._next_id=next_id
        self._cache.clear()
        self.current_page=page
        self._canvas(page)

    def close(self) -> None:
        """PDF와 캐시를 해제한다. 파일은 저장하지 않으므로 종료 전 별도 저장한다."""
        if not self._closed:
            self._active = None
            self._cache.clear()
            self.doc.close()
            self._closed = True

    def __enter__(self) -> AirBoardRenderer:
        """with 구문 지원."""
        return self

    def __exit__(self, *_: object) -> None:
        """with 종료 시 자원 해제."""
        self.close()
