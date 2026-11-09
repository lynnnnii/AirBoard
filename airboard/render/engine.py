"""AirBoard PDF display, ink layers, per-page undo, and export.

This module does not open windows or webcams. Inputs are calibrated slide pixel coordinates;
outputs are BGR images without UI elements. Call from the same thread.
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
    """Stroke data format: zero-based page, #RRGGBB color, and diameter in pixels."""
    stroke_id: int
    page: int
    tool: Tool
    color: str
    width: float
    points: list[Point]


@dataclass(frozen=True)
class RenderConfig:
    """Configuration loaded from the render section of config/default.json."""
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
        """Reject invalid dimensions and opacity early."""
        if min(self.max_width, self.max_height, self.cache_pages) < 1:
            raise ValueError('Rendering dimensions and cache_pages must be positive.')
        if not 0 < self.highlighter_opacity <= 1:
            raise ValueError('highlighter_opacity must be in (0, 1].')
        for value in (self.pen_width, self.highlighter_width, self.eraser_width):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Tool width must be a finite positive number.')
        color_bgr(self.pen_color)
        color_bgr(self.highlighter_color)

    @classmethod
    def from_json(cls, path: str | Path) -> RenderConfig:
        """Do not read or modify unrelated teammates' configuration sections."""
        return cls(**json.loads(Path(path).read_text(encoding='utf-8'))['render'])


def color_bgr(color: str) -> tuple[int, int, int]:
    """Convert hexadecimal colors to OpenCV BGR."""
    if len(color) != 7 or not color.startswith('#'):
        raise ValueError('color must use #RRGGBB format.')
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


@dataclass
class _Operation:
    """Store opacity per operation so later settings changes do not affect existing strokes."""
    stroke: Stroke | None = None
    opacity: float = 1.0
    clear: bool = False
    erased_ids: list[int] | None = None


@dataclass
class _Canvas:
    """Keep the original PDF separate from premultiplied color and alpha layers."""
    background: np.ndarray
    color: np.ndarray
    alpha: np.ndarray
    composed: np.ndarray


@dataclass
class _Active:
    """One continuous ink or eraser operation; overlapping masks do not repeatedly darken the same stroke."""
    stroke: Stroke
    opacity: float
    mask: np.ndarray
    bounds: tuple[int, int, int, int] | None = None


class AirBoardRenderer:
    """Own PDF and ink state without depending on tracking, gestures, or UI modules."""

    def __init__(self, pdf_path: str | Path, config: RenderConfig) -> None:
        """Open the PDF and cache its first page without pre-rendering every page."""
        self.pdf_path = Path(pdf_path).resolve()
        self.config = config
        self.doc = pymupdf.open(self.pdf_path)
        if self.doc.needs_pass or len(self.doc) == 0:
            self.doc.close()
            raise ValueError('A non-password-protected PDF with at least one page is required.')
        self.current_page = 0
        self._history: list[list[_Operation]] = [[] for _ in self.doc]
        self._redo_history: list[list[_Operation]] = [[] for _ in self.doc]
        self._cache: OrderedDict[int, _Canvas] = OrderedDict()
        self._active: _Active | None = None
        self._next_id = 0
        self._closed = False
        self._canvas(0)

    @property
    def page_count(self) -> int:
        """Total page count."""
        return len(self.doc)

    @property
    def slide_size(self) -> tuple[int, int]:
        """Current slide dimensions (width, height), provided to coordinate mapping."""
        height, width = self._canvas(self.current_page).background.shape[:2]
        return width, height

    def _canvas(self, index: int) -> _Canvas:
        """Read a rendered page from the LRU cache or render it once when absent."""
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
        """Default color, diameter, and opacity for each tool mode."""
        c = self.config
        if tool == 'pen':
            return c.pen_color, c.pen_width, 1.0
        if tool == 'highlighter':
            return c.highlighter_color, c.highlighter_width, c.highlighter_opacity
        return c.pen_color, c.eraser_width, 1.0

    @staticmethod
    def _paint_segment(active: _Active, point: Point) -> None:
        """Update the mask between the previous and current points; render isolated points as dots."""
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
        """Composite only the mask bounds; erasing removes ink alpha, not the PDF background."""
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
        """Replay operations in original order during undo or cache reconstruction."""
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
        """Start an operation; each continuous eraser action counts as one undo step."""
        if tool not in ('pen', 'highlighter', 'eraser', 'pointer'):
            raise ValueError(f'Unsupported tool: {tool}')
        default_color, default_width, opacity = self._settings(tool)
        selected_color = color if color is not None else default_color
        selected_width = width if width is not None else default_width
        color_bgr(selected_color)
        if not math.isfinite(selected_width) or selected_width <= 0:
            raise ValueError('width must be a finite positive number.')
        self.end_stroke()
        if tool == 'pointer':
            return
        self._next_id += 1
        stroke = Stroke(self._next_id, self.current_page, tool, selected_color, selected_width, [])
        self._active = _Active(stroke, opacity,
                               np.zeros(self._canvas(self.current_page).background.shape[:2], np.uint8))

    def add_point(self, x: float, y: float) -> bool:
        """Append slide pixel coordinates; out-of-bounds or NaN points end the stroke to avoid connecting lines."""
        if self._active is None:
            return False
        w, h = self.slide_size
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < w and 0 <= y < h):
            self.end_stroke()
            return False
        self._paint_segment(self._active, (float(x), float(y)))
        return True

    def end_stroke(self) -> None:
        """Finalize the current stroke; also call before tracking loss, tool changes, or page navigation."""
        active = self._active
        if active is None:
            return
        if active.stroke.points:
            self._apply(self._canvas(active.stroke.page), active)
            page = active.stroke.page
            self._history[page].append(_Operation(active.stroke, active.opacity))
            self._redo_history[page].clear()
        self._active = None

    def discard_preview(self) -> None:
        """Discard uncommitted preview ink when replacing it with the smoother's final stroke."""
        self._active = None

    def set_preview(self, points: list[Point] | None, tool: Tool = 'pen', *,
                    color: str | None = None, width: float | None = None) -> None:
        """Preview an in-progress smoothed stroke; rebuild its mask if the stroke shrinks.

        If points extend an existing stroke, draw only new segments. Do not mix with pen_down-based update API.
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
            raise ValueError('Preview coordinates must be finite pixel positions within the slide.')
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
        """Replace the preview with the tail-trimmed final stroke and commit it once."""
        self.discard_preview()
        if not points:
            return
        self.set_preview(points, tool, color=color, width=width)
        self.end_stroke()

    def update(self, cursor: Point | None, tool: Tool, pen_down: bool, *,
               eraser_active: bool = False, color: str | None = None,
               width: float | None = None) -> None:
        """Integration input adapter: fist eraser uses eraser_active independently of pen_down.

        Receive only the tool and calibrated coordinates selected by the UI; do not mutate GestureState.
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
        """Erase the topmost touched stroke while preserving the PDF background and other strokes."""
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
                self._redo_history[self.current_page].clear()
                self._cache.pop(self.current_page,None)
                self._canvas(self.current_page)
                return True
        return False

    def render_page(self) -> np.ndarray:
        """Return a BGR image copy without UI; composite preview ink only once."""
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
        """Navigate to a zero-based page; remain on the current page if out of range."""
        if not 0 <= page < self.page_count:
            return False
        self.end_stroke()
        self.current_page = page
        self._canvas(page)
        return True

    def change_page(self, offset: int) -> bool:
        """+1 means next page, -1 previous; finish the current operation before navigating."""
        return self.goto_page(self.current_page + offset)

    def undo(self) -> bool:
        """Undo the last operation on this page, including eraser operations."""
        self.end_stroke()
        history = self._history[self.current_page]
        if not history:
            return False
        canvas = self._canvas(self.current_page)
        self._redo_history[self.current_page].append(history.pop())
        self._rebuild_page(canvas, history)
        return True

    def redo(self) -> bool:
        """Redo the last undone operation on this page."""
        self.end_stroke()
        redo = self._redo_history[self.current_page]
        if not redo:
            return False
        operation = redo.pop()
        history = self._history[self.current_page]
        history.append(operation)
        self._rebuild_page(self._canvas(self.current_page), history)
        return True

    def _rebuild_page(self, canvas: _Canvas, history: list[_Operation]) -> None:
        """Reconstruct the current page's ink composition from operation history."""
        # Reuse the cached PDF background and rebuild only ink layers.
        canvas.color.fill(0)
        canvas.alpha.fill(0)
        canvas.composed[:] = canvas.background
        erased = {i for op in history for i in (op.erased_ids or [])}
        for operation in history:
            if operation.stroke is None or operation.stroke.stroke_id not in erased:
                self._replay(canvas, operation)

    def clear_page(self) -> None:
        """Clear current-page ink as an undoable operation."""
        self.end_stroke()
        operation = _Operation(clear=True)
        self._history[self.current_page].append(operation)
        self._redo_history[self.current_page].clear()
        self._replay(self._canvas(self.current_page), operation)

    def operation_count(self, page: int | None = None) -> int:
        """Return the count of committed operations without exposing internal history."""
        return len(self._history[self.current_page if page is None else page])

    def save_png(self, path: str | Path) -> Path:
        """Save the current page to PNG, including Unicode paths on Windows."""
        path = Path(path).resolve()
        if path.suffix.lower() != '.png':
            raise ValueError('PNG output must use the .png extension.')
        self.end_stroke()
        success, encoded = cv2.imencode('.png', self.render_page())
        if not success:
            raise OSError('PNG encoding failed')
        self._atomic_write(path, encoded.tobytes())
        return path

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        """Move only completed output files into the destination."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(content)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def save_pdf(self, path: str | Path) -> Path:
        """Export all pages as an image-composited PDF without altering source/current page.

        The raster PDF does not retain original searchable text, links, or separately editable strokes.
        Preserve displayed page sizes and rotation; export runs synchronously.
        """
        path = Path(path).resolve()
        if path.suffix.lower() != '.pdf':
            raise ValueError('PDF output must use the .pdf extension.')
        if path == self.pdf_path:
            raise ValueError('Choose an output file different from the original PDF.')
        self.end_stroke()
        output = pymupdf.open()
        try:
            for index in range(self.page_count):
                canvas = self._canvas(index)
                success, encoded = cv2.imencode('.png', canvas.composed)
                if not success:
                    raise OSError('페이지 PNG encoding failed')
                rect = self.doc[index].rect
                page = output.new_page(width=rect.width, height=rect.height)
                page.insert_image(page.rect, stream=encoded.tobytes(), keep_proportion=False)
            self._atomic_write(path, output.tobytes(garbage=4, deflate=True))
        finally:
            output.close()
        return path

    @property
    def document_key(self) -> str:
        """Use PDF contents to distinguish sessions for different files with the same filename."""
        if not hasattr(self, '_document_key'):
            digest = hashlib.sha256()
            with self.pdf_path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            self._document_key = digest.hexdigest()
        return self._document_key

    def save_session(self, path: str | Path) -> None:
        """Atomically save the current page and edit history, including in-progress ink."""
        history = [[asdict(op) for op in page] for page in self._history]
        if self._active is not None and self._active.stroke.points:
            active = self._active
            history[active.stroke.page].append(asdict(_Operation(active.stroke, active.opacity)))
        data = {'version':1, 'document':self.document_key,
                'render_size':[self.config.max_width,self.config.max_height],
                'current_page':self.current_page, 'history':history}
        self._atomic_write(Path(path),json.dumps(data,allow_nan=False).encode('utf-8'))

    def load_session(self, path: str | Path) -> None:
        """Restore only validated JSON; leave current ink unchanged on failure."""
        try:
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            if (data['version']!=1 or data['document']!=self.document_key
                    or data['render_size']!=[self.config.max_width,self.config.max_height]):
                raise ValueError('PDF or rendering dimensions do not match the saved session.')
            page=data['current_page']
            if type(page) is not int or not 0<=page<self.page_count:
                raise ValueError('Invalid page number')
            raw=data['history']
            if not isinstance(raw,list) or len(raw)!=self.page_count:
                raise ValueError('Invalid page history')
            history=[]
            next_id=0
            for index,operations in enumerate(raw):
                if not isinstance(operations,list):
                    raise ValueError('Invalid operation list')
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
                        raise ValueError('Invalid stroke attributes')
                    color_bgr(stroke.color)
                    points=[]
                    for x,y in stroke.points:
                        if not(math.isfinite(x) and math.isfinite(y) and 0<=x<w and 0<=y<h):
                            raise ValueError('Invalid stroke coordinates')
                        points.append((float(x),float(y)))
                    stroke.points=points
                    next_id=max(next_id,stroke.stroke_id)
                    restored.append(_Operation(stroke,opacity))
                history.append(restored)
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            raise ValueError(f'Could not restore the saved ink session: {path}') from exc
        self._active=None
        self._history=history
        self._next_id=next_id
        self._cache.clear()
        self.current_page=page
        self._canvas(page)

    def close(self) -> None:
        """Release PDF and caches without saving; save separately before closing."""
        if not self._closed:
            self._active = None
            self._cache.clear()
            self.doc.close()
            self._closed = True

    def __enter__(self) -> AirBoardRenderer:
        """Support context-manager usage."""
        return self

    def __exit__(self, *_: object) -> None:
        """Release resources on context-manager exit."""
        self.close()
