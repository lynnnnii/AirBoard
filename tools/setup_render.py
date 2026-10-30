"""기존 smoothing 설정을 유지하면서 render 기본 설정과 작은 테스트 PDF를 준비한다."""
from pathlib import Path
import json
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
RENDER_DEFAULTS = {
    'max_width': 1280, 'max_height': 640, 'cache_pages': 3,
    'pen_color': '#FF3030', 'pen_width': 4.0,
    'highlighter_color': '#FFFF00', 'highlighter_width': 24.0,
    'highlighter_opacity': 0.3, 'eraser_width': 40.0,
}


def main() -> None:
    """기존 키의 값은 바꾸지 않고 없는 render 키만 추가한다."""
    path = ROOT / 'config' / 'default.json'
    config = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    previous = config.get('render', {})
    config['render'] = {**RENDER_DEFAULTS, **previous}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    pdf_path = ROOT / 'data' / 'pdfs' / 'render_sample.pdf'
    if not pdf_path.exists():
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with pymupdf.open() as doc:
            for index, bg in enumerate(((.94, .96, 1), (1, .97, .90))):
                page = doc.new_page(width=960, height=480)
                page.draw_rect(page.rect, color=None, fill=bg)
                page.insert_text((50, 80), f'AirBoard test page {index + 1}', fontsize=28)
                page.insert_text((50, 130), 'Draw, highlight, erase, undo and return to this page.', fontsize=16)
            doc.save(pdf_path)
    gestures = ROOT / 'config' / 'gestures.json'
    if not gestures.exists():
        gestures.write_text(json.dumps({
            'DOUBLE_CLAP': 'toggle_highlighter',
            'SWIPE_RIGHT': 'page_next',
            'SWIPE_LEFT': 'page_prev',
        }, indent=2) + '\n', encoding='utf-8')
    print(f'설정 준비 완료: {path}')
    print(f'테스트 PDF: {pdf_path}')
    print('기존 smoothing 설정과 기존 render 값은 유지했습니다.')


if __name__ == '__main__':
    main()
