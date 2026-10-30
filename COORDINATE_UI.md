# 임준서 파트: 좌표 변환 및 OpenCV UI

## 설치와 실행 (Windows / VS Code)

저장소 루트, 즉 `main.py`가 있는 `C:\Users\justi\AirBoard\AirBoard`를 VS Code로 엽니다.
터미널에서 다음 명령을 실행합니다. Python 3.12와 MediaPipe 0.10.21 기준입니다.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
# 기본: 실제 웹캠의 핀치·주먹·박수 제스처
.\.venv\Scripts\python.exe -m airboard.ui.demo
# 외장 웹캠 인덱스 지정
.\.venv\Scripts\python.exe -m airboard.ui.demo --camera-index 1
# 웹캠 없이 테스트
.\.venv\Scripts\python.exe -m airboard.ui.demo --mouse
.\.venv\Scripts\python.exe -m airboard.ui.demo --virtual
# 녹화 영상 입력 (실시간 처리 속도 기준으로 판정)
.\.venv\Scripts\python.exe -m airboard.ui.demo --video data/demo.mp4
# 자동 테스트 / 창 없는 실행
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m airboard.ui.demo --virtual --headless --frames 120
```

VS Code에서 Ctrl+Shift+P → Python: Select Interpreter → 이 저장소 `.venv\Scripts\python.exe`를 선택합니다. `pip` 단독 대신 위와 같이 같은 Python의 `-m pip`를 사용해야 Anaconda/Windows Python 간 설치 위치 불일치를 피할 수 있습니다. 가상 환경 활성화나 PowerShell 실행 정책 변경은 필요하지 않습니다.

OpenCV 패키지는 `opencv-contrib-python` 하나만 사용하며 `import cv2`로 불러옵니다. 원래 환경에 opencv-python/headless 등이 함께 있다면 기존 환경 대신 위의 새 `.venv`를 사용하세요. UI/가상 손/마우스에는 MediaPipe를 import하지 않으며 실제 웹캠/영상 모드에서만 사용합니다.

## 동작

왼쪽 패널은 좌우 반전된 카메라 입력, 오른쪽은 흰 슬라이드 미리보기입니다.

- 웹캠: 핀치 유지→필기, 해제→획 종료, 박수 두 번→펜/형광펜, 주먹 유지→지우개, 해제→이전 도구, 검지 포인터→포인터. 인식은 팀원의 기존 `HandTracker`/`GestureRecognizer`를 사용합니다.
- `C`: 재캘리브레이션. 손끝을 **좌상→우상→우하→좌하**에 두고 각각 Space를 누릅니다. 또는 왼쪽 입력 패널에서 네 점을 클릭합니다. C를 다시 누르면 처음부터 시작합니다. 잘못된 사각형은 거부하고 이전 변환을 유지합니다.
- 상단 팔레트에 커서를 설정된 시간(기본 0.65초) 머무르면 색상/굵기 변경. 팔레트 조작 중에는 필기하지 않습니다.
- `S`: 설정 화면. 위/아래로 제스처 행 선택, 좌/우로 연결 기능 변경, Enter 저장·즉시 적용, Esc 취소. 제스처 이름 자체를 추가하려면 `config/gestures.json`을 편집합니다. 해당 이름은 인식기가 실제로 출력해야 합니다.
- 좌/우 방향키: 비상용 페이지 번호 변경. 제스처 페이지 넘김은 필기 중 차단합니다. `--pdf`가 없으면 번호와 명령만 확인하고, `--pdf` 실행에서는 실제 페이지를 이동합니다.
- `X`: 미리보기 필기 지우기. `Q` 또는 설정 화면 밖의 Esc: 종료.
- 마우스 모드: 왼쪽 입력 패널에서 왼쪽 버튼 드래그가 핀치입니다. 도구 전환은 웹캠 제스처로 확인합니다.
- `--virtual`: 자동 곡선 입력과 주기적인 핀치 해제로 획 분리를 보여줍니다.

사각형 밖·손 소실·도구 변경 시 기존 획을 종료합니다. 주먹/포인터를 해제한 순간 핀치가 계속 켜져 있다면 한 번 해제한 뒤 새 핀치를 해야 필기가 재개됩니다. 호모그래피는 손이 대체로 하나의 가상 평면에서 움직인다는 가정이 있습니다. 카메라 해상도·좌우 반전 조건이나 카메라 위치를 변경하면 재캘리브레이션하세요.

## 파일과 책임

| 위치 | 역할 |
| --- | --- |
| `airboard/calibration/homography.py` | 0~1 정규화 입력 → 슬라이드 픽셀, OpenCV 호모그래피, 유효성 검사, 저장 |
| `airboard/ui/controller.py` | 입력/출력 데이터, GestureResult 어댑터, 도구 우선순위·획 경계 |
| `airboard/ui/overlay.py` | OpenCV 커서·머무르기 팔레트·도구 알림 |
| `airboard/ui/settings.py` | JSON 설정 로드·검증·저장 |
| `airboard/ui/demo.py` | 웹캠/영상/마우스/가상 손 실행과 흰 배경 미리보기 |
| `config/default.json` | ui 설정값 (인식 담당의 설정을 추가할 수 있도록 ui 키 아래 분리) |
| `config/gestures.json` | 제스처→기능 기본 프리셋의 유일한 정의 위치 |
| `tests/test_coordinate_ui.py` | 화면 없이 실행하는 자동 테스트 |

개인 파일 `config/ui-user.json`, `config/calibration-user.json`은 Git에서 제외합니다. 잘못된 사용자 JSON으로 실행이 중단되면 해당 파일을 다른 이름으로 이동하여 기본 설정으로 복구할 수 있습니다.

## 통합 인터페이스 (기존 팀 모듈 수정 없음)

팀 문서의 정규화 좌표 계약을 유지합니다. 현재 실제 인식 모듈은 픽셀 좌표의 `GestureResult`를 출력하므로 어댑터가 정규화합니다. 다른 팀원의 내부 상태나 CFG를 참조하지 않습니다.

```python
from airboard.calibration import Calibration
from airboard.ui.settings import Settings
from airboard.ui.controller import UIController, from_gesture_result

settings = Settings()
calibration = Calibration(settings.ui['default_corners'], (1920, 1080))
ui = UIController(calibration, settings)

# 매 프레임: frame은 좌우 반전 후의 카메라 프레임
# hands = tracker.detect(frame)
# result = recognizer.update(hands, frame.shape[1], frame.shape[0], now)
# state = from_gesture_result(result, (frame.shape[1], frame.shape[0]))
# packet = ui.update(state, now)
# renderer.consume(packet)
```

`InputState(position, gestures)`의 position은 **좌우 반전 후 0~1 카메라 좌표**입니다. `PINCH/FIST/POINTER`는 유지되는 동안 매 프레임, `DOUBLE_CLAP/SWIPE_*`는 발생 프레임에만 넣습니다. 이미 인식한 박수 이벤트는 손이 소실된 프레임에도 처리합니다. 손이 소실되면 position=None을 반드시 전달하세요. 외부 입력 스레드가 멈추면 통합 루프에서 타임아웃을 판단하여 None을 보내야 합니다.

`RenderPacket`의 position은 `(0,0)~(width-1,height-1)` 슬라이드 픽셀 또는 None, tool은 pen/highlighter/eraser/pointer, active는 쓰기/지우기 여부, phase는 down/move/up/hover, color는 #RRGGBB, width는 논리 픽셀, opacity는 형광펜 불투명도, commands는 page_next/page_prev 튜플입니다. **up은 position이 None이어도 직전 획을 종료해야 합니다.** up 패킷의 tool이 새 도구일 수 있으므로 렌더러는 열린 획의 도구를 별도로 보관합니다. 스무딩 담당은 down/up에서 필터를 초기화하면 됩니다.

`PreviewRenderer`는 독립 시연을 위한 흰 배경 구현입니다. `--pdf`에서는 `PDFRenderer` 어댑터가 이예린의 render 모듈을 사용합니다. PDF 저장 UI와 떨림 보정 연결은 별도입니다. 미리보기 지우개는 흰 배경 전용이며 실제 PDF에 흰 선을 덮는 방식으로 연결하면 안 됩니다. 형광펜은 프레임별 합성이라 같은 곳을 반복하면 진해집니다. 팀 main.py는 변경하지 않았고 실행은 위 모듈 명령을 사용합니다.

## 확인 후 브랜치 업로드

아래는 직접 실행할 명령입니다. 커밋/푸시는 자동으로 수행하지 않았습니다.

```powershell
git status
git add airboard/calibration airboard/ui config/default.json config/gestures.json tests/test_coordinate_ui.py requirements-ui.txt COORDINATE_UI.md README.md .gitignore
git diff --cached --stat
git commit -m "feat: 좌표 캘리브레이션 및 OpenCV UI 구현"
git push origin features/coordinate-ui
```

공통 main/core 또는 다른 팀원의 인터페이스를 바꾸지 않았습니다. PR에는 현재 GestureResult 픽셀→정규화 어댑터를 사용함을 적고, 손 인식 담당과 통합 담당에게 연결부 리뷰를 요청하세요.

## 구현 시 검증 기록

- 저장소 전용 Python 3.12 환경: MediaPipe 0.10.21 / OpenCV 4.11.0 / NumPy 1.26.4.
- 최신 검증: `unittest discover -s tests -v` 전체 49개 통과 (좌표·UI·보정·렌더링·UI/PDF 연결).
- 가상 손 headless 120프레임 처리: 정상 종료.
- OpenCV 가상 손 창 30프레임 시작·자동 종료: 정상.
- 팀 HandTracker가 실제 모델을 로드하고 빈 영상 프레임을 처리한 뒤 GestureRecognizer → UI 어댑터 연결: 정상 (position=None).
- `pip check`: 패키지 충돌 없음.
- 사용자 웹캠 확인: 핀치, 영역 제한, 팔레트, 페이지 명령, 박수 전환. 박수는 인식이 불안정한 경우가 있습니다. 실제 강의 PDF의 표시·페이지별 필기·배경 보존은 자동 실행으로 확인했으며, 웹캠/PDF 조합의 수동 검증과 정량 성능 평가는 별도입니다.

## UI 사용성 업데이트

- 페이지·도구·이동 결과는 필기 영역 밖의 불투명 하단 상태줄에 표시합니다.
- 도구는 제스처로만 전환합니다. 숫자키 및 H/F/P 도구 전환은 사용하지 않습니다. 설정 안내는 큰 글씨와 진한 배경으로 표시합니다.
- 좌우 반전된 카메라 화면 기준 왼쪽 스와이프/왼쪽 방향키는 이전, 오른쪽은 다음입니다. 첫 페이지에서 이전으로 이동할 수 없으면 상태줄에 안내합니다.
- 설정 화면 S에서 R을 누르면 기본 제스처 매핑으로 복원합니다. Enter로 저장하고 Esc로 취소합니다.
- 유지되는 제스처를 도구 토글에 연결해도 제스처가 끝날 때까지 한 번만 전환합니다.

## PDF와 함께 확인하기

```powershell
.\.venv\Scripts\python.exe -m airboard.ui.demo --pdf "C:\Users\justi\Downloads\11. lecture-Segmentation.pdf"
# 전체 화면
.\.venv\Scripts\python.exe -m airboard.ui.demo --pdf "C:\Users\justi\Downloads\11. lecture-Segmentation.pdf" --fullscreen
```

기본 창은 1600×900이며 크기 조절이 가능합니다. PDF 모드에서는 카메라 패널을 줄이고 PDF 표시 영역을 넓힙니다.
스와이프 또는 방향키로 실제 PDF 페이지가 바뀌고, 페이지별 필기가 보존됩니다. X는 현재 페이지 필기만 지우며 PDF 원본은 유지합니다.
도구 전환은 제스처로만 합니다. PDF를 지정하지 않으면 기존 흰 배경 데모입니다.
이 연결은 좌표·UI→PDF 렌더러 통합이며, 떨림 보정은 아직 연결하지 않았습니다. 필기는 실행 중에만 유지되며 종료 시 자동 저장하지 않습니다.
외부 PDF 경로를 직접 지정할 수 있습니다. 팀 테스트용 분할 자료는 아래 별도 경로로 포함했습니다.

- 일반 화면에서 `R`: 캘리브레이션을 기본 사각형으로 복원하고 저장합니다. PDF 페이지 크기와 기존 필기는 유지되며, 핀치를 해제한 뒤 다시 쓰세요. 설정 화면 안의 `R`은 제스처 매핑 복원입니다.
- 설정 저장 성공 문구는 하단에 표시하지 않습니다.

## 저장소에 포함된 팀 테스트 PDF

`data/pdfs/ui-test/README.md`에 분할 파일과 수동 검증 순서를 정리했습니다.

```powershell
.\.venv\Scripts\python.exe -m airboard.ui.demo --pdf data/pdfs/ui-test/segmentation-p01-15.pdf
```

사용자가 요청한 분할 테스트 PDF 세 개만 `.gitignore` 예외로 포함합니다. 개인 캘리브레이션·제스처 설정은 계속 제외됩니다.
