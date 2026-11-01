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

## 통합 인터페이스

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

`PreviewRenderer`는 독립 시연을 위한 흰 배경 구현입니다. `--pdf`에서는 `PDFRenderer` 어댑터가 이예린의 render 모듈을 사용합니다. PDF 모드의 떨림 보정과 꼬리 제거는 팀 smoothing 모듈을 사용합니다. PDF 내보내기 UI는 별도입니다. 미리보기 지우개는 흰 배경 전용이며 실제 PDF에 흰 선을 덮는 방식으로 연결하면 안 됩니다. 형광펜은 프레임별 합성이라 같은 곳을 반복하면 진해집니다. 팀 main.py는 변경하지 않았고 실행은 위 모듈 명령을 사용합니다.

## 확인 후 브랜치 업로드

아래는 직접 실행할 명령입니다. 커밋/푸시는 자동으로 수행하지 않았습니다.

```powershell
git status
git add airboard/calibration airboard/ui config/default.json config/gestures.json tests/test_coordinate_ui.py requirements-ui.txt COORDINATE_UI.md README.md .gitignore
git diff --cached --stat
git commit -m "feat: 좌표 캘리브레이션 및 OpenCV UI 구현"
git push origin features/coordinate-ui
```

현재 main/core는 수정하지 않았지만 손 인식·제스처·필터와 PDF 저장 API는 통합 과정에서 변경했습니다. [INTEGRATION_CHANGES.md](INTEGRATION_CHANGES.md)에 변경 이유·호환성·리뷰 포인트를 정리했습니다.

## 구현 시 검증 기록

- 저장소 전용 Python 3.12 환경: MediaPipe 0.10.21 / OpenCV 4.11.0 / NumPy 1.26.4.
- 최신 검증: `unittest discover -s tests -v` 전체 88개 통과 (좌표·UI·보정·렌더링·UI/PDF 연결).
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

기본 창은 1800×1000이며 크기 조절이 가능합니다. PDF 모드에서는 카메라 패널을 줄이고 PDF 표시 영역을 넓힙니다.
스와이프 또는 방향키로 실제 PDF 페이지가 바뀌고, 페이지별 필기가 보존됩니다. X는 현재 페이지 필기만 지우며 PDF 원본은 유지합니다.
도구 전환은 제스처로만 합니다. PDF를 지정하지 않으면 기존 흰 배경 데모입니다.
PDF 모드는 좌표·UI→팀 떨림 보정/꼬리 제거→PDF 렌더러 순서로 연결됩니다. PDF별 마지막 페이지와 필기는 자동 저장되어 재실행 시 복원됩니다.
외부 PDF 경로를 직접 지정할 수 있습니다. 팀 테스트용 분할 자료는 아래 별도 경로로 포함했습니다.

- 일반 화면에서 `R`: 캘리브레이션을 기본 사각형으로 복원하고 저장합니다. PDF 페이지 크기와 기존 필기는 유지되며, 핀치를 해제한 뒤 다시 쓰세요. 설정 화면 안의 `R`은 제스처 매핑 복원입니다.
- 설정 저장 성공 문구는 하단에 표시하지 않습니다.

## 저장소에 포함된 팀 테스트 PDF

`data/pdfs/ui-test/README.md`에 분할 파일과 수동 검증 순서를 정리했습니다.

```powershell
.\.venv\Scripts\python.exe -m airboard.ui.demo --pdf data/pdfs/ui-test/segmentation-p01-15.pdf
```

사용자가 요청한 분할 테스트 PDF 세 개만 `.gitignore` 예외로 포함합니다. 개인 캘리브레이션·제스처 설정은 계속 제외됩니다.

## 입력 중단 및 빈 매핑 처리

- 영상 종료나 실행 중 카메라 읽기 실패 시 현재 획을 종료하고 PDF 창을 유지합니다. 방향키 페이지 이동·설정·종료는 계속 사용할 수 있습니다. 입력 재개는 프로그램을 다시 실행합니다.
- 빈 개인 제스처 매핑은 기본 JSON 매핑으로 복구합니다. 기본 JSON 자체가 비어 있으면 파일 경로를 포함한 오류로 안내하고, 빈 매핑 저장은 거부합니다.

## PDF 작업 자동 저장

PDF 모드에서는 마지막 페이지와 페이지별 펜·형광펜·지우개·전체 지우기 이력을 `data/sessions/`에 JSON으로 자동 저장합니다. 획 종료·페이지 이동·전체 지우기·종료 시 저장하며, 긴 획은 약 2초 간격으로 중간 저장합니다. 동일 내용의 PDF를 다시 열면 복원됩니다. PDF 내용이 바뀌면 별도 작업으로 취급하며 원본 PDF는 수정하지 않습니다. 저장 파일은 개인 데이터이므로 Git에서 제외합니다.

흰 배경 데모는 자동 저장 대상이 아닙니다. 저장이 실패하면 오류로 안내하며, 손상된 상태 파일은 덮어쓰지 않고 시작을 중단합니다. 렌더링 크기를 변경한 경우 기존 저장 상태는 자동 적용하지 않습니다.

## 필기 보정과 페이지 제스처

PDF 모드에서 One Euro 보정을 적용합니다. UI 필기용 기본 설정은 획 시작/끝 삭제를 끕니다. `config/default.json`의 `smoothing` 설정을 사용합니다. `--no-smoothing`은 필터와 꼬리 제거를 모두 꺼 비교합니다. 흰 배경 독립 데모에는 보정을 적용하지 않습니다.

페이지 이동은 한 손을 펼친 채 연속으로 가로 이동해야 합니다. 필기·주먹·포인터·박수·팔레트 조작 직후에는 기본 0.15초 동안 차단하고, 일반 한 손 페이지 이동 후에는 0.45초간 재입력을 막습니다. 손을 펼치고 잠시 멈춘 뒤 이동하세요. 값은 `ui.navigation_quiet_seconds`와 `ui.navigation_cooldown_seconds`로 조정합니다.

### 필기 감각 및 스와이프 조정

UI/PDF 필기는 `ui.handwriting_smoothing`으로 공통 보정 설정을 덮어씁니다. 기본은 min_cutoff=1.2, beta=0.008, start_skip=0, end_trim=0으로, 획을 자동 삭제하지 않고 반응을 빠르게 합니다. 순수 필터 비교는 `--no-smoothing`을 추가하세요.

UI 데모의 핀치는 `gestures` 설정을 읽습니다. 기본 시작 비율 0.20을 3프레임 확인하고, 해제 비율 0.38을 2프레임 연속 넘으면 멈춥니다. 손 소실·주먹 뒤에는 손가락을 벌린 후 새 핀치가 필요합니다. 실제 손과 카메라 조건에서 추가 튜닝이 필요할 수 있습니다.

페이지 이동은 손가락 세 개 이상이 펴진 한 손의 스와이프를 허용합니다. 잠깐의 손가락 검출 흔들림은 허용하되 필기·주먹·포인터 이동은 이력에서 제외합니다.

### 페이지 이동 진단

손 개수·핀치 비율·이벤트·NAV 진단 표시는 일반 UI에서 제거했습니다. 페이지 이동은 기존 제스처와 방향키로 조작합니다.

- 기본 캘리브레이션은 카메라 각 변의 3% 여백(가로·세로 94% 영역)입니다. PDF 모드 카메라 패널은 512×384이며 창은 1800×1000으로 시작합니다. 손 검출은 캘리브레이션 이전에 전체 카메라 영상에서 수행합니다.

### 박수와 페이지 이동 분리

양손이 보이거나 박수를 세는 중, 박수 전환 직후에는 스와이프를 차단하고 이동 이력을 비웁니다. 두 손을 벌렸다가 붙이는 동작을 두 번 하며 각 접촉 사이에는 손을 다시 벌려야 합니다. UI 기본 박수 간격은 최대 2.4초이며 접촉 직전의 짧은 가림은 허용합니다. 박수 직후 페이지를 넘길 때는 약 0.8초 뒤 한 손으로 움직이세요. 관련 값은 `config/default.json`의 `gestures`에서 조절합니다. 실제 카메라에서 추가 확인이 필요합니다.

- 박수의 페이지 차단과 필기 차단을 분리했습니다. 박수 대기/페이지 이동 잠금 중에도 한 손으로 돌아와 손가락을 벌린 후 새 핀치를 하면 필기를 재개합니다.
- `--trace-gestures`는 화면 진단 없이 `data/diagnostics/`에 숫자·판정 상태만 기록합니다. 카메라 이미지·음성은 저장하지 않으며 Git 제외 대상입니다. 실제 박수 실패나 필기 끊김 원인을 조사할 때 사용합니다.

- 실제 사용자 수치 로그에서 필기 중 경계 주먹 오인식으로 핀치가 잠기는 사례를 확인했습니다. 검지 접힘 기준과 4프레임 주먹 확인을 추가해 순간 오판을 줄였습니다. 박수 가림/재검출 조건을 보완했으며 실제 카메라 인식률은 추가 확인이 필요합니다. 좌표 튐 분석용 수치 로그에는 원본/보정 좌표를 함께 기록합니다.

- 필기 중 핀치를 유지하면 경계 주먹 판정이 지우개 전환을 덮어쓰지 않습니다. 지우려면 먼저 손가락을 벌려 pen-up한 뒤 주먹을 쥡니다. 한 손 스와이프 이동 기준은 화면 폭 14%, 재입력 대기는 0.45초입니다. 박수 인식/보호 설정은 유지했습니다.

- 회전 필기 중 한 프레임의 핀치 거리 추정 변화로 획이 끊기는 사례를 수치 로그에서 확인했습니다. UI 데모는 해제 비율 0.38을 2프레임 연속 확인하여 필기 연속성을 보완합니다. 손 소실·양손·영역 이탈 시의 안전한 획 종료는 유지합니다.

- 직선 필기의 잔떨림을 줄이도록 필기용 One Euro 설정과 원본 좌표 차분 기반 속도 추정을 조정했습니다. 중간 정도의 핀치 해제를 확인하는 동안은 마지막 잉크점을 유지하고, 명확히 벌린 손(비율 0.6 이상)은 즉시 획을 종료합니다. 확인 중 단발 오판은 획을 끊지 않습니다. 실제 글씨 모양·지연은 사용자 카메라에서 추가 평가가 필요합니다.

- 형광펜 전환의 짧은 보호 시간 중 한 손으로 돌아와 손가락을 벌린 기록을 유지합니다. 보호 시간이 끝나면 첫 핀치에서 필기를 시작하며, 손 소실·주먹·양손 상태는 기존 입력 차단을 유지합니다.

- UI 데모는 손 검출 시작 기준을 `tracking.detection_confidence=0.4`로 지정해 양손 재검출을 보완합니다(모듈 기본값은 0.5 유지). 박수 접촉 기준을 소폭 완화하되 벌림→접촉 두 번·가림 검증·페이지 차단을 유지합니다. 검출된 박수가 직전 필기 상태 또는 팔레트 단순 통과 때문에 버려지지 않도록 처리합니다. 실제 인식률 향상은 카메라에서 확인해야 합니다.

## 담당 범위 밖의 변경

이번 브랜치의 손 검출·제스처·보정·렌더러 변경은 [INTEGRATION_CHANGES.md](INTEGRATION_CHANGES.md)를 기준으로 검토하세요. 해당 문서는 기존 팀원 구현과 이번 통합 수정의 범위, 행동 변화, 기본값과 UI 프로필 차이를 구분합니다.
