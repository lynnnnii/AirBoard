# AirBoard

핀치 제스처 기반 PDF 허공 필기 팀 프로젝트.

## 좌표 변환·UI 및 PDF 통합 데모

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m airboard.ui.demo --pdf data/pdfs/ui-test/segmentation-p01-15.pdf
```

기본은 웹캠 입력입니다. `--mouse` 또는 `--virtual`로 카메라 없이 실행할 수 있습니다.
PDF 없이 실행하면 흰 배경 독립 데모입니다. PDF 모드에는 팀 보정기와 렌더러가 연결되어 있습니다.
기본 창은 1800×1000이며 `--fullscreen`으로 전체 화면 실행이 가능합니다.

- 핀치: 필기. 박수 두 번: 펜/형광펜. 필기를 끝낸 뒤 주먹: 지우개. 검지 포인터: 포인터.
- 손가락을 펼쳐 **위로 이동: 이전 페이지 / 아래로 이동: 다음 페이지**. `↑/↓` 방향키도 같은 동작이며 기존 `←/→` 비상키도 지원합니다.
- 지우개는 닿은 **획 전체**를 지웁니다. 겹친 곳에서는 위에 그린 획부터 하나씩 지우며, 멈춘 채 유지하면 다른 획까지 계속 지우지 않습니다. 주먹을 유지하고 이동하면 다른 획도 지울 수 있습니다.
- 펜: **파랑·빨강·검정, 1/2/3px (기본 2px)**. 형광펜: **노란색, 16px 고정**. 팔레트 색·굵기는 펜에 적용됩니다.
- 핀치를 벌리기 시작하면 획이 종료됩니다. 끝 꼬리 2개 입력점을 제거하며 짧은 점·획은 유지합니다. 형광펜 전환 후 포인터 자세에서 새 핀치를 하면 바로 쓸 수 있습니다.
- C: 캘리브레이션. 일반 화면 R: 기본 영역 복원. S: 제스처 설정.
- Z: 마지막 필기/지우기 실행 취소. X: 현재 페이지 필기 지우기. Q: 종료.
- PDF별 페이지와 필기는 로컬 자동 저장되어 재실행 시 복원됩니다. 원본 PDF는 수정하지 않습니다.

## macOS에서 실행

Python 3.10~3.12가 필요합니다. `mediapipe==0.10.21`이 3.13 이상을 지원하지 않으므로
Homebrew 기본 `python3`가 3.13 이상이면 버전을 지정해 가상 환경을 만드세요. Apple Silicon에서 확인했습니다.

```bash
python3.12 -m venv .venv    # 3.10~3.12 중 설치된 버전
.venv/bin/python -m pip install -r requirements-ui.txt
.venv/bin/python -m airboard.ui.demo --pdf data/pdfs/ui-test/segmentation-p01-15.pdf
```

- 위 Windows 명령의 `.\.venv\Scripts\python.exe`는 맥에서 `.venv/bin/python`입니다. 옵션은 동일합니다.
- 카메라 권한: 처음 실행하면 터미널(또는 IDE)의 카메라 접근 허용을 묻습니다.
  거부했다면 시스템 설정 → 개인정보 보호 및 보안 → 카메라에서 켭니다.
- 아이폰 연속성 카메라 등으로 카메라 번호가 밀리면 `--camera-index 1`을 지정합니다.
- 카메라 없이 확인: `--mouse`, `--virtual`, `--video 파일.mp4`.
- 현재 진입점은 `airboard.ui.demo`입니다. `main.py`는 아직 구현되지 않았습니다.

## 문서

- [좌표·UI 설치, 설정, 인터페이스](COORDINATE_UI.md)
- [다른 담당 모듈의 변경 이유·호환성·리뷰 포인트](INTEGRATION_CHANGES.md)
- [분할 PDF 테스트 자료와 확인 순서](data/pdfs/ui-test/README.md)

현재 `main.py` 최종 통합과 필기 합성 PDF 내보내기 UI는 별도 작업입니다.
실제 박수 인식률·필기 지연·FPS는 추가 평가가 필요합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

전체 자동 테스트 125개 통과. 개인 필기·캘리브레이션·제스처 설정과 수치 로그는 Git에서 제외합니다.
기존 개인 설정의 좌우 스와이프 매핑과 옛 팔레트·꼬리 제거 기본값은 자동 이관됩니다. 검지 쓰기, 두 손가락 멈춤, `[ / ]` 글씨 크기 조절은 계속 지원합니다.
