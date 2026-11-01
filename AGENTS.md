# [AGENTS.md / CLAUDE.md](http://CLAUDE.md) 

이 파일은 AirBoard 저장소에서 작업하는 Claude Code(및 팀원)를 위한 안내서입니다. 코드를 수정하기 전에 끝까지 읽어 주세요.

## 프로젝트 개요

**AirBoard**: 엄지·검지 핀치 제스처로 PDF(PPT는 PDF로 변환) 슬라이드 위에 허공 필기하는 시스템. 가천대학교 「기초 컴퓨터비전」 팀 프로젝트 (팀 Maroon 5).

- 입력: 실시간 웹캠 영상(발표자를 향해 고정), 강의 자료 PDF, 테스트용 사전 녹화 영상
- 출력: 전체 화면 슬라이드 위의 실시간 필기, 도구·상태 커서, 도구 전환 알림, 필기가 합성된 PDF/이미지
- 최종 목표: 발표장에서 노트북 웹캠만으로 **라이브 데모**

## 반드시 지킬 제약

- **Python + OpenCV가 핵심.** 화면 표시는 OpenCV 창(`cv2.imshow`)으로 합니다. PowerPoint 창 위에 투명하게 덧그리는 등 OS 창 제어는 하지 않습니다.
- 사전학습 모델은 **MediaPipe Hands만** 사용합니다. 핀치·제스처 판정, 떨림 보정, 좌표 변환은 직접 구현하는 것이 프로젝트의 핵심이므로 다른 라이브러리로 대체하지 않습니다.
- 노트북 CPU에서 실시간(목표 25 FPS 이상)으로 돌아가야 합니다. GPU를 가정하지 않습니다.
- 라이브 데모 안전장치를 깨뜨리지 않습니다: 웹캠 대신 녹화 영상 입력, 키보드 페이지 넘기기는 항상 동작해야 합니다.

## 처리 파이프라인


| 단계          | 기법                                           | 모듈               |
| ----------- | -------------------------------------------- | ---------------- |
| 1. 입력       | `cv2.VideoCapture`, `cv2.flip`(좌우 반전)        | `input/`         |
| 2. 손 인식     | MediaPipe Hands, 랜드마크 21개, 양손 추적             | `tracking/`      |
| 3. 필기 상태 판정 | 핀치 거리 정규화 + 히스테리시스                           | `gestures/`      |
| 3-1. 도구 전환  | 박수 두 번, 주먹, 검지 포인터, 스와이프                     | `gestures/`      |
| 4. 좌표 변환    | 4점 캘리브레이션, `cv2.getPerspectiveTransform`     | `calibration/`   |
| 5. 떨림 보정    | EMA / One Euro 필터, 획 끝 꼬리 제거                 | `smoothing/`     |
| 6. 렌더링      | PyMuPDF(fitz), `cv2.line`, `cv2.addWeighted` | `render/`        |
| 7. 저장       | 필기 합성 PDF·이미지 저장, 획 단위 undo                  | `render/`, `io/` |


## 저장소 구조 (안)

```
airboard/
  main.py                  # 진입점 (인자 파싱, 메인 루프)
  config/
    default.json           # 임계값, 색상, 굵기 등 파라미터
    gestures.json          # 제스처 → 기능 매핑 (사용자 설정)
  airboard/
    input/                 # 웹캠 / 녹화 영상 입력 추상화
    tracking/              # MediaPipe 래퍼, 손 선택 로직
    gestures/              # 핀치, 주먹, 박수, 포인터, 스와이프 판정
    calibration/           # 가상 필기 영역 캘리브레이션, 호모그래피
    smoothing/             # EMA, One Euro, 꼬리 제거
    render/                # PDF 렌더링, 필기 레이어, 지우개, 저장
    ui/                    # 커서, 팔레트, 도구 알림, 설정 화면
    core/                  # 공용 데이터 타입, 상태 머신, 이벤트
  models/                  # MediaPipe 모델 파일 (hand_landmarker.task)
  eval/                    # 평가 스크립트, 라벨 파일
  data/                    # 녹화 영상·테스트 PDF (용량 큰 파일은 git에 올리지 않음)
  tests/

```

구조를 바꿀 때는 이 섹션도 함께 수정합니다.

## 모듈 간 인터페이스

모듈은 아래 데이터만 주고받습니다. 다른 모듈의 내부 상태를 직접 참조하지 않습니다. 형식을 바꾸면 이 섹션을 먼저 수정하고 팀에 공유합니다.

```python
# tracking → gestures
HandFrame(frame_idx: int, t_ms: float, hands: list[Hand])
Hand(handedness: str, landmarks: np.ndarray)  # shape (21, 3), 정규화 좌표, 좌우 반전 후

# gestures → 메인 루프
GestureState(tool: Tool, pen_down: bool, cursor: tuple[float, float] | None)
GestureEvent  # "toggle_highlighter", "page_next", "page_prev", "eraser_on", "eraser_off" 등

# calibration: 카메라 정규화 좌표 → 슬라이드 픽셀 좌표 (3x3 호모그래피)
# smoothing: 슬라이드 좌표 시퀀스 → 보정된 좌표

# render: 획 데이터
Stroke(stroke_id: int, page: int, tool: Tool, color: str, width: float,
       points: list[tuple[float, float]])  # 슬라이드 좌표계

```

`Tool`은 `pen`, `highlighter`, `eraser`, `pointer` 중 하나입니다.

## 제스처 판정 규칙 (기본 프리셋)


| 제스처           | 기능                 | 판정 기준                                                                                        |
| ------------- | ------------------ | -------------------------------------------------------------------------------------------- |
| 엄지·검지 붙이기/떼기  | 쓰기 / 펜 들기          | 비율 = dist(4, 8) / dist(0, 9). 0.25 미만이면 쓰기 시작, 0.35 초과면 멈춤 (히스테리시스)                          |
| 박수 두 번        | 펜 ↔ 형광펜            | 양 손바닥 중심 거리가 빠르게 0 근처로 줄었다 벌어지면 1회. 겹쳐서 한 손만 보이는 프레임도 접촉으로 처리. 약 1초 안에 2회일 때만 전환             |
| 주먹            | 지우개 (펴면 이전 도구로 복귀) | 중지·약지·새끼 끝(12, 16, 20)이 각 PIP 관절보다 손목 쪽이면 접힘. 네 손가락 모두 접혀야 주먹. **핀치보다 먼저 검사**, 몇 프레임 유지돼야 전환 |
| 검지만 펴기        | 레이저 포인터            | 필기 없이 커서만 표시                                                                                 |
| 좌우 스와이프       | 이전·다음 페이지          | 필기 중에는 비활성                                                                                   |
| 팔레트에 손가락 머무르기 | 색상·굵기 변경           | 화면 위쪽 팔레트 영역에 일정 시간 체류                                                                       |


- 모든 도구 전환은 **최소 유지 프레임**을 거친 뒤 적용해 순간 오판을 막습니다.
- 획 꼬리 제거: 펜을 내린 직후 몇 프레임의 좌표는 버리고(시작 꼬리), 펜을 든 순간 획의 마지막 몇 점을 거슬러 지웁니다(끝 꼬리). 끝 꼬리는 `pen_down`이 꺼지기 전에 이미 획에 들어가 있기 때문입니다.
- 임계값과 프레임 수는 코드에 하드코딩하지 말고 `config/default.json`에서 읽습니다.
- 제스처 인식과 기능 실행은 분리합니다. 매핑은 `config/gestures.json`에서만 정의합니다.
- 화면에 여러 손이 잡히면 캘리브레이션 영역 안에 있고 가장 큰 손을 우선 추적합니다.

## 실행 (명령은 구현에 맞춰 갱신)

```bash
pip install -r requirements.txt

python main.py --pdf data/pdfs/sample.pdf                     # 웹캠 입력
python main.py --pdf data/pdfs/sample.pdf --video data/recordings/s1.mp4   # 녹화 영상 입력
python main.py --pdf ... --no-smoothing --no-hysteresis        # 비교 시연용
python -m eval.run --labels eval/labels/                      # 평가
python -m airboard.smoothing.compare                          # 떨림 보정 방식 비교 화면 (웹캠)
python -m unittest discover tests                             # 단위 테스트

```

키보드 단축키(데모 비상용): `←/→` 페이지 이동, `c` 재캘리브레이션, `z` 실행 취소, `s` 저장, `q` 종료.

## 평가

결과는 숫자로 남깁니다. 평가 데이터(test)는 튜닝(dev)과 녹화 세션 단위로 분리합니다.

- 핀치 판정: 라벨 프레임 기준 precision / recall / F1, 히스테리시스 유무 비교
- 제스처 인식: 제스처별 인식률과 오작동 횟수 (특히 박수 오인식)
- 떨림 보정: 정해진 직선·원 따라 그리기의 평균 궤적 오차(px)와 지연(ms), 보정 방식별 비교
- 실시간성: FPS
- 거리별 성능: 카메라와 1m / 2m / 3m에서 손 검출률과 핀치 판정 정확도

## 코딩 규칙

- Python 3.10 이상, 타입 힌트 사용, 함수와 클래스에는 짧은 docstring.
- 판정 로직(gestures, smoothing, calibration)은 OpenCV 창이나 웹캠 없이 테스트할 수 있게 **순수 함수/클래스**로 작성합니다. 입력은 랜드마크 배열, 출력은 상태/이벤트.
- 판정 로직에는 `tests/`에 단위 테스트를 추가합니다. 실제 녹화에서 뽑은 랜드마크 시퀀스를 픽스처로 써도 됩니다.
- 매 프레임 실행되는 코드에서 무거운 연산(PDF 재렌더링 등)을 반복하지 않습니다. 페이지 이미지는 캐시합니다.
- 주석과 커밋 메시지는 한국어로 써도 됩니다.
- 녹화 영상, 대용량 PDF, 개인 얼굴이 나오는 데이터는 커밋하지 않습니다 (`.gitignore`의 `data/` 참고).

## 팀 역할


| 담당  | 영역                                                             |
| --- | -------------------------------------------------------------- |
| 김현보 | 손 인식 및 제스처 판정 (`tracking/`, `gestures/`)                       |
| 이수빈 | 떨림 보정 (`smoothing/`), 보정 전후 비교                                 |
| 이예린 | 슬라이드 렌더링 및 필기 관리 (`render/`)                                   |
| 임준서 | 좌표 변환 및 UI (`calibration/`, `ui/`, 사용자 설정)                     |
| 조항민 | 통합, 테스트, 데모·발표 ([`main.py`](http://main.py), `core/`, `eval/`) |


다른 사람 담당 모듈의 인터페이스를 바꿔야 하면 PR 설명에 이유를 적고 해당 담당자를 리뷰어로 지정합니다.
## PDF 작업 복원 API

`AirBoardRenderer.document_key`는 PDF 내용 해시입니다. `save_session(path)`와 `load_session(path)`는 페이지·편집 이력을 JSON으로 저장·복원하는 공개 API입니다. 기존 입력 인터페이스는 유지하며 UI PDF 어댑터가 호출합니다. `data/sessions/`는 개인 상태로 Git 제외 대상입니다.

## 좌표·UI 브랜치의 공통 모듈 변경

손 검출 선택 인자, 제스처 생성자의 config, 핀치 재시작/주먹 우선순위/박수 및 스와이프 동작, One Euro 수치 출력이 변경되었습니다. 기존 호출 형식은 유지하지만 행동·평가 결과가 완전히 같지는 않습니다. `INTEGRATION_CHANGES.md`의 담당별 설명과 리뷰 지점을 확인하세요. UI `InputState`에는 기본 True인 `navigation_allowed`가 추가되었으며, 실제 GestureResult 어댑터가 한 손/도구 상태에 맞춰 설정합니다.
