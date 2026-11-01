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
- 손가락을 펼쳐 좌우로 이동: 페이지 넘김. 방향키는 비상용 페이지 이동입니다.
- C: 캘리브레이션. 일반 화면 R: 기본 영역 복원. S: 제스처 설정.
- X: 현재 페이지 필기 지우기. Q: 종료.
- PDF별 페이지와 필기는 로컬 자동 저장되어 재실행 시 복원됩니다. 원본 PDF는 수정하지 않습니다.

## 문서

- [좌표·UI 설치, 설정, 인터페이스](COORDINATE_UI.md)
- [다른 담당 모듈의 변경 이유·호환성·리뷰 포인트](INTEGRATION_CHANGES.md)
- [분할 PDF 테스트 자료와 확인 순서](data/pdfs/ui-test/README.md)

현재 `main.py` 최종 통합과 필기 합성 PDF 내보내기 UI는 별도 작업입니다.
실제 박수 인식률·필기 지연·FPS는 추가 평가가 필요합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

전체 자동 테스트 88개 통과. 개인 필기·캘리브레이션·제스처 설정과 수치 로그는 Git에서 제외합니다.
