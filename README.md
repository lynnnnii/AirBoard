# AirBoard

핀치 제스처 기반 PDF 허공 필기 팀 프로젝트.

## 좌표 변환 및 UI (임준서)

설치, 웹캠·마우스·가상 손 실행, 4점 캘리브레이션, JSON 설정, 팀 모듈 연결은
[COORDINATE_UI.md](COORDINATE_UI.md)를 참고하세요.

```powershell
.\.venv\Scripts\python.exe -m airboard.ui.demo
```

`--pdf "PDF 경로"`를 추가하면 팀 렌더러로 실제 PDF를 표시하고 페이지별로 필기합니다.
창은 기본 1600×900이며 `--fullscreen`으로 전체 화면 실행이 가능합니다.
현재 `main.py`는 통합 전입니다. UI 데모에는 떨림 보정·PDF 저장 UI가 아직 연결되지 않았습니다.
