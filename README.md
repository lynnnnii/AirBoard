# AirBoard

핀치 제스처 기반 PDF 허공 필기 팀 프로젝트.

## 좌표 변환 및 UI (임준서)

설치, 웹캠·마우스·가상 손 실행, 4점 캘리브레이션, JSON 설정, 팀 모듈 연결은
[COORDINATE_UI.md](COORDINATE_UI.md)를 참고하세요.

```powershell
.\.venv\Scripts\python.exe -m airboard.ui.demo
```

현재 `main.py`는 통합 전입니다. 위 명령은 독립 UI 데모를 실행하며,
PDF 렌더링·저장 기능은 팀 렌더러 연결 후 사용할 수 있습니다.
