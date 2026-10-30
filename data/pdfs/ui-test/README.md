# UI/PDF 수동 테스트 자료

사용자가 제공한 `11. lecture-Segmentation.pdf`를 페이지 순서와 내용을 유지해 세 파일로 분할했습니다.
44쪽 전체가 중복·누락 없이 포함됩니다. 원본은 수정하지 않았습니다.

| 파일 | 원본 PDF 페이지 | 쪽수 |
| --- | --- | --- |
| segmentation-p01-15.pdf | 1–15 | 15 |
| segmentation-p16-30.pdf | 16–30 | 15 |
| segmentation-p31-44.pdf | 31–44 | 14 |

저장소 루트에서 실행:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt
.\.venv\Scripts\python.exe -m airboard.ui.demo --pdf data/pdfs/ui-test/segmentation-p01-15.pdf
```

다른 파일은 `--pdf` 경로를 바꿔 사용합니다. 전체 화면은 `--fullscreen`을 추가합니다.
화면의 페이지 번호는 각 분할 파일의 첫 쪽부터 1로 시작합니다.

## 확인 순서

1. PDF와 카메라가 보이는지 확인합니다.
2. 핀치 필기·색상·굵기 변경·주먹 지우개를 확인합니다.
3. 스와이프 또는 방향키로 실제 페이지를 이동하고, 돌아왔을 때 필기가 유지되는지 확인합니다.
4. C로 네 점을 지정하고 R로 기본 캘리브레이션을 복원합니다.
5. S에서 제스처 매핑 편집·Enter 저장·Esc 취소·R 기본 매핑 복원을 확인합니다.
6. 박수 두 번으로 펜/형광펜 전환을 확인합니다. 인식 안정성은 추가 튜닝 대상입니다.

PDF 지우개는 필기 레이어만 지웁니다. 필기는 종료 시 자동 저장되지 않습니다.
떨림 보정은 이 UI 데모에 아직 연결되지 않았습니다.
이 자료는 팀 테스트용 강의 자료이며 프로젝트가 제작한 문서가 아닙니다.
