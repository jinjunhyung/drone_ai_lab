# pdm_ai_lab — 예측정비 실습 묶음 (AGENTS.md · 스킬로 AI와 함께 분석)

장비 정비 기록으로 '정비 뒤 30일 안 재고장'을 예측하고 이번 달 재점검할 장비를 고른다. 코드는 AI 에이전트가 쓰고, 사람은 규칙(`AGENTS.md`)과 스킬을 만들고 결과를 검증한다. 데이터는 교육용 가상 데이터다.

## 시작
```powershell
cd C:\workspaces\pdm_ai_lab
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts/check.py
```
VS Code에서 **File ▸ Open Folder** → 이 폴더를 연다. 순서는 실습지(사이트 '예측정비 실습 — AI와 함께')를 본다.

## 규칙 · 스킬 파일
- `AGENTS.md` — 규칙의 원본. 여기에만 쓴다. `CLAUDE.md` · `GEMINI.md`는 이 파일을 불러오는 한 줄이다.
- `.agents/skills/` — 스킬. 고친 뒤 `python scripts/sync_skills.py`를 실행하면 `.claude/skills/` · `.github/skills/`에도 복사된다.

## 진행 확인 · 따라잡기
- `python scripts/check.py` — 블록 1 ~ 8 가운데 어디까지 했는지, 결과가 맞는지 본다(아무것도 고치지 않는다).
- `python scripts/catchup.py 5` — 블록 5까지 비어 있는 규칙 · 코드 · 스킬을 기준 답으로 채운다(3 ~ 8).
