---
name: pdm-risk
description: 장비 정비 기록으로 '정비 뒤 30일 안 재고장' 확률을 계산해 재점검 우선순위 · 장비별 근거 · 재점검 계획 보고서를 만든다. 재점검, 정비 우선순위, 고장 위험, 어느 장비를 먼저 볼지, 특정 장비(EQ-0000)가 왜 위험한지 묻는 요청에 쓴다.
---

# pdm-risk — 재점검 우선순위 스킬

실습에서 학습한 모델(`artifacts/pdm_model.joblib`)로 정비 기록에 점수를 매긴다. **숫자는 반드시 스크립트 출력에서만 가져온다.** 확률 · 순위 · 대수를 짐작하거나 지어내지 않는다.

## 순서
1. `artifacts/pdm_model.joblib`과 `artifacts/model_card.json`이 있는지 확인한다. 없으면 멈추고 "블록 5의 `python scripts/train.py`를 먼저 실행하세요"라고 안내한다.
2. 요청에 맞게 스크립트를 실행한다 (pdm_ai_lab 폴더, 가상환경 켠 터미널).
   - 이번 달 상위 목록: `python .agents/skills/pdm-risk/scripts/score.py`
   - 대수를 바꿀 때: `--top 15` · 장비 유형만: `--type 발전기` · 다른 달 CSV: `--input data/파일.csv`
   - 장비 1대의 근거: `python .agents/skills/pdm-risk/scripts/score.py --explain EQ-0051`
3. 출력 표를 바탕으로 답한다.
   - 목록 요청: 표를 그대로 보여 주고, 상위 3대의 '위험을 올린 요인'을 한 줄씩 풀어 쓴다.
   - 근거 요청: 기여도가 큰 요인 3개를 '무엇이 · 얼마이고 · 그래서 위험이 올라감' 순서로 설명한다.
4. 보고서를 요청하면 `resources/report_template.md` 틀로 `reports/<YYYY-MM>_재점검계획.md`를 만든다.

## 지킬 것
- 모델 정보(`artifacts/model_card.json`)의 `limits`를 보고서 끝에 그대로 적는다. 최종 판단은 정비 담당자가 한다고 쓴다.
- 재점검 대수는 사용자가 말하지 않으면 `capacity_per_month`를 쓴다.
- 데이터는 교육용 가상 데이터다. 실제 부대 · 장비 정보와 섞지 않는다.
- 스크립트가 오류를 내면 오류 메시지를 그대로 보여 주고, 코드를 마음대로 고치지 않는다. 고칠 때는 사용자에게 먼저 묻는다.
