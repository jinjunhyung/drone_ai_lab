---
name: pdm-train
description: 장비 정비 기록으로 '정비 뒤 30일 안 재고장' 예측 모델을 학습 · 평가하고 artifacts/에 저장한다. 모델 학습, 다시 학습, 성능 평가, 되돌려 보기(백테스트), 위험 표시 기준(임계값)을 정해 달라는 요청에 쓴다.
---

# pdm-train — 재고장 예측 모델 학습 · 평가 절차

`scripts/train.py`를 이 절차대로 만들고 실행해 `artifacts/`에 모델 · 모델 정보 · 되돌려 보기 결과를 저장한다.

## 순서
1. `pdm_lib`의 `load_history()` → `clean()` → `add_time_features()`로 데이터를 준비한다. 피처는 `FEATURES`, 타깃은 `TARGET`을 쓴다.
2. `maintenance_date`로 기간을 나눈다. 학습: 2024-06-30까지, 검증: 2024-07-01 ~ 2024-12-31, 시험: 2025-01-01부터. 무작위로 나누지 않는다.
3. 위험 표시 기준: 학습 기간으로 `make_model()`을 학습해 검증 기간 확률을 구한다. 기준 0.05 ~ 0.94(0.01 간격) 가운데 `놓친 재고장 수 × 300 + 기준 이상 건수 × 60`(만원)이 가장 작은 값을 고른다.
4. 2025 되돌려 보기: 2024-12-31까지로 새 모델을 학습해 시험 기간에 확률(`prob` 열)을 매긴다. `monthly_topk(test, "prob", 20)`의 `hits`를 `model` 열로, `monthly_topk(test, "equipment_age_years", 20)`의 `hits`를 `oldest` 열로, 달마다 `min(20, 그 달 건수) × 그 달 재고장 비율`(소수 첫째 자리)을 `random` 열로 하는 표를 만든다.
5. 이번 달 예측용 모델은 전체 기간(2023 ~ 2025)으로 다시 학습한다.
6. 저장: 모델은 `joblib.dump(모델, MODEL_PATH)`, 되돌려 보기 표는 `artifacts/backtest_2025.csv`(utf-8-sig), 모델 정보는 `make_card(df, 기준, test, 표)`를 `CARD_PATH`에 JSON(utf-8, `ensure_ascii=False`)으로 쓴다.
7. `python scripts/train.py` 실행 뒤 `python scripts/check.py`를 실행해 결과를 그대로 보여 준다.

## 지킬 것
- `availability_rate` · `equipment_id` · `unit_code` · `maintenance_date` · `fault_code`는 모델에 넣지 않는다.
- 성능을 말할 때는 PR-AUC와 월 20대 적중 수를 기준선(무작위 · 오래된 순)과 함께 적는다.
- 기준(임계값)을 시험 기간(2025)으로 고르지 않는다.
- 스크립트가 오류를 내면 오류 메시지를 그대로 보여 주고, `pdm_lib.py`의 함수 이름 · 상수를 바꾸지 않는다.
