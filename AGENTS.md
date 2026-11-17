# AGENTS.md — 드론 예측정비 실습 (drone_ai_lab)

드론 정비 기록으로 '정비 뒤 30일 안 재고장'을 예측하고, 이번 달 재점검할 드론 20대를 고르는 프로젝트다.
데이터는 **교육용 가상 데이터**다.

## 폴더
- `data/` — 학습 기록 `drone_train.csv`(2023 ~ 2024), 시험 기록 `drone_test.csv`(2025, 결과 없음), 이번 달 기록 `drone_this_month.csv`(2026-01, 결과 없음)
- `pdm_lib.py` — 공통 함수. 분석 스크립트 · 앱 · 스킬이 함께 쓴다
- `scripts/` — 학습 스크립트 `train.py`, 진행 확인 `check.py`
- `app.py` — 정비 우선순위 앱 (Streamlit)
- `artifacts/` — 학습한 모델 `pdm_model.joblib`, 모델 정보 `model_card.json`, 검증 되돌려 보기 `backtest_2025.csv`, 시험 기간 예측 `test_predictions_2025.csv`
- `reports/` — 분석 결과 · 보고서 · 검증 기록을 쓰는 곳
- `.agents/skills/` — 스킬

## 명령 (Windows PowerShell, drone_ai_lab 폴더, 가상환경 켠 상태)
- 진행 확인: `python scripts/check.py`
- 모델 학습: `python scripts/train.py`
- 앱 실행: `streamlit run app.py`
- 점수 매기기: `python .agents/skills/pdm-risk/scripts/score.py`

## 고정 약속 (바꾸지 않는다)
- `pdm_lib.py`의 함수 이름 · 인자 · 상수(`FEATURES`, `TARGET` 등)와 위 파일 이름 · 위치는 바꾸지 않는다. 앱과 스킬, 진행 확인이 이 이름을 쓴다.
- 모델은 `pdm_lib.make_model()`로 만들고, 모델 정보는 `pdm_lib.make_card()`로 만든다.

## 규칙
<!-- 실습에서 직접 씁니다. 블록 1: 기본 규칙 · 블록 3: 데이터 처리 · 블록 5: 분석 · 검증 · 블록 8: 보고 -->

### 기본
- 답은 한국어, 업무 보고서 문체로 쓴다.
- `data/` 원본 CSV는 고치지 않는다. 새 파일은 `reports/` 또는 `artifacts/`에 만든다.
- 한 번에 결과물 하나만 만든다. 요청하지 않은 그래프 · 파일을 만들지 않는다.
- 코드를 고칠 때는 바꿀 파일과 이유를 먼저 말한다.
- API 키 · 실제 부대 정보 · 개인정보를 파일에 쓰지 않는다.

### 데이터 처리
- `downtime_hours`의 999는 미입력 코드다. 행을 지우지 않고 결측(`np.nan`)으로 바꾼다.
- `maintenance_cost`가 10000을 넘으면 자릿수 입력 실수다. 100으로 나눈다.
- `fault_code` · `fault_category`의 결측은 예방정비(고장 없음)다. 최빈값으로 채우지 않고 `"N/A"`로 채운다.
- `days_since_last_maintenance`의 결측은 드론의 첫 정비다. 비워 두고 `is_first_maintenance`(0/1) 열을 만든다.
- 시간 피처는 그 정비일까지의 기록만 쓴다(같은 날짜 포함, 뒤 날짜는 쓰지 않는다). 시험(2025) · 이번 달(2026-01) 데이터의 시간 피처는 앞 기간 기록과 이어서 계산한다.

### 분석 · 검증
- 예측 시점은 '정비를 마친 직후'다. 그때 알 수 없는 값은 모델에 넣지 않는다. `availability_rate`(그 달 전체 가동률)는 넣지 않는다.
- `equipment_id` · `unit_code` · `maintenance_date` · `fault_code`는 모델에 넣지 않는다. 피처는 `pdm_lib.FEATURES`만 쓴다.
- 데이터는 무작위로 나누지 않고 기간으로 나눈다: 학습 ~ 2024-06-30, 검증 2024-07-01 ~ 2024-12-31, 시험(`drone_test.csv`, 정답 없음) 2025-01-01 ~ 2025-12-31.
- 성능은 검증 기간(2024-07 ~ 12)의 PR-AUC와 '월 20대 적중 수'로 보고, 반드시 기준선(무작위 · 오래된 순)과 함께 적는다. 정확도만 보고하지 않는다.
- 위험 표시 기준(임계값)은 검증 기간에서 고른다. 시험 기간(2025)으로 고르지 않는다.
- 최종 모델(2023 ~ 2024 전체 학습)로 `drone_test.csv` 1,241건 모두에 재고장 확률을 매겨 `artifacts/test_predictions_2025.csv`(`equipment_id, maintenance_date, prob`)로 저장한다.
- 코드를 고친 뒤에는 `python scripts/check.py`를 실행하고 결과를 그대로 보여 준다.

### 보고
- 장비 위험도 · 순위 · 확률은 **pdm-risk 스킬 스크립트 출력만** 인용한다. 계산하지 않은 숫자를 쓰지 않는다.
- 숫자마다 어느 파일 · 어느 명령에서 나왔는지 적는다.
- 보고서 끝에 모델 정보(`artifacts/model_card.json`)의 `limits`를 그대로 적는다.