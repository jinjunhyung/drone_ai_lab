"""체크포인트: 노트북 실습 4 ~ 7과 같은 방법으로 모델을 학습해 artifacts/에 저장한다.

노트북을 끝까지 못 했어도 이 스크립트를 실행하면 앱(app.py)과 에이전트 스킬을 쓸 수 있다.
pdm_lib.py의 clean · add_time_features(실습 3)는 먼저 채워 둬야 한다.
실행 (pdm_lab 폴더에서):  python scripts/train.py
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdm_lib import (ART, CARD_PATH, FEATURES, MODEL_PATH, TARGET, add_time_features, clean,  # noqa: E402
                     load_history, make_card, make_model, monthly_topk)

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):            # Windows에서 결과를 다른 프로그램(에이전트)으로 넘길 때 한글이 깨지지 않게
    sys.stdout.reconfigure(encoding="utf-8")
CAPACITY = 20                 # 한 달에 재점검할 수 있는 장비 수
COST_MISS, COST_CHECK = 300, 60   # 놓친 재고장 1건 · 재점검 1건 비용 (만원, 교육용 가정)


def pick_threshold(y, prob):
    """검증 기간에서 '놓친 재고장 비용 + 재점검 비용'이 가장 작은 임계값."""
    ths = np.round(np.arange(0.05, 0.95, 0.01), 2)
    cost = [((prob < t) & (y == 1)).sum() * COST_MISS + (prob >= t).sum() * COST_CHECK for t in ths]
    return float(ths[int(np.argmin(cost))])


def main():
    df = add_time_features(clean(load_history()))
    d = df["maintenance_date"]
    train, valid, test = df[d < "2024-07-01"], df[(d >= "2024-07-01") & (d < "2025-01-01")], df[d >= "2025-01-01"]

    m = make_model().fit(train[FEATURES], train[TARGET])                       # 임계값 고르기용
    thr = pick_threshold(valid[TARGET].to_numpy(), m.predict_proba(valid[FEATURES])[:, 1])

    m = make_model().fit(df[d < "2025-01-01"][FEATURES], df[d < "2025-01-01"][TARGET])   # 2025 시험용
    test = test.assign(prob=m.predict_proba(test[FEATURES])[:, 1])
    back = monthly_topk(test, "prob", CAPACITY).rename(columns={"hits": "model"})
    back["oldest"] = monthly_topk(test, "equipment_age_years", CAPACITY)["hits"]
    back["random"] = [round(min(CAPACITY, len(g)) * g[TARGET].mean(), 1)
                      for _, g in test.groupby(test["maintenance_date"].dt.to_period("M"))]

    final = make_model().fit(df[FEATURES], df[TARGET])                          # 이번 달 예측용 (전체 기간)
    ART.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    back.to_csv(ART / "backtest_2025.csv", index=False, encoding="utf-8-sig")
    card = make_card(df, thr, test, back, CAPACITY, (COST_MISS, COST_CHECK))
    CARD_PATH.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {MODEL_PATH.name}, {CARD_PATH.name}, backtest_2025.csv")
    print(f"임계값 {thr} · 2025 PR-AUC {card['test_2025']['pr_auc']} · 월 {CAPACITY}대 적중 "
          f"모델 {back['model'].sum()} / 오래된 순 {back['oldest'].sum()} / 무작위 {back['random'].sum():.0f}")


if __name__ == "__main__":
    main()
