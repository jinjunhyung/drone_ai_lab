"""pdm-train 스킬 절차에 따른 드론 재고장 예측 모델 학습 · 평가 · 2025 시험 예측 스크립트."""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pdm_lib import (  # noqa: E402
    ART,
    CARD_PATH,
    FEATURES,
    HISTORY_CSV,
    MODEL_PATH,
    TARGET,
    TEST_CSV,
    add_time_features,
    clean,
    load_history,
    make_card,
    make_model,
    monthly_topk,
)

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CAPACITY = 20
COST_MISS, COST_CHECK = 300, 60


def pick_threshold(y, prob):
    """검증 기간에서 '놓친 재고장 수 × 300 + 기준 이상 건수 × 60'(만원)이 가장 작은 값을 고른다."""
    ths = np.round(np.arange(0.05, 0.95, 0.01), 2)
    cost = [((prob < t) & (y == 1)).sum() * COST_MISS + (prob >= t).sum() * COST_CHECK for t in ths]
    return float(ths[int(np.argmin(cost))])


def predict_test_2025(final_model):
    """앞 기간(drone_train.csv)과 이어서 시간 피처를 계산한 뒤 drone_test.csv(1,241건)에 확률을 매긴다."""
    raw_tr = pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
    raw_te = pd.read_csv(TEST_CSV, encoding="utf-8-sig")
    raw_te["_test_order"] = np.arange(len(raw_te))
    raw_tr["_is_test"] = 0
    raw_te["_is_test"] = 1

    comb = pd.concat([raw_tr, raw_te], ignore_index=True)
    comb["maintenance_date"] = pd.to_datetime(comb["maintenance_date"])
    comb = comb.sort_values(["equipment_id", "maintenance_date"]).reset_index(drop=True)
    comb = add_time_features(clean(comb))

    te_feat = comb[comb["_is_test"] == 1].sort_values("_test_order").reset_index(drop=True)
    prob = final_model.predict_proba(te_feat[FEATURES])[:, 1]
    return pd.DataFrame(
        {
            "equipment_id": te_feat["equipment_id"],
            "maintenance_date": te_feat["maintenance_date"].dt.strftime("%Y-%m-%d"),
            "prob": np.round(prob, 4),
        }
    )


def main():
    # 1. 데이터 준비 (drone_train.csv, 2023 ~ 2024)
    df = add_time_features(clean(load_history()))
    d = df["maintenance_date"]

    # 2. 기간 분할 (무작위로 나누지 않음)
    train = df[d <= "2024-06-30"]
    valid = df[(d >= "2024-07-01") & (d <= "2024-12-31")].copy()

    # 3. 위험 표시 기준(임계값) 선정 및 검증 기간 예측
    m_val = make_model().fit(train[FEATURES], train[TARGET])
    val_prob = m_val.predict_proba(valid[FEATURES])[:, 1]
    valid = valid.assign(prob=val_prob)
    thr = pick_threshold(valid[TARGET].to_numpy(), val_prob)

    # 4. 검증 기간(2024-07 ~ 12) 되돌려 보기(백테스트)
    back = monthly_topk(valid, "prob", CAPACITY).rename(columns={"hits": "model"})
    back["oldest"] = monthly_topk(valid, "equipment_age_years", CAPACITY)["hits"]
    back["random"] = [
        round(min(CAPACITY, len(g)) * g[TARGET].mean(), 1)
        for _, g in valid.groupby(valid["maintenance_date"].dt.to_period("M"))
    ]

    # 5. 전체 학습 기간(2023 ~ 2024)으로 최종 모델 재학습
    final = make_model().fit(df[FEATURES], df[TARGET])

    # 6. 2025 시험 기간(drone_test.csv, 1,241건) 예측 생성
    test_pred = predict_test_2025(final)

    # 7. 결과 저장
    ART.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    back.to_csv(ART / "backtest_2025.csv", index=False, encoding="utf-8-sig")
    test_pred.to_csv(ART / "test_predictions_2025.csv", index=False, encoding="utf-8-sig")
    card = make_card(df, thr, valid, back, CAPACITY, (COST_MISS, COST_CHECK))
    CARD_PATH.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"저장 완료: {MODEL_PATH.name}, {CARD_PATH.name}, backtest_2025.csv, test_predictions_2025.csv ({len(test_pred)}건)")
    print(
        f"임계값 {thr} · 검증 PR-AUC {card['test_2025']['pr_auc']} (기준선 재고장 비율 {card['test_2025']['base_rate']}) · "
        f"월 {CAPACITY}대 적중(6개월 합): 모델 {back['model'].sum()}대 / 오래된 순 {back['oldest'].sum()}대 / 무작위 {back['random'].sum():.1f}대"
    )


if __name__ == "__main__":
    main()
