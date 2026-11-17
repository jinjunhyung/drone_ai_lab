"""예측정비 실습 공통 함수 — 노트북 · 앱(app.py) · 에이전트 스킬(score.py)이 함께 쓴다.

✏️ 표시가 있는 두 함수(clean, add_time_features)는 실습 3에서 직접 채운다.
나머지는 그대로 쓴다. 이 파일을 고치면 노트북 · 앱 · 스킬 결과가 함께 바뀐다.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
HISTORY_CSV = DATA / "D2_pdm_history.csv"
MONTH_CSV = DATA / "D2_pdm_this_month.csv"
MODEL_PATH = ART / "pdm_model.joblib"
CARD_PATH = ART / "model_card.json"

TARGET = "failure_within_30days"
FAIL_TYPES = ["사후정비", "긴급정비"]                 # 고장 때문에 한 정비 (예방정비가 아닌 것)
CAT_COLS = ["equipment_type", "maintenance_type", "fault_category"]
BASE_NUM = ["equipment_age_years", "severity", "downtime_hours", "repair_duration_hours",
            "parts_replaced_count", "maintenance_cost", "maintenance_success",
            "cumulative_operating_hours", "days_since_last_maintenance", "is_first_maintenance"]
TIME_COLS = ["fail_count_90d", "days_since_last_failure", "hours_since_last_pm"]
LEAK_COLS = ["availability_rate"]                     # 그 달 전체 가동률 — 예측 시점에는 모른다
FEATURES = CAT_COLS + BASE_NUM + TIME_COLS

FEATURE_KO = {
    "equipment_type": "장비 유형", "maintenance_type": "정비 구분", "fault_category": "고장 유형",
    "equipment_age_years": "경과연수", "severity": "고장 심각도", "downtime_hours": "가동중단 시간",
    "repair_duration_hours": "수리 시간", "parts_replaced_count": "교체 부품 수", "maintenance_cost": "정비 비용",
    "maintenance_success": "정비 성공 여부", "cumulative_operating_hours": "누적 가동시간",
    "days_since_last_maintenance": "직전 정비 뒤 경과일", "is_first_maintenance": "첫 정비 여부",
    "fail_count_90d": "최근 90일 고장 수", "days_since_last_failure": "직전 고장 뒤 경과일",
    "hours_since_last_pm": "예방정비 뒤 가동시간", "availability_rate": "그 달 가동률",
}


def load_history(path=HISTORY_CSV):
    """정비 기록 CSV를 읽어 날짜를 날짜형으로 바꾸고 장비 · 날짜 순으로 정렬한다."""
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["maintenance_date"])
    return df.sort_values(["equipment_id", "maintenance_date"]).reset_index(drop=True)


def clean(df):
    """이상치 · 결측을 처리한다 (실습 3-1 · 3-2)."""
    df = df.copy()
    df.loc[df["downtime_hours"] >= 999, "downtime_hours"] = np.nan      # 999 = 미입력 코드
    typo = df["maintenance_cost"] > 10000                                 # 자릿수 입력 실수 (정상 최대 약 1,800만원)
    df.loc[typo, "maintenance_cost"] = df.loc[typo, "maintenance_cost"] / 100
    df["fault_category"] = df["fault_category"].fillna("N/A")            # 예방정비 = 고장 없음
    df["fault_code"] = df["fault_code"].fillna("N/A")
    df["is_first_maintenance"] = df["days_since_last_maintenance"].isna().astype(int)
    return df


def add_time_features(df):
    """장비마다 지난 기록을 보고 시간 피처 3개를 만든다 (실습 3-3). 이번 정비까지 포함해서 센다.
    fail_count_90d: 최근 90일 고장 정비 수 · days_since_last_failure: 직전 고장 정비 뒤 경과일
    hours_since_last_pm: 마지막 예방정비 뒤 가동시간"""
    df = df.sort_values(["equipment_id", "maintenance_date"]).copy()
    n90, since_fail, since_pm = {}, {}, {}
    for _, g in df.groupby("equipment_id", sort=False):
        fails = g[g["maintenance_type"].isin(FAIL_TYPES)]
        pms = g[g["maintenance_type"] == "예방정비"]
        for i, r in g.iterrows():
            d = r["maintenance_date"]
            past = fails[fails["maintenance_date"] <= d]["maintenance_date"]
            n90[i] = int((past >= d - pd.Timedelta(days=90)).sum())
            since_fail[i] = (d - past.max()).days if len(past) else np.nan
            pm_past = pms[pms["maintenance_date"] <= d]
            since_pm[i] = (r["cumulative_operating_hours"] - pm_past["cumulative_operating_hours"].iloc[-1]
                           if len(pm_past) else np.nan)
    df["fail_count_90d"] = pd.Series(n90)
    df["days_since_last_failure"] = pd.Series(since_fail)
    df["hours_since_last_pm"] = pd.Series(since_pm).round(1)
    return df


def make_model(cat_cols=CAT_COLS):
    """범주형은 원-핫, 수치형은 그대로 LightGBM에 넣는 파이프라인 (결측은 LightGBM이 처리)."""
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
                            remainder="passthrough", verbose_feature_names_out=False).set_output(transform="pandas")
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=4, min_child_samples=60, reg_lambda=5.0,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
    return Pipeline([("pre", pre), ("model", lgbm)])


def monthly_topk(df, score_col, k=20):
    """달마다 score_col이 높은 k건을 골랐을 때 실제 재고장 수. 열: month, picked, hits."""
    rows = []
    for m, g in df.groupby(df["maintenance_date"].dt.to_period("M")):
        top = g.nlargest(k, score_col)
        rows.append({"month": str(m), "picked": len(top), "hits": int(top[TARGET].sum())})
    return pd.DataFrame(rows)


def make_card(df, threshold, test, back, capacity=20, cost=(300, 60)):
    """앱 · 스킬이 읽는 모델 정보. df=학습 전체, test=prob 열이 있는 2025 시험 기간, back=monthly 비교표."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    d = df["maintenance_date"]
    return {
        "model": "LightGBM (pdm_lib.make_model)",
        "target": "정비 뒤 30일 안 재고장(사후 · 긴급정비) 여부",
        "trained_on": f"{d.min():%Y-%m-%d} ~ {d.max():%Y-%m-%d} ({len(df):,}건)",
        "features": FEATURES,
        "excluded": {"availability_rate": "그 달 전체 가동률 — 예측 시점에는 알 수 없음(누수)",
                     "fault_code": "50종 — fault_category로 대신함", "equipment_id · unit_code": "식별자"},
        "capacity_per_month": capacity,
        "threshold": float(threshold),
        "threshold_rule": f"검증 기간(2024-07 ~ 12) 비용 최소: 놓친 재고장 {cost[0]}만원, 재점검 {cost[1]}만원",
        "test_2025": {
            "pr_auc": round(float(average_precision_score(test[TARGET], test["prob"])), 3),
            "roc_auc": round(float(roc_auc_score(test[TARGET], test["prob"])), 3),
            "base_rate": round(float(test[TARGET].mean()), 3),
            "top20_hits": {k: float(back[k].sum()) for k in ["model", "oldest", "random"]},
        },
        "limits": ["교육용 가상 데이터로 학습했다. 실제 장비에 그대로 쓰지 않는다.",
                   "확률은 '정비 뒤 30일' 기준이다. 정비 기록이 없는 장비는 판단하지 않는다.",
                   "최종 판단은 정비 담당자가 한다. 모델은 점검 순서를 정하는 참고 자료다."],
    }


def contributions(model, X):
    """행마다 피처별 기여도(SHAP 값, 로그 오즈 단위). 원-핫으로 나뉜 열은 원래 컬럼으로 합친다."""
    Xt = model["pre"].transform(X)
    raw = model["model"].booster_.predict(Xt, pred_contrib=True)[:, :-1]     # 마지막 열 = 기준값
    contrib = pd.DataFrame(raw, columns=Xt.columns, index=X.index)
    groups = {c: next((k for k in CAT_COLS if c.startswith(k + "_")), c) for c in Xt.columns}
    return contrib.T.groupby(groups).sum().T[X.columns]


def reasons(model, X, top=3):
    """행마다 위험을 가장 많이 올린 피처 top개를 '이름(값)' 글로 돌려준다."""
    c = contributions(model, X)
    out = []
    for i in X.index:
        up = c.loc[i][c.loc[i] > 0].sort_values(ascending=False).head(top)
        out.append(", ".join(f"{FEATURE_KO.get(f, f)}({_fmt(X.at[i, f])})" for f in up.index))
    return out


def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return "없음" if np.isnan(v) else f"{v:,.1f}".rstrip("0").rstrip(".")
    return str(v)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("artifacts/pdm_model.joblib 이 없습니다. 블록 5를 끝내거나("
                                "python scripts/train.py) python scripts/catchup.py 5 를 실행하세요.")
    return joblib.load(MODEL_PATH), json.loads(CARD_PATH.read_text(encoding="utf-8"))


def score_month(model, history, month):
    """이번 달 정비 기록에 재고장 확률을 매긴다. 시간 피처는 지난 기록과 이어서 계산한다.
    한 장비가 이번 달에 여러 번 정비했으면 가장 최근 기록 하나로 판단한다."""
    month = month.copy()
    month["maintenance_date"] = pd.to_datetime(month["maintenance_date"])
    month["_new"] = 1
    both = pd.concat([history.assign(_new=0), month], ignore_index=True)
    both = add_time_features(clean(both))
    new = both[both["_new"] == 1].drop(columns=["_new", TARGET], errors="ignore")
    new = new.sort_values("maintenance_date").drop_duplicates("equipment_id", keep="last")
    new["prob"] = model.predict_proba(new[FEATURES])[:, 1]
    new["reasons"] = reasons(model, new[FEATURES])
    return new.sort_values("prob", ascending=False).reset_index(drop=True)
