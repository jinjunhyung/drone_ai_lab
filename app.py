"""정비 우선순위 앱 — 이번 달 정비 기록에서 재점검할 장비를 고른다.

실행 (pdm_ai_lab 폴더, 가상환경 켠 상태):  streamlit run app.py
먼저 블록 5(python scripts/train.py)로 artifacts/ 에 모델을 저장해 둔다.
"""
import json
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

from pdm_lib import (ART, FEATURE_KO, FEATURES, MONTH_CSV, TARGET, contributions, load_history, load_model,
                     score_month)

st.set_page_config(page_title="정비 우선순위", page_icon="🛠️", layout="wide")
st.title("🛠️ 정비 우선순위 — 이번 달 재점검 목록")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_history():
    return load_history()


@st.cache_data
def score(month_csv):
    """이번 달 기록에 확률을 매기고 높은 순으로 정렬한다."""
    model, _ = get_model()
    month = pd.read_csv(month_csv, encoding="utf-8-sig")
    ranked = score_month(model, get_history(), month)
    ranked["확률(%)"] = (ranked["prob"] * 100).round(1)
    ranked = ranked.sort_values("prob", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    return ranked


try:
    model, card = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

with st.sidebar:
    st.header("설정")
    up = st.file_uploader("이번 달 정비 기록 CSV", type=["csv"])
    cap = st.slider("이번 달 재점검 가능 대수", 5, 40, card["capacity_per_month"])
    thr = st.slider("위험 표시 기준(확률)", 0.05, 0.95, float(card["threshold"]), 0.01)
    st.caption(f"모델: {card['model']} · 학습 {card['trained_on']}")

ranked = score(up if up is not None else MONTH_CSV)
ranked["위험"] = ranked["prob"].ge(thr).map({True: "높음", False: "-"})
ranked["재점검"] = ranked.index <= cap

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "① 재점검 목록", "② 장비별 근거", "③ 만약에", "④ 되돌려 보기(2025)", "⑤ 모델 정보", "⑥ AI 정비 소견 (Ollama)"
])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("이번 달 정비 장비", f"{len(ranked)}대")
    c2.metric("위험 높음 (기준 이상)", f"{int(ranked['prob'].ge(thr).sum())}대")
    c3.metric("재점검 대상", f"{min(cap, len(ranked))}대")
    show = ranked[["재점검", "equipment_id", "equipment_type", "unit_code", "maintenance_date", "maintenance_type",
                   "확률(%)", "위험", "reasons"]].rename(columns={
        "equipment_id": "장비", "equipment_type": "유형", "unit_code": "부대", "maintenance_date": "최근 정비일",
        "maintenance_type": "정비 구분", "reasons": "위험을 올린 요인"})
    show["최근 정비일"] = show["최근 정비일"].dt.strftime("%Y-%m-%d")
    st.dataframe(show, width="stretch", height=480)
    st.download_button("재점검 목록 받기 (CSV)", show[show["재점검"]].to_csv(index=False).encode("utf-8-sig"),
                       file_name="재점검_목록.csv", mime="text/csv")

with tab2:
    eid = st.selectbox("장비", ranked["equipment_id"], format_func=lambda e: f"{e} · "
                       f"{ranked.loc[ranked['equipment_id'] == e, '확률(%)'].iloc[0]}%")
    row = ranked[ranked["equipment_id"] == eid]
    c = contributions(model, row[FEATURES]).iloc[0].sort_values()
    bar = pd.DataFrame({"요인": [FEATURE_KO.get(f, f) for f in c.index], "기여도": c.to_numpy()})
    st.write(f"**{eid}** 재고장 확률 **{row['확률(%)'].iloc[0]}%** — 오른쪽(+)은 위험을 올린 요인, 왼쪽(−)은 낮춘 요인")
    st.bar_chart(bar, x="요인", y="기여도", horizontal=True, sort=False)
    vals = row[FEATURES].iloc[0].astype(str)
    st.dataframe(pd.DataFrame({"항목": [FEATURE_KO.get(f, f) for f in FEATURES], "값": vals.to_numpy()}),
                 width="stretch", hide_index=True)

with tab3:
    eid3 = st.selectbox("장비 ", ranked["equipment_id"], key="whatif")
    base = ranked[ranked["equipment_id"] == eid3][FEATURES].copy()
    a, b, c_, d = st.columns(4)
    new = base.copy()
    new["fail_count_90d"] = a.number_input("최근 90일 고장 수", 0, 20, int(base["fail_count_90d"].iloc[0]))
    hpm = base["hours_since_last_pm"].iloc[0]
    new["hours_since_last_pm"] = b.number_input("예방정비 뒤 가동시간", 0.0, 5000.0,
                                                0.0 if pd.isna(hpm) else float(hpm), 50.0)
    new["maintenance_success"] = c_.selectbox("정비 성공 여부", [1, 0], index=[1, 0].index(int(base["maintenance_success"].iloc[0])))
    new["equipment_age_years"] = d.number_input("경과연수", 0.0, 40.0, float(base["equipment_age_years"].iloc[0]), 0.5)
    p0, p1 = model.predict_proba(base)[0, 1], model.predict_proba(new)[0, 1]
    st.metric("재고장 확률", f"{p1 * 100:.1f}%", f"{(p1 - p0) * 100:+.1f}%p (지금 {p0 * 100:.1f}%)")
    st.caption("예: 예방정비를 하면 '예방정비 뒤 가동시간'이 0이 된다. 값을 바꿔 확률이 얼마나 내려가는지 본다.")

with tab4:
    back = pd.read_csv(ART / "backtest_2025.csv", encoding="utf-8-sig")
    tot = back[["model", "oldest", "random"]].sum()
    st.write(f"2025년 매달 {card['capacity_per_month']}대씩 골랐을 때 실제로 30일 안에 재고장난 장비 수 — "
             f"모델 **{tot['model']:.0f}대**, 오래된 순 {tot['oldest']:.0f}대, 무작위(기대값) {tot['random']:.0f}대")
    st.line_chart(back.rename(columns={"model": "모델", "oldest": "오래된 순", "random": "무작위"}),
                  x="month", y=["모델", "오래된 순", "무작위"])
    st.dataframe(back, width="stretch", hide_index=True)

with tab5:
    t = card["test_2025"]
    m1, m2, m3 = st.columns(3)
    m1.metric("2025 PR-AUC", t["pr_auc"], f"무작위 {t['base_rate']}")
    m2.metric("2025 ROC-AUC", t["roc_auc"])
    m3.metric("위험 표시 기준", card["threshold"])
    st.write("**예측 대상**:", card["target"])
    st.write("**임계값 정한 방법**:", card["threshold_rule"])
    st.write("**뺀 컬럼**")
    st.table(pd.DataFrame(card["excluded"].items(), columns=["컬럼", "이유"]))
    st.write("**한계**")
    for lim in card["limits"]:
        st.write("- " + lim)

with tab6:
    st.subheader("🤖 Ollama AI 정비반장 소견 및 질의응답")
    col_o1, col_o2, col_o3 = st.columns([2, 1.5, 1.5])
    eid6 = col_o1.selectbox(
        "점검 소견을 생성할 드론 선택",
        ranked["equipment_id"],
        format_func=lambda e: f"{ranked.index[ranked['equipment_id'] == e][0]}위 · {e} ({ranked.loc[ranked['equipment_id'] == e, 'equipment_type'].iloc[0]}, {ranked.loc[ranked['equipment_id'] == e, '확률(%)'].iloc[0]}%)",
        key="ollama_eid",
    )
    ollama_url = col_o2.text_input("Ollama 서버 주소", value="http://localhost:11434")
    ollama_model = col_o3.text_input("Ollama 모델명", value="qwen2.5:14b")

    r6 = ranked[ranked["equipment_id"] == eid6].iloc[0]
    user_q = st.text_input(
        "추가 질문 (선택 사항)",
        value="이 드론의 재고장 위험 요인을 분석하고, 현장 정비사가 오늘 즉시 확인해야 할 핵심 점검 항목 3가지를 업무 보고서 문체로 알려줘.",
    )

    if st.button("AI 정비 소견 생성", type="primary"):
        prompt = (
            f"당신은 군 드론 예측정비 전문 AI 보조관입니다. 아래는 2026-01 정비 기록에 대한 LightGBM 예측 결과입니다.\n"
            f"- 드론 번호: {r6['equipment_id']} ({r6['equipment_type']}, 부대: {r6['unit_code']})\n"
            f"- 최근 정비일: {r6['maintenance_date']:%Y-%m-%d} ({r6['maintenance_type']})\n"
            f"- 30일 내 재고장 예측 확률: {r6['확률(%)']}% (위험 표시 기준: {thr * 100:.0f}%)\n"
            f"- 위험을 올린 핵심 요인(SHAP 상위 3): {r6['reasons']}\n"
            f"- 상세 수치: 최근 90일 고장 {int(r6['fail_count_90d'])}회, 직전 고장 뒤 경과일 {r6['days_since_last_failure']}일, "
            f"예방정비 뒤 비행시간 {r6['hours_since_last_pm']}h, 정비 성공 여부 {int(r6['maintenance_success'])}, "
            f"누적 비행시간 {r6['cumulative_operating_hours']}h, 경과연수 {r6['equipment_age_years']}년\n\n"
            f"요청사항: {user_q}\n"
            f"(마지막 줄에는 반드시 '※ 최종 판단은 정비 담당자가 합니다.'를 적으세요.)"
        )
        req_body = json.dumps({"model": ollama_model, "prompt": prompt, "stream": False}).encode("utf-8")
        req = urllib.request.Request(
            f"{ollama_url.rstrip('/')}/api/generate",
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with st.spinner(f"Ollama ({ollama_model}) 모델이 정비 소견을 작성 중입니다..."):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    res_data = json.loads(resp.read().decode("utf-8"))
                    st.success("Ollama AI 정비 소견 생성 완료")
                    st.markdown(res_data.get("response", ""))
            except Exception as err:
                st.warning(f"Ollama 서버({ollama_url})에 연결할 수 없어 규칙 기반 표준 소견을 대신 표시합니다. ({type(err).__name__})")
                st.markdown(
                    f"**[{r6['equipment_id']} ({r6['equipment_type']} · {r6['unit_code']}) 표준 재점검 소견]**\n"
                    f"1. **위험도 평가**: 재고장 확률 **{r6['확률(%)']}%** (기준 {thr * 100:.0f}% 대비 {'위험 높음' if r6['prob'] >= thr else '관찰 대상'})\n"
                    f"2. **핵심 위험 요인**: `{r6['reasons']}`\n"
                    f"3. **권고 조치**: {'직전 정비 미완료(정비 성공 여부=0) 상태이므로 비행 투입 전 미해결 결함 부위를 최우선 재수리하십시오.' if int(r6['maintenance_success']) == 0 else '반복 고장 이력 및 누적 비행 피로도를 고려하여 구동·센서계 정밀 재점검을 권고합니다.'}\n\n"
                    f"※ 최종 판단은 정비 담당자가 합니다."
                )

st.divider()
st.subheader("부대별 요약")
unit_summary = (
    ranked.assign(위험_높음=ranked["prob"].ge(thr))
    .groupby("unit_code", as_index=False)
    .agg(
        정비_장비_수=("equipment_id", "count"),
        재점검_대상_수=("재점검", "sum"),
        위험_높음_수=("위험_높음", "sum"),
    )
    .rename(columns={
        "unit_code": "부대",
        "정비_장비_수": "정비 장비 수",
        "재점검_대상_수": "재점검 대상 수",
        "위험_높음_수": "위험 높음 수",
    })
    .sort_values("재점검 대상 수", ascending=False)
    .reset_index(drop=True)
)
st.dataframe(unit_summary, width="stretch", hide_index=True)
st.write(f"재점검 대상 합계: **{int(unit_summary['재점검 대상 수'].sum())}대**")

