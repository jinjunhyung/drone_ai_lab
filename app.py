"""정비 우선순위 앱 — 이번 달 정비 기록에서 재점검할 장비를 고른다.

실행 (pdm_ai_lab 폴더, 가상환경 켠 상태):  streamlit run app.py
먼저 블록 5(python scripts/train.py)로 artifacts/ 에 모델을 저장해 둔다.
"""
import json
import urllib.error
import urllib.request

import pandas as pd
import streamlit as st

import db
from pdm_lib import (ART, FEATURE_KO, FEATURES, MONTH_CSV, TARGET, contributions, load_history, load_model,
                     score_month)

st.set_page_config(page_title="정비 우선순위", page_icon="🛠️", layout="wide")
st.title("🛠️ 정비 우선순위 — 이번 달 재점검 목록")
db.init_db()


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
    st.caption(f"저장소: **{db.backend_name()}**")

ranked = score(up if up is not None else MONTH_CSV)
ranked["위험"] = ranked["prob"].ge(thr).map({True: "높음", False: "-"})
ranked["재점검"] = ranked.index <= cap

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "① 재점검 목록", "② 장비별 근거", "③ 만약에", "④ 되돌려 보기(2025)", "⑤ 모델 정보", "⑥ 정비 요청 · AI 소견 (Postgres)"
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
    st.subheader("🗄️ AI 정비 소견 및 재점검 요청 관리 (Postgres 연동)")
    st.info(f"현재 연결된 저장소: **{db.backend_name()}**  (교육용 가상 데이터만 입력)")

    eid6 = st.selectbox(
        "드론 선택",
        ranked["equipment_id"],
        format_func=lambda e: f"{ranked.index[ranked['equipment_id'] == e][0]}위 · {e} · {ranked.loc[ranked['equipment_id'] == e, 'equipment_type'].iloc[0]} ({ranked.loc[ranked['equipment_id'] == e, 'unit_code'].iloc[0]}, 확률 {ranked.loc[ranked['equipment_id'] == e, '확률(%)'].iloc[0]}%)",
        key="req_eid",
    )
    r6 = ranked[ranked["equipment_id"] == eid6].iloc[0]
    default_priority = "높음" if r6["prob"] >= thr else ("보통" if r6["prob"] >= 0.15 else "낮음")
    auto_note = (
        f"[AI 재점검 소견 | 확률 {r6['확률(%)']}%] {r6['equipment_type']}({r6['unit_code']}) "
        f"주요 위험 요인: {r6['reasons']} → "
        f"{'정비 미완료(성공=0) 결함 부위 최우선 재수리 요망' if int(r6['maintenance_success']) == 0 else '반복 고장·누적 피로 부위 정밀 재점검 요망'}"
    )

    st.markdown(
        f"- **AI 분석 요약**: **{r6['equipment_id']}** ({r6['equipment_type']} · {r6['unit_code']}) | "
        f"재고장 확률 **{r6['확률(%)']}%** ({'위험 높음' if r6['prob'] >= thr else '관찰 대상'}) | "
        f"핵심 요인: `{r6['reasons']}`"
    )

    with st.form("add_request_form", clear_on_submit=False):
        f_col1, f_col2 = st.columns([1, 2])
        with f_col1:
            priority = st.selectbox("우선순위", db.PRIORITIES, index=db.PRIORITIES.index(default_priority))
            requester_role = st.text_input("요청자 역할명", value="드론 정비반장")
        with f_col2:
            title = st.text_area("요청 내용 (AI 자동 소견 포함 · 편집 가능)", value=auto_note, height=90)
        submitted = st.form_submit_button("Postgres 저장소에 정비 요청 등록", type="primary")
        if submitted:
            try:
                new_id = db.add_request(
                    equipment_code=eid6,
                    title=title,
                    priority=priority,
                    requester_role=requester_role,
                )
                st.success(f"정비 요청이 {db.backend_name()}에 저장되었습니다. (요청 번호: #{new_id})")
            except ValueError as err:
                st.error(str(err))

    st.divider()
    st.markdown("#### 📋 등록된 정비 요청 목록 (Postgres 실시간 조회)")
    status_options = ["전체"] + list(db.STATUSES)
    selected_status = st.radio("상태별 보기", status_options, horizontal=True)
    filter_status = None if selected_status == "전체" else selected_status
    requests_df = db.list_requests(status=filter_status)

    if requests_df.empty:
        st.write("조회된 정비 요청이 없습니다.")
    else:
        display_df = requests_df.rename(
            columns={
                "id": "번호",
                "created_at": "접수일시",
                "equipment_code": "장비코드",
                "title": "요청 내용 (AI 소견)",
                "priority": "우선순위",
                "requester_role": "요청자 역할명",
                "status": "상태",
            }
        )
        st.dataframe(display_df, width="stretch", hide_index=True)

    all_requests_df = db.list_requests()
    if not all_requests_df.empty:
        with st.form("update_status_form"):
            id_options = [
                f"#{row['id']} ({row['equipment_code']} · {row['title'][:40]} | 현재: {row['status']})"
                for _, row in all_requests_df.iterrows()
            ]
            u_col1, u_col2 = st.columns([3, 1])
            with u_col1:
                selected_req = st.selectbox("상태를 변경할 요청 번호", id_options)
            with u_col2:
                new_status = st.selectbox("변경할 상태", db.STATUSES)
            if st.form_submit_button("상태 변경 저장"):
                req_id = int(selected_req.split(" ")[0].lstrip("#"))
                db.update_status(req_id, new_status)
                st.success(f"요청 #{req_id}의 상태를 '{new_status}'(으)로 변경했습니다.")
                st.rerun()

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

