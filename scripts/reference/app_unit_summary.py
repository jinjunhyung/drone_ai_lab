
st.divider()
st.subheader("부대별 요약")
by_unit = (ranked.assign(위험높음=ranked["prob"].ge(thr))
           .groupby("unit_code").agg(정비장비=("equipment_id", "count"), 재점검대상=("재점검", "sum"), 위험높음=("위험높음", "sum"))
           .sort_values("재점검대상", ascending=False).rename_axis("부대").reset_index())
st.dataframe(by_unit, width="stretch", hide_index=True)
st.caption(f"재점검 대상 합계 {int(by_unit['재점검대상'].sum())}대 · 위험 높음 합계 {int(by_unit['위험높음'].sum())}대")
