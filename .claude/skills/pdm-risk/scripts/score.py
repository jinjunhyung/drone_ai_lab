"""pdm-risk 스킬 도구: 학습한 모델로 정비 기록에 재고장 확률을 매기고 결과를 표로 출력한다.

pdm_ai_lab 폴더에서 실행한다.
  python .agents/skills/pdm-risk/scripts/score.py                         # 이번 달 상위 20대
  python .agents/skills/pdm-risk/scripts/score.py --top 15                # 상위 15대
  python .agents/skills/pdm-risk/scripts/score.py --type 발전기           # 장비 유형으로 거르기
  python .agents/skills/pdm-risk/scripts/score.py --explain EQ-0051       # 장비 1대의 근거
  python .agents/skills/pdm-risk/scripts/score.py --input data/다른달.csv  # 다른 CSV
결과는 화면(마크다운)에 출력하고, reports/score_<달>.json 에도 저장한다.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):            # Windows에서 결과를 다른 프로그램(에이전트)으로 넘길 때 한글이 깨지지 않게
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402

from pdm_lib import FEATURE_KO, FEATURES, MONTH_CSV, contributions, load_history, load_model, score_month  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(MONTH_CSV))
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--type", default=None, help="장비 유형: 차량 · 통신장비 · 발전기 · 화기")
    ap.add_argument("--explain", default=None, help="장비 번호 (예: EQ-0051)")
    a = ap.parse_args()

    model, card = load_model()
    month = pd.read_csv(ROOT / a.input if not Path(a.input).is_absolute() else a.input, encoding="utf-8-sig")
    ranked = score_month(model, load_history(), month)
    ranked.index = ranked.index + 1
    period = ranked["maintenance_date"].dt.strftime("%Y-%m").mode().iloc[0]
    thr, top = card["threshold"], a.top or card["capacity_per_month"]

    if a.explain:
        row = ranked[ranked["equipment_id"] == a.explain]
        if row.empty:
            sys.exit(f"{a.explain}: 이 기간 정비 기록에 없습니다.")
        c = contributions(model, row[FEATURES]).iloc[0].sort_values(ascending=False)
        print(f"## {a.explain} · 순위 {row.index[0]} / {len(ranked)} · 재고장 확률 {row['prob'].iloc[0] * 100:.1f}% "
              f"(위험 표시 기준 {thr * 100:.0f}%)\n")
        print("| 요인 | 값 | 기여도(+면 위험을 올림) |\n| --- | --- | --- |")
        for f, v in c.items():
            print(f"| {FEATURE_KO.get(f, f)} | {row[f].iloc[0]} | {v:+.3f} |")
        return

    view = ranked if not a.type else ranked[ranked["equipment_type"] == a.type]
    pick = view.head(top)
    print(f"## {period} 재점검 후보 상위 {len(pick)}대 (정비 장비 {len(view)}대 중, 위험 표시 기준 {thr * 100:.0f}%)\n")
    print("| 전체 순위 | 장비 | 유형 | 부대 | 최근 정비일 | 정비 구분 | 확률 | 위험 | 위험을 올린 요인 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, r in pick.iterrows():
        print(f"| {i} | {r['equipment_id']} | {r['equipment_type']} | {r['unit_code']} | {r['maintenance_date']:%Y-%m-%d} "
              f"| {r['maintenance_type']} | {r['prob'] * 100:.1f}% | {'높음' if r['prob'] >= thr else '-'} | {r['reasons']} |")
    print(f"\n- 위험 높음(기준 이상): {int((view['prob'] >= thr).sum())}대 · 월 재점검 가능 대수(모델 정보): "
          f"{card['capacity_per_month']}대")
    print(f"- 모델: 2025 시험 PR-AUC {card['test_2025']['pr_auc']}, 월 {card['capacity_per_month']}대 적중 "
          f"{card['test_2025']['top20_hits']['model']:.0f}대 (오래된 순 {card['test_2025']['top20_hits']['oldest']:.0f}대)")

    out = ROOT / "reports" / f"score_{period}.json"
    out.parent.mkdir(exist_ok=True)
    keep = ["equipment_id", "equipment_type", "unit_code", "maintenance_type", "prob", "reasons"]
    rec = view[keep].assign(rank=view.index, maintenance_date=view["maintenance_date"].dt.strftime("%Y-%m-%d"))
    out.write_text(json.dumps({"period": period, "threshold": thr, "top": top,
                               "rows": rec.round({"prob": 4}).to_dict("records")}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"- 전체 결과 저장: reports/{out.name}")


if __name__ == "__main__":
    main()
