"""진행 확인: 블록 1 ~ 8 가운데 어디까지 했는지, AI가 만든 결과가 맞는지 한 화면에 보여 준다.

아무것도 고치지 않는다. 몇 번을 실행해도 된다. AGENTS.md · 스킬 · pdm_lib.py · artifacts/ · reports/ 를 본다.
실행 (pdm_ai_lab 폴더, 가상환경 켠 상태):  python scripts/check.py
"""
import json
import logging
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):            # Windows에서 결과를 다른 프로그램(에이전트)으로 넘길 때 한글이 깨지지 않게
    sys.stdout.reconfigure(encoding="utf-8")

DONE, PART, TODO = "[완료]", "[진행]", "[    ]"
NAMES = {1: "준비 · 기본 규칙", 2: "EDA", 3: "데이터 준비", 4: "규칙 없는 모델 감사", 5: "규칙 · 스킬로 다시 학습",
         6: "운영 기준", 7: "앱", 8: "스킬 · 보고서 검수"}
BLANK = "(여기에 적기)"


def read(path):
    return path.read_text(encoding="utf-8") if path.exists() else ""


def rules(section):
    """AGENTS.md '## 규칙' 아래 '### <section>'에 적은 규칙 줄(- 로 시작) 목록."""
    m = re.search(rf"(?ms)^### {re.escape(section)}\s*\n(.*?)(?=^#{{2,3}} |\Z)", read(ROOT / "AGENTS.md"))
    return [ln.strip() for ln in (m.group(1) if m else "").splitlines() if ln.strip().startswith("- ")]


def noted(block):
    """reports/verification.md 의 '## 블록 N' 칸을 다 적었는가. (적은 칸 수, 전체 칸 수)"""
    m = re.search(rf"(?ms)^## 블록 {block} .*?\n(.*?)(?=^## |\Z)", read(ROOT / "reports" / "verification.md"))
    items = [ln for ln in (m.group(1) if m else "").splitlines() if ln.strip().startswith("- ")]
    return sum(BLANK not in ln for ln in items), len(items)


def note_msg(block):
    done, total = noted(block)
    return done == total and total > 0, f"검증 기록 {done}/{total}칸"


def merge(parts):
    """[(ok, 글)] → (표시, 글). 모두 ok면 완료, 하나도 아니면 아직, 그 밖은 진행."""
    oks = [ok for ok, _ in parts]
    mark = DONE if all(oks) else TODO if not any(oks) else PART
    return mark, " · ".join(("" if ok else "✗ ") + msg for ok, msg in parts)


def block1():
    miss = []
    for mod in ["pandas", "numpy", "sklearn", "lightgbm", "matplotlib", "streamlit", "joblib"]:
        try:
            __import__(mod)
        except Exception:
            miss.append(mod)
    n = len(rules("기본"))
    return merge([(not miss, "라이브러리 있음" if not miss else f"라이브러리 없음: {', '.join(miss)}"),
                  (n >= 3, f"기본 규칙 {n}줄 (3줄 이상)")])


def block3():
    n = len(rules("데이터 처리"))
    parts = [(n >= 3, f"데이터 처리 규칙 {n}줄 (3줄 이상)")]
    try:
        from pdm_lib import add_time_features, clean, load_history
        d = add_time_features(clean(load_history()))
        bad = []
        if float(d["maintenance_cost"].max()) != 478.0:
            bad.append(f"비용 최대 {d['maintenance_cost'].max():g}(정답 478)")
        if int(d["downtime_hours"].isna().sum()) != 4:
            bad.append(f"가동중단 결측 {int(d['downtime_hours'].isna().sum())}건(정답 4)")
        if int(d["fault_category"].isna().sum()) != 0 or int((d["fault_category"] == "N/A").sum()) != 1707:
            bad.append("fault_category 결측 처리가 다름(N/A 1,707건)")
        if "is_first_maintenance" not in d or int(d["is_first_maintenance"].sum()) != 150:
            bad.append("is_first_maintenance 합이 150이 아님")
        if len(d) != 2346:
            bad.append(f"행 수 {len(d)}(정답 2346, 행을 지우지 않는다)")
        if int(d["fail_count_90d"].sum()) != 3427:
            bad.append(f"fail_count_90d 합 {int(d['fail_count_90d'].sum())}(정답 3427)")
        if int(d["days_since_last_failure"].isna().sum()) != 772 or round(float(d["days_since_last_failure"].sum())) != 129294:
            bad.append("days_since_last_failure가 다름")
        if int(d["hours_since_last_pm"].isna().sum()) != 1 or abs(float(d["hours_since_last_pm"].sum()) - 30350.5) > 30:
            bad.append("hours_since_last_pm이 다름")
        parts.append((not bad, "clean · 시간 피처 값이 정답과 같음" if not bad else "값이 다름: " + ", ".join(bad)))
    except NotImplementedError:
        parts.append((False, "pdm_lib.py의 clean · add_time_features를 아직 만들지 않음"))
    except Exception as e:
        parts.append((False, f"pdm_lib.py 실행 오류 {type(e).__name__}: {str(e)[:50]}"))
    parts.append(note_msg(3))
    return merge(parts)


def block5():
    n = len(rules("분석 · 검증"))
    leak_rule = any("availability_rate" in r for r in rules("분석 · 검증"))
    parts = [(n >= 4 and leak_rule, f"분석 · 검증 규칙 {n}줄" + ("" if leak_rule else " (누수 컬럼 규칙 없음)"))]
    skill = read(ROOT / ".agents" / "skills" / "pdm-train" / "SKILL.md")
    desc = re.search(r"(?m)^description:\s*(.+)$", skill)
    written = bool(desc) and not desc.group(1).strip().startswith("(") and "(데이터를 읽고" not in skill
    parts.append((written, "pdm-train 스킬 작성함" if written else "pdm-train 스킬의 괄호 칸이 남음"))
    art = ROOT / "artifacts"
    need = ["pdm_model.joblib", "model_card.json", "backtest_2025.csv", "test_predictions_2025.csv"]
    have = [f for f in need if (art / f).exists()]
    if len(have) < 4:
        parts.append((False, f"artifacts/ 파일 {len(have)}/4"))
        return merge(parts)
    card = json.loads(read(art / "model_card.json"))
    t, hits, bad = card["test_2025"], card["test_2025"]["top20_hits"], []
    if abs(t["pr_auc"] - 0.774) > 0.01:
        bad.append(f"검증 PR-AUC {t['pr_auc']}(정답 0.774)")
    if abs(hits["model"] - 97) > 2 or abs(hits["oldest"] - 50) > 0:
        bad.append(f"적중 모델 {hits['model']:.0f} / 오래된 순 {hits['oldest']:.0f}(정답 97 / 50)")
    if abs(card["threshold"] - 0.26) > 0.02:
        bad.append(f"기준 {card['threshold']}(비용 300 · 60이면 0.26)")
    try:
        import joblib
        used = set(getattr(joblib.load(art / "pdm_model.joblib"), "feature_names_in_", []))
        banned = used & {"availability_rate", "equipment_id", "unit_code", "maintenance_date", "fault_code"}
        if banned:
            bad.append(f"모델에 넣으면 안 되는 컬럼: {', '.join(sorted(banned))}")
    except Exception as e:
        bad.append(f"모델 파일을 읽지 못함({type(e).__name__})")
    parts.append((not bad, f"PR-AUC {t['pr_auc']} · 적중 {hits['model']:.0f} / {hits['oldest']:.0f} · 기준 {card['threshold']}"
                  if not bad else "값이 다름: " + ", ".join(bad)))
    return merge(parts)


def block7():
    if not (ROOT / "artifacts" / "pdm_model.joblib").exists():
        return TODO, "모델 파일이 없음 (블록 5를 먼저)"
    try:
        logging.disable(logging.CRITICAL)
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
        logging.disable(logging.NOTSET)
        if at.exception:
            return PART, f"앱 실행 오류: {str(at.exception[0].value)[:70]}"
    except Exception as e:
        return PART, f"앱을 실행하지 못함 {type(e).__name__}: {str(e)[:50]}"
    added = "부대별" in read(ROOT / "app.py")
    return (DONE if added else PART), ("앱 실행 오류 없음 · 부대별 요약 있음" if added else "앱 실행 오류 없음 · 부대별 요약은 아직")


def block8():
    n = len(rules("보고"))
    parts = [(n >= 2, f"보고 규칙 {n}줄 (2줄 이상)")]
    rep = ROOT / "reports"
    score = rep / "score_2026-01.json"
    plans = [p for p in rep.glob("*.md") if "재점검" in p.name] if rep.exists() else []
    if not score.exists():
        parts.append((False, "pdm-risk 스킬 스크립트를 아직 실행하지 않음"))
    elif not plans:
        parts.append((False, "에이전트 보고서(reports/…재점검계획.md)가 아직 없음"))
    else:
        truth = {r["equipment_id"]: round(r["prob"] * 100, 1) for r in json.loads(read(score))["rows"]}
        pairs = re.findall(r"((?:DR|EQ)-\d{4})[^\n]*?(\d{1,3}\.\d)\s?%", read(plans[0]))
        wrong = [f"{e} {p}%(스킬 {truth.get(e, '없음')})" for e, p in pairs if truth.get(e) is None or abs(truth[e] - float(p)) > 0.05]
        if not pairs:
            parts.append((False, f"{plans[0].name}에서 장비 · 확률을 찾지 못함"))
        else:
            parts.append((not wrong, f"보고서 숫자 {len(pairs)}개가 스킬 출력과 같음" if not wrong
                          else f"스킬 출력과 다른 숫자 {len(wrong)}개: " + ", ".join(wrong[:3])))
    parts.append(note_msg(8))
    return merge(parts)


def main():
    rows = {1: block1(), 2: merge([note_msg(2)])}
    rows[3] = block3() if "라이브러리 없음" not in rows[1][1] else (TODO, "블록 1을 먼저 끝냅니다")
    rows[4] = merge([note_msg(4)])
    rows[5] = block5()
    rows[6] = merge([note_msg(6)])
    rows[7] = block7()
    rows[8] = block8()

    print("예측정비 실습(AI와 함께) 진행 확인")
    print("-" * 66)
    for b, (mark, msg) in rows.items():
        print(f"{mark} 블록 {b} {NAMES[b]}\n         {msg}")
    print("-" * 66)
    line = " ".join(f"{b}{'O' if m == DONE else '~' if m == PART else '.'}" for b, (m, _) in rows.items())
    todo = [b for b, (m, _) in rows.items() if m != DONE]
    print("한 줄 요약 :", line, "  (O 완료 · ~ 진행 중 · . 아직 · ✗ 는 남은 항목)")
    print("다음 할 일 :", f"블록 {todo[0]} {NAMES[todo[0]]}" if todo else "모두 끝났습니다")
    auto = [b for b in todo if b in (1, 3, 5, 8)]                      # 따라잡기가 채워 줄 수 있는 블록
    if auto:
        print(f"따라잡기   : python scripts/catchup.py {max(3, auto[0])}   (규칙 · 코드 · 스킬을 기준 답으로 채웁니다. 검증 기록은 직접 씁니다)")


if __name__ == "__main__":
    main()
