"""따라잡기: 지정한 블록까지 비어 있는 규칙 · 코드 · 스킬을 기준 답으로 채워 다음 블록으로 넘어갈 수 있게 한다.

  python scripts/catchup.py 3      # 블록 3까지: 기본 · 데이터 처리 규칙, pdm_lib.py의 clean · add_time_features
  python scripts/catchup.py 5      # 블록 5까지: + 분석 · 검증 규칙, pdm-train 스킬, scripts/train.py, 모델 학습
  python scripts/catchup.py 8      # 블록 8까지: + 보고 규칙, pdm-risk 스킬 스크립트 실행

- 이미 쓴 규칙 · 스킬은 건드리지 않는다. 비어 있는 곳만 채운다. 고치기 전 파일은 backup/<시각>/ 에 둔다.
- 검증 기록(reports/verification.md)과 에이전트 보고서는 채우지 않는다. 직접 한다.
- 실행 전에 VS Code에서 AGENTS.md · pdm_lib.py · SKILL.md를 저장하고 탭을 닫는다.
기준 답은 scripts/reference/ 에 있다. 실습지의 '답 보기'와 같은 내용이다.
"""
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "scripts" / "reference"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
SECTIONS = {"기본": 1, "데이터 처리": 3, "분석 · 검증": 5, "보고": 8}      # 규칙 묶음 → 그 규칙을 쓰는 블록


def run(*args):
    sys.stdout.flush()
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main():
    if len(sys.argv) != 2 or not sys.argv[1].isdigit() or not 3 <= int(sys.argv[1]) <= 8:
        sys.exit("사용법: python scripts/catchup.py <블록 번호 3 ~ 8>   예) python scripts/catchup.py 5")
    upto = int(sys.argv[1])
    backup = ROOT / "backup" / datetime.now().strftime("%m%d_%H%M%S")
    backup.mkdir(parents=True)
    done = []

    def keep(path):
        if path.exists() and not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)

    agents = ROOT / "AGENTS.md"                                         # 1) 규칙: 비어 있는 묶음만 채운다
    text, ref = agents.read_text(encoding="utf-8"), (REF / "AGENTS.md").read_text(encoding="utf-8")
    for name, block in SECTIONS.items():
        if block > upto:
            continue
        pat = rf"(?ms)^(### {re.escape(name)}\s*\n)(.*?)(?=^#{{2,3}} |\Z)"
        cur = re.search(pat, text)
        if cur and not any(ln.strip().startswith("- ") for ln in cur.group(2).splitlines()):
            body = re.search(pat, ref).group(2).strip("\n")
            text = text[:cur.start()] + cur.group(1) + body + "\n\n" + text[cur.end():]
            done.append(f"AGENTS.md 규칙 '{name}'")
    if text != agents.read_text(encoding="utf-8"):
        keep(agents)
        agents.write_text(text.rstrip("\n") + "\n", encoding="utf-8")

    lib = ROOT / "pdm_lib.py"                                           # 2) 블록 3: 데이터 준비 함수
    if "NotImplementedError" in lib.read_text(encoding="utf-8"):
        keep(lib)
        shutil.copy2(REF / "pdm_lib.py", lib)
        done.append("pdm_lib.py (clean · add_time_features)")

    if upto >= 5:                                                       # 3) 블록 5: 스킬 · 학습 스크립트 · 모델
        skill = ROOT / ".agents" / "skills" / "pdm-train" / "SKILL.md"
        if not skill.exists() or "(데이터를 읽고" in skill.read_text(encoding="utf-8"):
            keep(skill)
            skill.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REF / "pdm-train" / "SKILL.md", skill)
            done.append("pdm-train 스킬")
        train = ROOT / "scripts" / "train.py"
        if not train.exists():
            shutil.copy2(REF / "train.py", train)
            done.append("scripts/train.py")
        if not (ROOT / "artifacts" / "pdm_model.joblib").exists():
            print("모델 파일이 없어 scripts/train.py를 실행합니다 …")
            run("scripts/train.py")
            done.append("모델 학습")
        run("scripts/sync_skills.py")

    if upto >= 8 and not (ROOT / "reports" / "score_2026-01.json").exists():   # 4) 블록 8: 스킬 스크립트 실행
        run(".agents/skills/pdm-risk/scripts/score.py", "--top", "5")
        done.append("pdm-risk 스크립트 실행")

    if not any(backup.iterdir()):
        backup.rmdir()
    print(f"\n블록 {upto}까지 채운 것 ({len(done)}): " + (", ".join(done) if done else "없음 (이미 되어 있습니다)"))
    if backup.exists():
        print(f"고치기 전 파일: backup/{backup.name}/")
    print("\n다음 순서로 이어 갑니다.")
    print(" 1. 채워진 규칙 · 코드 · 스킬을 실습지 설명과 함께 읽습니다. 왜 그렇게 썼는지 한 줄로 말할 수 있어야 합니다.")
    print(" 2. reports/verification.md 의 해당 블록 칸은 직접 확인해 적습니다.")
    print(" 3. python scripts/check.py 로 진행을 확인합니다.")


if __name__ == "__main__":
    main()
