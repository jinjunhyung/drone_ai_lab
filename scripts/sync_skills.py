"""스킬 복사: .agents/skills/ 의 스킬을 .claude/skills/ · .github/skills/ 로 복사한다.

Codex · Gemini CLI · Antigravity는 .agents/skills/ 를, Claude Code는 .claude/skills/ 를, GitHub Copilot은 .github/skills/ 를 읽는다.
스킬은 .agents/skills/ 에서만 고치고, 고친 뒤 이 스크립트를 실행한다.
실행 (pdm_ai_lab 폴더):  python scripts/sync_skills.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    src = ROOT / ".agents" / "skills"
    for name in (".claude", ".github"):
        dst = ROOT / name / "skills"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
        print(f"복사함 (.agents/skills → {name}/skills):", ", ".join(sorted(p.name for p in dst.iterdir() if p.is_dir())))


if __name__ == "__main__":
    main()
