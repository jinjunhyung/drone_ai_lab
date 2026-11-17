"""드론 정비 · 재점검 요청 저장소 (SQLite 내 PC 파일 및 Supabase Postgres 겸용).

- 연결 주소를 찾는 순서:
    1) .streamlit/secrets.toml 의 [connections.sql] url   (Streamlit Cloud에서는 앱 설정 Secrets 칸)
    2) 환경 변수 DATABASE_URL
    3) 둘 다 없으면 sqlite:///data/app.db   (이 폴더 안의 파일)
- 표가 없으면 init_db()가 만든다.
- 데이터는 교육용 가상 데이터만 넣는다.
"""
from __future__ import annotations

import os
import tomllib
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import (Column, DateTime, Integer, MetaData, String, Table,
                        create_engine, insert, select, update)
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parent
DEFAULT_URL = f"sqlite:///{(ROOT / 'data' / 'app.db').as_posix()}"
STATUSES = ["접수", "진행", "완료"]
PRIORITIES = ["높음", "보통", "낮음"]

metadata = MetaData()
requests_table = Table(
    "maintenance_requests", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime, nullable=False),
    Column("equipment_code", String(20), nullable=False),
    Column("title", String(200), nullable=False),
    Column("priority", String(10), nullable=False),
    Column("requester_role", String(50), nullable=False),
    Column("status", String(10), nullable=False),
)

_engines: dict[str, Engine] = {}


def database_url() -> str:
    """연결 주소. secrets.toml → st.secrets → 환경 변수 → 기본 SQLite 순서로 찾는다."""
    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        try:
            with open(secrets_path, "rb") as f:
                url = tomllib.load(f).get("connections", {}).get("sql", {}).get("url")
                if url:
                    return str(url)
        except Exception:
            pass
    try:
        import streamlit as st
        url = st.secrets["connections"]["sql"]["url"]
        if url:
            return str(url)
    except Exception:
        pass
    return os.environ.get("DATABASE_URL") or DEFAULT_URL


def backend_name(url: str | None = None) -> str:
    """화면에 보일 저장소 이름: 'SQLite (내 PC 파일)' 또는 'Postgres (Supabase 등)'."""
    url = url or database_url()
    return "SQLite (내 PC 파일)" if url.startswith("sqlite") else "Postgres (Supabase 등)"


def _driver_url(url: str) -> str:
    """Supabase에서 복사한 postgresql:// 주소에 드라이버(psycopg 3)를 붙인다."""
    for head in ("postgresql://", "postgres://"):
        if url.startswith(head):
            return "postgresql+psycopg://" + url[len(head):]
    return url


def get_engine(url: str | None = None) -> Engine:
    url = _driver_url(url or database_url())
    if url not in _engines:
        if url.startswith("sqlite:///"):
            os.makedirs(os.path.dirname(url.removeprefix("sqlite:///")) or ".", exist_ok=True)
        _engines[url] = create_engine(url, pool_pre_ping=True)
    return _engines[url]


def init_db(url: str | None = None) -> None:
    """표가 없으면 만든다. 있으면 그대로 둔다."""
    metadata.create_all(get_engine(url))


def add_request(equipment_code: str, title: str, priority: str, requester_role: str,
                url: str | None = None) -> int:
    """정비 요청 한 건을 '접수' 상태로 저장하고 번호(id)를 돌려준다."""
    if not equipment_code or not title.strip():
        raise ValueError("장비와 요청 내용은 비울 수 없습니다.")
    if priority not in PRIORITIES:
        raise ValueError(f"우선순위는 {PRIORITIES} 중 하나입니다.")
    with get_engine(url).begin() as con:
        res = con.execute(insert(requests_table).values(
            created_at=datetime.now().replace(microsecond=0),
            equipment_code=equipment_code,
            title=title.strip()[:200],
            priority=priority,
            requester_role=requester_role.strip() or "정비반장",
            status="접수",
        ))
        return int(res.inserted_primary_key[0])


def list_requests(status: str | None = None, url: str | None = None) -> pd.DataFrame:
    """정비 요청 목록 (최근 것부터). status를 주면 그 상태만."""
    q = select(requests_table).order_by(requests_table.c.id.desc())
    if status:
        q = q.where(requests_table.c.status == status)
    with get_engine(url).connect() as con:
        return pd.DataFrame(con.execute(q).mappings().all(),
                            columns=[c.name for c in requests_table.columns])


def update_status(request_id: int, status: str, url: str | None = None) -> None:
    """요청 하나의 상태를 바꾼다 (접수 · 진행 · 완료)."""
    if status not in STATUSES:
        raise ValueError(f"상태는 {STATUSES} 중 하나입니다.")
    with get_engine(url).begin() as con:
        con.execute(update(requests_table).where(requests_table.c.id == int(request_id)).values(status=status))
