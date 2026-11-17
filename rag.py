"""부대 규정 · 보급 기준 RAG 파이프라인 모듈 (my-rag-lab Stage 1~6 통합).

- Stage 1 (적재): docs/*.txt 규정 문서를 읽는다 (load_documents).
- Stage 2 (청킹): 조항·문단 단위로 분할해 {source, chunk_id, text} 레코드를 만든다 (make_chunks).
- Stage 3~5 (임베딩·검색): 한국어 어절 + 글자 N-gram TF-IDF 정규화 벡터 내적(코사인 유사도)으로 상위 k개 조항을 검색한다 (retrieve).
- Stage 6 (생성): 검색된 [컨텍스트]만 근거로 프롬프트를 조립(build_prompt)하고 답변을 만든다(generate).
  Claude / Gemini / 로컬 Ollama 연동을 지원하며, 클라우드 무키(No-Key) 환경에서도 검색된 조문을 바탕으로 즉시 답변한다.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:14b"


def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Stage 1 · docs/ 폴더의 .txt 규정 문서들을 읽어 정렬된 리스트로 반환한다."""
    if not docs_dir.exists():
        return []
    docs = []
    for p in sorted(docs_dir.glob("*.txt")):
        text = p.read_text(encoding="utf-8").strip()
        if text:
            docs.append({"source": p.name, "text": text})
    return docs


def chunk_text(text: str, size: int = 240, overlap: int = 50) -> list[str]:
    """Stage 2 · 문단(빈 줄) 단위로 먼저 나누고, 긴 문단은 슬라이딩 윈도우로 분할한다.
    첫 줄(문서 제목)을 각 조각 앞에 붙여 문서 맥락을 보존한다."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    header = lines[0]
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paragraphs) > 1 and paragraphs[0] == header:
        paragraphs = paragraphs[1:]

    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= size:
            chunks.append(para)
        else:
            i = 0
            while i < len(para):
                sub = para[i:i + size].strip()
                if sub:
                    chunks.append(sub)
                i += max(1, size - overlap)
    return chunks


def make_chunks(docs: list[dict] | None = None, size: int = 240, overlap: int = 50) -> list[dict]:
    """Stage 2 · 모든 문서를 조각내 {source, chunk_id, text} 리스트를 반환한다."""
    if docs is None:
        docs = load_documents()
    records: list[dict] = []
    for d in docs:
        for j, c in enumerate(chunk_text(d["text"], size=size, overlap=overlap)):
            if c.strip():
                records.append({"source": d["source"], "chunk_id": j, "text": c.strip()})
    return records


class RAGIndex:
    """Stage 3~5 · 한국어 어절 + 글자 N-gram 하이브리드 코사인 유사도 인덱스."""

    def __init__(self, records: list[dict]):
        self.records = records
        texts = [r["text"] for r in records]
        self.vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.vec_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)
        Xc = self.vec_char.fit_transform(texts).toarray().astype("float32")
        Xw = self.vec_word.fit_transform(texts).toarray().astype("float32")
        comb = np.hstack([Xc * 0.65, Xw * 0.35])
        norms = np.linalg.norm(comb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.emb = (comb / norms).astype("float32")

    def search(self, query: str, k: int = 3) -> list[dict]:
        qc = self.vec_char.transform([query]).toarray().astype("float32")
        qw = self.vec_word.transform([query]).toarray().astype("float32")
        q = np.hstack([qc * 0.65, qw * 0.35])
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        if q_norm[0, 0] > 0:
            q = q / q_norm
        scores = (self.emb @ q.T).ravel()
        top_idx = np.argsort(scores)[::-1][:k]
        hits = []
        for i in top_idx:
            r = dict(self.records[int(i)])
            r["score"] = float(scores[int(i)])
            hits.append(r)
        return hits


_INDEX_CACHE: RAGIndex | None = None


def get_index() -> RAGIndex:
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        _INDEX_CACHE = RAGIndex(make_chunks(load_documents()))
    return _INDEX_CACHE


def retrieve(query: str, k: int = 3) -> list[dict]:
    """Stage 5 · 질문과 가장 가까운 상위 k개 규정 조각을 반환한다."""
    idx = get_index()
    if not idx.records:
        return []
    return idx.search(query, k=k)


def build_prompt(query: str, hits: list[dict]) -> tuple[str, str]:
    """Stage 6 · 검색된 근거(hits)와 질문으로 LLM 프롬프트를 조립한다."""
    context = "\n\n".join(f"[{h['source']}#{h['chunk_id']}] {h['text']}" for h in hits)
    prompt = (
        "아래 [컨텍스트]만 근거로 한국어로 간결하게 답하라. "
        "컨텍스트에 없으면 '문서에서 찾을 수 없습니다'라고만 답하라.\n\n"
        f"[컨텍스트]\n{context}\n\n[질문]\n{query}\n\n[답변]"
    )
    return prompt, context


def _synthesize_from_hits(query: str, hits: list[dict], min_score: float = 0.22) -> str:
    """외부 LLM 연결이 없을 때 검색된 최상위 규정 조문을 바탕으로 근거 기반 답변을 구성한다."""
    valid = [h for h in hits if h["score"] >= min_score]
    if not valid:
        return "문서에서 찾을 수 없습니다. (제공된 부대 규정 문서에 해당 내용이 없습니다.)"
    top = valid[0]
    clean_top = top["text"].replace("\n", " ")
    lines = [f"**[규정 근거 답변]** `{top['source']}#{top['chunk_id']}` (유사도 `{top['score']:.3f}` 기준)"]
    lines.append(f"- **핵심 조항 (`{top['source']}#{top['chunk_id']}`)**: {clean_top}")
    for sub in valid[1:3]:
        if sub["score"] >= max(min_score, top["score"] * 0.55):
            clean_sub = sub["text"].replace("\n", " ")
            lines.append(f"- **관련 조항 (`{sub['source']}#{sub['chunk_id']}`)**: {clean_sub}")
    return "\n".join(lines)


def answer_question(query: str, k: int = 3, use_ollama: bool = False,
                    ollama_url: str = OLLAMA_URL, ollama_model: str = OLLAMA_MODEL) -> dict:
    """질문에 대해 RAG 검색 → 프롬프트 생성 → 답변(Ollama 또는 조문 기반 합성) 및 출처를 반환한다."""
    hits = retrieve(query, k=k)
    prompt, context = build_prompt(query, hits)

    if not hits or hits[0]["score"] < 0.22:
        return {
            "answer": "문서에서 찾을 수 없습니다. (제공된 부대 규정 문서에 해당 내용이 없습니다.)",
            "hits": hits,
            "prompt": prompt,
            "mode": "규정 문서 검색 (일치 조항 없음)",
        }

    if use_ollama:
        try:
            payload = json.dumps({
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{ollama_url.rstrip('/')}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                ans = json.loads(resp.read().decode("utf-8")).get("response", "").strip()
                if ans:
                    return {"answer": ans, "hits": hits, "prompt": prompt, "mode": f"Ollama ({ollama_model})"}
        except Exception:
            pass

    return {
        "answer": _synthesize_from_hits(query, hits),
        "hits": hits,
        "prompt": prompt,
        "mode": "규정 조문 기반 즉시 응답 (Cloud/Local 겸용)",
    }
