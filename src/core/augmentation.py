import re
from typing import List, Dict, Any, Tuple

from src.core.config import config
from src.core.document_models import AugmentationResult, AugmentedContextItem, AugmentationConfigModel, RetrievalResult


def token_estimate(text: str) -> int:
    """Rough token estimator: words / 0.75 (~1.33 tokens per word)."""
    if not text:
        return 0
    words = len(text.strip().split())
    return int(words / 0.75)


def sentence_split(text: str) -> List[str]:
    """Simple sentence splitter using punctuation. Conservative for truncation."""
    if not text:
        return []
    # Split on period/question/exclamation followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def normalize_results(results: List[RetrievalResult]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for r in results:
        citation = _format_citation(r)
        normalized.append({
            "chunk_id": r.chunk_id,
            "document_id": r.document_id,
            "text": r.content,
            "citation": citation,
            "score": r.final_score,
            "metadata": r.metadata or {}
        })
    return normalized


def _format_citation(r: RetrievalResult) -> str:
    style = config.augmentation.citation_style
    doc = r.document_id
    meta = r.metadata or {}
    if style == "page":
        page = meta.get("page_number") or getattr(r.citation, "page_number", None)
        if page is not None:
            return f"{doc}, p.{page}"
    if style == "paragraph":
        para = meta.get("paragraph_number") or getattr(r.citation, "paragraph_number", None)
        if para is not None:
            return f"{doc}, para.{para}"
    if style == "section":
        section = meta.get("section") or getattr(r.citation, "section", None)
        if section:
            return f"{doc}, {section}"
    # fallback
    return f"{doc}"


def deduplicate(items: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    """Remove near-duplicates using simple Jaccard similarity over word sets."""
    def sim(a: str, b: str) -> float:
        sa, sb = set(a.lower().split()), set(b.lower().split())
        if not sa or not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb)
        return inter / union if union else 0.0

    kept: List[Dict[str, Any]] = []
    for item in items:
        dup = False
        for k in kept:
            if sim(item["text"], k["text"]) >= threshold:
                dup = True
                break
        if not dup:
            kept.append(item)
    return kept


def rank_and_filter(items: List[Dict[str, Any]], min_score: float, top_k: int) -> List[Dict[str, Any]]:
    filtered = [it for it in items if (it.get("score") or 0) >= min_score]
    filtered.sort(key=lambda x: x.get("score", 0), reverse=True)
    return filtered[:top_k]


def assemble_context(items: List[Dict[str, Any]], max_tokens: int) -> Tuple[List[AugmentedContextItem], int]:
    used: List[AugmentedContextItem] = []
    total_tokens = 0

    for it in items:
        text = it["text"]
        base_tokens = token_estimate(text)
        if total_tokens + base_tokens > max_tokens:
            # trim by sentences
            sentences = sentence_split(text)
            trimmed = []
            t_tokens = 0
            for s in sentences:
                s_tokens = token_estimate(s)
                if total_tokens + t_tokens + s_tokens > max_tokens:
                    break
                trimmed.append(s)
                t_tokens += s_tokens
            text = " ".join(trimmed)
            base_tokens = t_tokens
        if not text:
            continue
        used.append(AugmentedContextItem(
            chunk_id=it["chunk_id"],
            document_id=it["document_id"],
            text=text,
            citation=it["citation"],
            score=it.get("score", 0.0)
        ))
        total_tokens += base_tokens
        if total_tokens >= max_tokens:
            break

    return used, total_tokens


def build_prompt(query: str, context_items: List[AugmentedContextItem], conversation_history: List[Dict[str, str]] | None) -> Tuple[str, List[str]]:
    citations = [ci.citation for ci in context_items]

    system_instructions = (
        "You are a precise assistant. Use only the provided context. "
        "Cite sources inline using [CIT:chunk_id] and list citations at the end."
    )

    history_str = ""
    if conversation_history:
        lines = []
        for m in conversation_history[-10:]:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role.upper()}: {content}")
        history_str = "\n".join(lines)

    context_block_lines = []
    for ci in context_items:
        context_block_lines.append(f"- ({ci.score:.2f}) {ci.text} [CIT:{ci.chunk_id}]")
    context_block = "\n".join(context_block_lines)

    prompt = (
        f"SYSTEM:\n{system_instructions}\n\n"
        f"QUERY:\n{query}\n\n"
        f"HISTORY:\n{history_str}\n\n" if history_str else f"SYSTEM:\n{system_instructions}\n\nQUERY:\n{query}\n\n"
    )
    prompt += f"CONTEXT:\n{context_block}\n\nCITATIONS:\n" + "\n".join(citations)

    return prompt, citations


def augment(
    query: str,
    retrieved_results: List[RetrievalResult],
    conversation_history: List[Dict[str, str]] | None = None,
    override_config: AugmentationConfigModel | None = None,
) -> AugmentationResult:
    cfg = override_config or AugmentationConfigModel(
        max_context_tokens=config.augmentation.max_context_tokens,
        min_score_threshold=config.augmentation.min_score_threshold,
        top_k=config.augmentation.top_k,
        dedup_similarity=config.augmentation.dedup_similarity,
        citation_style=config.augmentation.citation_style,
    )

    normalized = normalize_results(retrieved_results)
    deduped = deduplicate(normalized, cfg.dedup_similarity)
    ranked = rank_and_filter(deduped, cfg.min_score_threshold, cfg.top_k)
    context_items, token_count = assemble_context(ranked, cfg.max_context_tokens)
    prompt, citations = build_prompt(query, context_items, conversation_history)

    ctx_meta = {
        "total_chunks": len(retrieved_results),
        "used_chunks": len(context_items),
        "token_count": token_count,
    }

    return AugmentationResult(
        augmented_prompt=prompt,
        citations=citations,
        context_items=context_items,
        context_metadata=ctx_meta,
    )
