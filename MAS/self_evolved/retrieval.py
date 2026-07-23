"""Constraint-aware retrieval tools for self-evolved topologies."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

_CONSECUTIVE_NUMBERS = re.compile(
    r"(?<!\d)(\d{2,4})(?:\s*[,/]?\s+)(\d{2,4})(?:\s*[,/]?\s+)(\d{2,4})"
    r"(?:(?:\s*[,/]?\s+)(\d{2,4}))?(?!\d)"
)
_DOT_RANGE = re.compile(r"(?<!\d)(\d{2,6})\s*\.\.\s*(\d{2,6})(?!\d)")
_TITLE = re.compile(r"(?:^|\n)\s*(?:---\s*)?title:\s*([^\n]+)", re.IGNORECASE)


def _bounded_numeric_variants(query: str, *, max_variants: int = 12) -> list[str]:
    """Turn literal intervals into a bounded set of lexical-search variants.

    Short intervals are enumerated because the underlying lexical index treats a
    synthetic ``2001..2004`` token differently from the year printed in a source.
    Wide intervals cannot be enumerated safely, so retain three representative
    values and one range-free query.  The latter retrieves documents that state a
    matching value anywhere in the interval without multiplying tool calls by the
    interval width.
    """

    text = str(query or "").strip()
    if not text:
        return []

    dot_match = _DOT_RANGE.search(text)
    if dot_match:
        start, end = (int(value) for value in dot_match.groups())
        if 0 <= end - start < max_variants:
            return [
                f"{text[: dot_match.start()]}{value}{text[dot_match.end() :]}".strip()
                for value in range(start, end + 1)
            ]
        if end >= start:
            without_range = re.sub(
                r"\s+", " ", f"{text[: dot_match.start()]} {text[dot_match.end() :]}"
            ).strip()
            sampled_values = list(dict.fromkeys((start, (start + end) // 2, end)))
            sampled = [
                f"{text[: dot_match.start()]}{value}{text[dot_match.end() :]}".strip()
                for value in sampled_values
            ]
            return list(dict.fromkeys([without_range, *sampled]))

    sequence_match = _CONSECUTIVE_NUMBERS.search(text)
    if sequence_match:
        values = [int(value) for value in sequence_match.groups() if value is not None]
        if values == list(range(values[0], values[0] + len(values))):
            return [
                f"{text[: sequence_match.start()]}{value}{text[sequence_match.end() :]}".strip()
                for value in values
            ]

    return [text]


def _document_title(snippet: str) -> str:
    match = _TITLE.search(str(snippet or ""))
    if match is None:
        return ""
    words = match.group(1).strip(" -\t").split()
    title = " ".join(words[:12])
    if title.lower() in {"home", "history", "news", "untitled"}:
        return ""
    return title


def augment_with_constraint_search(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add a MANTA-only rank-fusion search tool when search and document read exist."""

    copied = list(tools)
    names = {str(tool.get("name", "")) for tool in copied if isinstance(tool, dict)}
    if "constraint_search" in names or not {"search", "get_document"}.issubset(names):
        return copied

    search_index = next(
        index for index, tool in enumerate(copied) if str(tool.get("name", "")) == "search"
    )
    search_tool = copied[search_index]
    search_handler = search_tool.get("handler")
    if not callable(search_handler):
        return copied

    original_search_handler: Callable[[dict[str, Any]], Any] = search_handler

    def _manta_search(args: dict[str, Any]) -> Any:
        enriched = dict(args)
        enriched.setdefault("k", 12)
        return original_search_handler(enriched)

    copied[search_index] = {
        **search_tool,
        "description": (
            f"{search_tool.get('description', 'Search documents.')} "
            "MANTA uses recall depth 12 unless k is supplied."
        ),
        "handler": _manta_search,
    }
    search_handler = _manta_search

    def _constraint_search(args: dict[str, Any]) -> list[dict[str, Any]]:
        raw_queries = args.get("queries", [])
        if isinstance(raw_queries, str):
            try:
                parsed_queries = json.loads(raw_queries)
            except (TypeError, ValueError):
                parsed_queries = []
            raw_queries = parsed_queries
        if not isinstance(raw_queries, list):
            return []
        queries = list(dict.fromkeys(str(item).strip() for item in raw_queries if str(item).strip()))
        queries = queries[:6]

        clause_rankings: list[dict[str, tuple[int, dict[str, Any]]]] = []
        handler: Callable[[dict[str, Any]], Any] = search_handler
        for query in queries:
            best_for_clause: dict[str, tuple[int, dict[str, Any]]] = {}
            for variant in _bounded_numeric_variants(query):
                results = handler({"query": variant, "k": 20})
                if not isinstance(results, list):
                    continue
                for rank, item in enumerate(results, start=1):
                    if not isinstance(item, dict):
                        continue
                    docid = str(item.get("docid", "")).strip()
                    if not docid:
                        continue
                    previous = best_for_clause.get(docid)
                    if previous is None or rank < previous[0]:
                        best_for_clause[docid] = (rank, item)

            clause_rankings.append(best_for_clause)

        # Keep entity-bridge evidence separate from direct clause rankings. A bridge hit
        # supports the target clue in the context of a candidate entity, but it must not
        # masquerade as direct evidence for both the source and target clauses.
        bridge_rankings: list[dict[str, tuple[int, dict[str, Any]]]] = [
            {} for _ in clause_rankings
        ]
        bridge_calls = 0
        for source_index, source_ranking in enumerate(clause_rankings):
            ordered_sources = sorted(
                source_ranking.values(),
                key=lambda item: (item[0], -float(item[1].get("score", 0.0) or 0.0)),
            )
            for _, source_item in ordered_sources[:2]:
                title = _document_title(str(source_item.get("snippet", "")))
                if not title:
                    continue
                for target_index, target_query in enumerate(queries):
                    if target_index == source_index or bridge_calls >= 18:
                        continue
                    bridge_calls += 1
                    results = handler({"query": f'"{title}" {target_query}', "k": 20})
                    if not isinstance(results, list):
                        continue
                    for rank, item in enumerate(results, start=1):
                        if not isinstance(item, dict):
                            continue
                        docid = str(item.get("docid", "")).strip()
                        if not docid:
                            continue
                        bridged_rank = rank + 1
                        previous = bridge_rankings[target_index].get(docid)
                        if previous is None or bridged_rank < previous[0]:
                            bridge_rankings[target_index][docid] = (bridged_rank, item)

        combined_rankings: list[dict[str, tuple[int, dict[str, Any], bool]]] = []
        for direct, bridged in zip(clause_rankings, bridge_rankings, strict=True):
            combined = {
                docid: (rank, item, True) for docid, (rank, item) in direct.items()
            }
            for docid, (rank, item) in bridged.items():
                previous = combined.get(docid)
                if previous is None or rank < previous[0]:
                    combined[docid] = (rank, item, previous is not None)
            combined_rankings.append(combined)

        fused: dict[str, dict[str, Any]] = {}
        for best_for_clause in combined_rankings:
            for docid, (rank, item, is_direct) in best_for_clause.items():
                record = fused.setdefault(
                    docid,
                    {
                        "docid": docid,
                        "constraint_support": 0,
                        "direct_constraint_support": 0,
                        "bridge_constraint_support": 0,
                        "rrf_score": 0.0,
                        "best_lexical_score": 0.0,
                        "snippet": str(item.get("snippet", "")),
                    },
                )
                record["constraint_support"] += 1
                support_key = (
                    "direct_constraint_support" if is_direct else "bridge_constraint_support"
                )
                record[support_key] += 1
                record["rrf_score"] += 1.0 / (20.0 + rank)
                lexical_score = float(item.get("score", 0.0) or 0.0)
                if lexical_score > record["best_lexical_score"]:
                    record["best_lexical_score"] = lexical_score
                    record["snippet"] = str(item.get("snippet", ""))

        fused_ranked = sorted(
            fused.values(),
            key=lambda item: (
                -int(item["constraint_support"]),
                -float(item["rrf_score"]),
                -float(item["best_lexical_score"]),
                str(item["docid"]),
            ),
        )

        # Preserve clue coverage instead of letting broad, omnibus documents occupy
        # every slot. One best document per independent clause comes first, followed by
        # cross-clause fusion winners and a second document per clause for redundancy.
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        def add(docid: str) -> None:
            if docid in selected_ids or docid not in fused:
                return
            selected.append(fused[docid])
            selected_ids.add(docid)

        direct_fused = sorted(
            fused.values(),
            key=lambda item: (
                -int(item["direct_constraint_support"]),
                -float(item["rrf_score"]),
                -float(item["best_lexical_score"]),
                str(item["docid"]),
            ),
        )
        if direct_fused and int(direct_fused[0]["direct_constraint_support"]) > 1:
            add(str(direct_fused[0]["docid"]))

        ordered_by_clause = [
            sorted(
                ranking.items(),
                key=lambda item: (
                    item[1][0],
                    -float(item[1][1].get("score", 0.0) or 0.0),
                    item[0],
                ),
            )
            for ranking in clause_rankings
        ]
        ordered_bridges = [
            sorted(
                ranking.items(),
                key=lambda item: (
                    item[1][0],
                    -float(item[1][1].get("score", 0.0) or 0.0),
                    item[0],
                ),
            )
            for ranking in bridge_rankings
        ]
        if ordered_by_clause:
            most_specific = max(
                range(len(ordered_by_clause)),
                key=lambda index: len(
                    set(re.findall(r"[a-z0-9]+", queries[index].lower()))
                ),
            )
            for docid, _ in ordered_by_clause[most_specific][:5]:
                add(docid)
        for index, ranking in enumerate(ordered_by_clause):
            if ranking and index != most_specific:
                add(ranking[0][0])
        for item in fused_ranked:
            add(str(item["docid"]))
            if len(selected) >= 10:
                break
        for ranking in ordered_bridges:
            if ranking:
                add(ranking[0][0])
            if len(selected) >= 14:
                break
        for ranking in ordered_by_clause:
            if len(ranking) > 1:
                add(ranking[1][0])
            if len(selected) >= 14:
                break
        return selected[:16]

    copied.append(
        {
            "name": "constraint_search",
            "description": (
                "Search several independent clue clauses and rank documents by cross-clause "
                "support using reciprocal-rank fusion. Prefer this for multi-constraint entity "
                "identification. Supply 3-6 narrow clauses, not paraphrases of one broad query. "
                "Numeric ranges are expanded with a bounded strategy automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 6,
                    }
                },
                "required": ["queries"],
                "additionalProperties": False,
            },
            "handler": _constraint_search,
        }
    )
    return copied
