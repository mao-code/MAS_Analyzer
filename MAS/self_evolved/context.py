"""Shared context controller: deterministic visibility infrastructure.

Generalizes the per-topology ``message_selector`` lambdas in the LangGraph
engine: every read goes through ``visible_packets`` which applies structural
filtering (recipient/kind/round, latest-per-sender) plus the agent's
``ContextPolicy`` — sender share-scope enforcement, summary-only compaction,
and per-reader packet bounds. The controller also keeps an append-only
evidence ledger so synthesis stages always see a bounded digest of branch or
global evidence (defense against "evidence lost before synthesis").
Visibility is decided in code, never by prompts.
"""

from __future__ import annotations

import re
from typing import Any

from ..artifacts import RelayPacket, _bounded_text, packet_content
from .spec import ContextPolicy, TopologySpec

DIGEST_SENDER = "shared_context"
DIGEST_KIND = "evidence_digest"
_DIGEST_MAX_ENTRIES = 12
_DIGEST_CLAIM_CHARS = 160
_DIGEST_EVIDENCE_CHARS = 100
_DIGEST_EVIDENCE_ITEMS = 3
_DEFAULT_READER_CHARS = 320


class SharedContextController:
    def __init__(self, spec: TopologySpec) -> None:
        self._spec = spec

    @property
    def spec(self) -> TopologySpec:
        return self._spec

    def set_spec(self, spec: TopologySpec) -> None:
        """Re-point visibility at a mutated spec; reads are lazy so nothing
        else needs to be recomputed."""

        self._spec = spec

    def visible_packets(
        self,
        state: dict[str, Any],
        *,
        agent_id: str,
        kinds: set[str] | None = None,
        round_index: int | None = None,
        discussion_index: int | None = None,
        latest_only: bool = True,
    ) -> list[RelayPacket]:
        policy = self._spec.agent(agent_id).context

        allowed_kinds = kinds
        if policy.visible_kinds:
            policy_kinds = set(policy.visible_kinds)
            allowed_kinds = policy_kinds if kinds is None else (kinds & policy_kinds)

        allowed_senders = self._resolve_senders(agent_id, policy.visible_from)

        selected: list[RelayPacket] = []
        for message in state.get("messages", []):
            if agent_id not in message.get("recipients", []):
                continue
            if allowed_kinds is not None and str(message.get("kind", "")) not in allowed_kinds:
                continue
            if round_index is not None and int(message.get("round", -1)) != round_index:
                continue
            if (
                discussion_index is not None
                and int(message.get("discussion_index", -1)) != discussion_index
            ):
                continue
            sender = str(message.get("sender", ""))
            if allowed_senders is not None and sender not in allowed_senders:
                continue
            if not self._share_scope_allows(sender, agent_id):
                continue
            selected.append(message)

        if not latest_only:
            return self._apply_reader_bounds(sorted(selected, key=_message_sort_key), policy)

        latest: dict[tuple[str, str], RelayPacket] = {}
        for message in selected:
            key = (str(message.get("sender", "")), str(message.get("kind", "")))
            current = latest.get(key)
            if current is None or _message_sort_key(message) >= _message_sort_key(current):
                latest[key] = message
        return self._apply_reader_bounds(sorted(latest.values(), key=_message_sort_key), policy)

    # -- evidence ledger -------------------------------------------------------

    def record_evidence(self, state: dict[str, Any], artifact: Any, *, agent_id: str) -> None:
        """Append a bounded claim+evidence entry for one produced artifact."""

        if artifact is None:
            return
        raw_evidence = artifact.get("evidence_summary", [])
        evidence = [
            _bounded_text(item, max_chars=_DIGEST_EVIDENCE_CHARS)
            for item in (raw_evidence if isinstance(raw_evidence, list) else [])[
                :_DIGEST_EVIDENCE_ITEMS
            ]
            if str(item or "").strip()
        ]
        claim = _bounded_text(
            artifact.get("summary", "") or artifact.get("answer", ""),
            max_chars=_DIGEST_CLAIM_CHARS,
        )
        if not claim and not evidence:
            return
        state.setdefault("evidence_ledger", []).append(
            {
                "agent_id": str(agent_id),
                "group_id": str(self._spec.agent(agent_id).group_id),
                # Packet-style key ("round", like RelayPacket), not "round_index".
                "round": int(state.get("round_index", 0)),
                "artifact_id": str(artifact.get("artifact_id", "")),
                "claim": claim,
                "evidence": evidence,
                "confidence": float(artifact.get("confidence", 0.5)),
            }
        )

    def evidence_digest_packet(
        self,
        state: dict[str, Any],
        *,
        agent_id: str,
        scope: str | None = None,
    ) -> RelayPacket | None:
        """Bounded ledger digest for one reader; None when out of scope/empty.

        ``scope`` overrides the reader's ``evidence_access`` policy — used by
        aggregator stages, which always receive at least their branch view.
        The packet is prompt-only context: it is never emitted into
        ``state["messages"]``.
        """

        effective = scope or self._spec.agent(agent_id).context.evidence_access
        if effective == "own":
            return None
        ledger = list(state.get("evidence_ledger", []))
        if effective == "branch":
            branch = self.branch_agent_ids(agent_id)
            ledger = [entry for entry in ledger if str(entry.get("agent_id", "")) in branch]
        entries = ledger[-_DIGEST_MAX_ENTRIES:]
        if not entries:
            return None

        lines = []
        for entry in entries:
            line = f"{entry.get('agent_id', '')}[r{entry.get('round', 0)}]: " + str(
                entry.get("claim", "")
            )
            if entry.get("evidence"):
                line += " | evidence: " + "; ".join(str(item) for item in entry["evidence"])
            lines.append(line)
        content = f"Evidence ledger (scope={effective}, {len(entries)} entries): " + " || ".join(
            lines
        )
        round_index = int(state.get("round_index", 0))
        return {
            "message_id": f"digest_{agent_id}_r{round_index}_{len(ledger)}",
            "dispatch_id": int(state.get("dispatch_id", -1)),
            "sender": DIGEST_SENDER,
            "recipients": [agent_id],
            "kind": DIGEST_KIND,
            "phase": str(state.get("phase", "")),
            "round": round_index,
            "discussion_index": int(state.get("discussion_index", 0)),
            "artifact_id": None,
            "content": content,
            "payload": {"scope": effective, "entries": [dict(entry) for entry in entries]},
        }

    def branch_agent_ids(self, agent_id: str) -> set[str]:
        """Agents in the same root branch: walk up to the root-group anchor,
        then collect its entire attached subtree."""

        spec = self._spec
        top = agent_id
        seen = {top}
        while True:
            group = spec.group(spec.agent(top).group_id)
            parent = group.parent_agent_id
            if parent is None or parent in seen:
                break
            seen.add(parent)
            top = parent

        resolved: set[str] = set()
        frontier = [top]
        while frontier:
            current = frontier.pop()
            if current in resolved:
                continue
            resolved.add(current)
            subgroup = spec.subgroup_of(current)
            if subgroup is not None:
                frontier.extend(subgroup.member_ids)
        return resolved

    # -- policy application ----------------------------------------------------

    def _share_scope_allows(self, sender: str, reader: str) -> bool:
        """Defense-in-depth: the sender's share_scope bounds who may read its
        packets even if a packet was mis-addressed."""

        if sender == reader or not sender:
            return True
        try:
            sender_node = self._spec.agent(sender)
        except KeyError:
            return True  # non-spec senders (controller digests, meta agents)
        scope = sender_node.context.share_scope
        if scope == "global":
            return True
        group = self._spec.group(sender_node.group_id)
        parent_ids: set[str] = set()
        if group.parent_agent_id is not None:
            parent_ids.add(group.parent_agent_id)
        if group.leader_id and group.leader_id != sender:
            parent_ids.add(group.leader_id)
        if scope == "parent_only":
            return reader in parent_ids
        # scope == "group": same group, the attachment/leader parent, or the
        # sender's own subgroup children (task packets flow downward).
        if reader in group.member_ids or reader in parent_ids:
            return True
        subgroup = self._spec.subgroup_of(sender)
        return subgroup is not None and reader in subgroup.member_ids

    def _apply_reader_bounds(
        self, packets: list[RelayPacket], policy: ContextPolicy
    ) -> list[RelayPacket]:
        """summary_only compaction and per-reader max_packet_chars bounding.

        Returns compacted copies; the packets stored in state stay intact.
        """

        if not policy.summary_only and int(policy.max_packet_chars) <= 0:
            return packets
        max_chars = (
            int(policy.max_packet_chars)
            if int(policy.max_packet_chars) > 0
            else _DEFAULT_READER_CHARS
        )
        bounded: list[RelayPacket] = []
        for packet in packets:
            compacted = dict(packet)
            payload = dict(packet.get("payload", {}) or {})
            if policy.summary_only:
                summary = (
                    packet_content(payload, max_chars=max_chars)
                    if payload
                    else _bounded_text(packet.get("content", ""), max_chars=max_chars)
                )
                compact_payload = {"artifact_id": payload.get("artifact_id"), "summary": summary}
                if "confidence" in payload:
                    compact_payload["confidence"] = payload["confidence"]
                compacted["payload"] = compact_payload
                compacted["content"] = summary
            else:
                for key in ("summary", "answer_artifact", "critique", "revision_request"):
                    if payload.get(key):
                        payload[key] = _bounded_text(payload[key], max_chars=max_chars)
                compacted["payload"] = payload
                compacted["content"] = _bounded_text(packet.get("content", ""), max_chars=max_chars)
            bounded.append(compacted)
        return bounded

    def _resolve_senders(self, agent_id: str, visible_from: tuple[str, ...]) -> set[str] | None:
        if not visible_from:
            return None
        spec = self._spec
        resolved: set[str] = set()
        group = spec.group(spec.agent(agent_id).group_id)
        for token in visible_from:
            if token == "group":
                resolved.update(member for member in group.member_ids if member != agent_id)
            elif token == "parent":
                if group.parent_agent_id is not None:
                    resolved.add(group.parent_agent_id)
                elif group.leader_id and group.leader_id != agent_id:
                    resolved.add(group.leader_id)
            elif token == "children":
                subgroup = spec.subgroup_of(agent_id)
                if subgroup is not None:
                    resolved.update(subgroup.member_ids)
            else:
                resolved.add(token)
        return resolved


def _message_sort_key(message: RelayPacket) -> tuple[int, int, int]:
    match = re.search(r"(\d+)$", str(message.get("message_id", "m_0")))
    return (
        int(message.get("round", 0)),
        int(message.get("discussion_index", 0)),
        int(match.group(1)) if match else 0,
    )
