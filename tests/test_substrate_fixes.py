"""Regression tests for substrate / bias fixes.

Covers three fixes that affect every topology arm equally and therefore matter for
unbiased topology comparison:

1. ``build_artifact`` answer/summary fallback chains (operator-precedence fix).
2. Search-snippet budget in the tool-loop compaction (evidence-starvation fix).
3. Recency-correct tie-break in ``latest_artifact_by_agent`` (determinism fix).
"""

import json
import os
import types
import unittest
from unittest import mock

from cli.resume import _llm_payload_needs_rerun, _metadata_needs_rerun
from MAS.artifacts import ArtifactRecord, build_artifact, latest_artifact_by_agent
from MAS.config import OpenRouterConfig
from MAS.llm import OpenRouterLLMClient


def _artifact(text: str) -> ArtifactRecord:
    return build_artifact(
        text=text,
        artifact_id="a",
        dispatch_id=0,
        node_name="n",
        stage_role="worker",
        round_index=0,
        discussion_index=0,
        agent_id="agent_0",
        role="debater",
        source_artifact_ids=[],
        tool_records=[],
        llm_payload={},
    )


class TestBuildArtifactFallbacks(unittest.TestCase):
    def test_canonical_answer_artifact_and_summary(self) -> None:
        art = _artifact(json.dumps({"answer_artifact": "Paris", "summary": "the capital"}))
        self.assertEqual(art["answer"], "Paris")
        self.assertEqual(art["summary"], "the capital")
        self.assertEqual(art["status"], "ok")

    def test_legacy_answer_key_is_used(self) -> None:
        # Model drifted to the legacy "answer" key instead of "answer_artifact".
        # The precedence bug used to leave answer empty and fall back to the raw
        # JSON blob; now the "answer" key is honored.
        art = _artifact(json.dumps({"answer": "Berlin"}))
        self.assertEqual(art["answer"], "Berlin")
        self.assertEqual(art["summary"], "Berlin")

    def test_missing_summary_falls_back_to_answer(self) -> None:
        art = _artifact(json.dumps({"answer_artifact": "Tokyo"}))
        self.assertEqual(art["answer"], "Tokyo")
        self.assertEqual(art["summary"], "Tokyo")  # not "" anymore

    def test_non_json_text_is_fallback_status(self) -> None:
        art = _artifact("just plain text answer")
        self.assertEqual(art["answer"], "just plain text answer")
        self.assertEqual(art["status"], "fallback")

    def test_structured_answer_artifact_is_serialized_as_json(self) -> None:
        art = _artifact(
            json.dumps(
                {
                    "answer_artifact": {
                        "categories": ["Electronics"],
                        "products": [{"id": 893292, "name": "TV"}],
                    }
                }
            )
        )

        self.assertEqual(
            json.loads(art["answer"]),
            {"categories": ["Electronics"], "products": [{"id": 893292, "name": "TV"}]},
        )


class TestSearchSnippetBudget(unittest.TestCase):
    def _client(self) -> OpenRouterLLMClient:
        return OpenRouterLLMClient(OpenRouterConfig(), {"default": "openai/gpt-4o-mini"})

    def test_search_snippet_is_not_starved(self) -> None:
        client = self._client()
        long_snippet = "word " * 1000  # ~5000 chars
        compact = client._compact_tool_output_for_prompt(
            {"docid": "d1", "score": 1.0, "snippet": long_snippet},
            preview_chars=160,
        )
        preview = compact["snippet_preview"]
        # Old behavior capped this at ~160 chars; the agent that ran the search must
        # see far more of its own retrieval evidence.
        self.assertGreater(len(preview), 1000)
        self.assertEqual(compact["docid"], "d1")

    def test_document_text_budget_unchanged(self) -> None:
        # get_document outputs carry a "text" key and keep the large document budget.
        client = self._client()
        compact = client._compact_tool_output_for_prompt(
            {"docid": "d2", "text": "T" * 20000},
            preview_chars=160,
        )
        self.assertGreater(len(compact["text_preview"]), 5000)


class TestOpenRouterSamplingOverrides(unittest.TestCase):
    def _client(self) -> OpenRouterLLMClient:
        return OpenRouterLLMClient(OpenRouterConfig(), {"default": "openai/gpt-4o-mini"})

    def test_completion_ceiling_is_applied(self) -> None:
        client = self._client()
        with mock.patch.dict(os.environ, {"OPENROUTER_MAX_TOKENS": "1024"}, clear=False):
            request = client._apply_openrouter_sampling_overrides(
                {"model": "test-model"}, temperature=0.0
            )

        self.assertEqual(request["max_tokens"], 1024)

    def test_completion_ceiling_has_safe_default(self) -> None:
        client = self._client()
        with mock.patch.dict(os.environ, {}, clear=True):
            request = client._apply_openrouter_sampling_overrides(
                {"model": "test-model"}, temperature=0.0
            )

        self.assertEqual(request["max_tokens"], 4096)

    def test_nonpositive_completion_ceiling_is_rejected(self) -> None:
        client = self._client()
        with (
            mock.patch.dict(os.environ, {"OPENROUTER_MAX_TOKENS": "0"}, clear=False),
            self.assertRaisesRegex(ValueError, "must be >= 1"),
        ):
            client._apply_openrouter_sampling_overrides({"model": "test-model"}, temperature=0.0)


class TestLatestArtifactRecency(unittest.TestCase):
    def _mk(
        self, *, round_index: int, discussion: int, dispatch: int, node: str, answer: str
    ) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=f"{round_index}-{discussion}-{dispatch}-{node}",
            agent_id="agent_0",
            round_index=round_index,
            discussion_index=discussion,
            dispatch_id=dispatch,
            node_name=node,
            answer=answer,
        )

    def test_latest_by_ordering(self) -> None:
        arts = [
            self._mk(round_index=0, discussion=0, dispatch=0, node="z_initial", answer="first"),
            self._mk(round_index=0, discussion=1, dispatch=1, node="a_revision", answer="second"),
        ]
        latest = latest_artifact_by_agent(arts)
        # Newer (discussion=1) wins even though its node name sorts BEFORE the older
        # one alphabetically — the old lexical tie-break would have been fragile here.
        self.assertEqual(latest["agent_0"]["answer"], "second")

    def test_append_order_breaks_exact_ties(self) -> None:
        arts = [
            self._mk(round_index=0, discussion=0, dispatch=0, node="x", answer="older"),
            self._mk(round_index=0, discussion=0, dispatch=0, node="y", answer="newer"),
        ]
        latest = latest_artifact_by_agent(arts)
        self.assertEqual(latest["agent_0"]["answer"], "newer")


class TestEmptyCompletionContract(unittest.TestCase):
    """Provider empty/malformed completions must be retried across providers and,
    if still empty, recorded as a typed failure — never a silent empty artifact."""

    @staticmethod
    def _msg(content=None, tool_calls=None):
        return types.SimpleNamespace(content=content, tool_calls=tool_calls)

    @classmethod
    def _completion(cls, choices):
        return types.SimpleNamespace(
            choices=choices,
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=20),
        )

    @classmethod
    def _choice(cls, message):
        return types.SimpleNamespace(message=message)

    def _client(self, responses):
        client = OpenRouterLLMClient(OpenRouterConfig(), {"default": "m"})
        seq = list(responses)
        calls = {"n": 0}

        class _Chat:
            def create(self, **_kwargs):
                idx = min(calls["n"], len(seq) - 1)
                calls["n"] += 1
                return seq[idx]

        client.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Chat()))
        return client, calls

    def setUp(self) -> None:
        # Keep retry backoff tiny so the test is fast.
        self._sleep_patch = mock.patch("MAS.llm.time.sleep", lambda *_a, **_k: None)
        self._sleep_patch.start()

    def tearDown(self) -> None:
        self._sleep_patch.stop()

    def _generate(self, client):
        return client.generate(
            prompt="q", agent_type="default", task_id="t", run_index=0, agent_id="a"
        )

    def test_empty_then_usable_is_retried(self) -> None:
        empty = self._completion([])
        good = self._completion([self._choice(self._msg(content="Paris is the capital."))])
        client, calls = self._client([empty, good])
        result = self._generate(client)
        self.assertEqual(result.text, "Paris is the capital.")
        self.assertEqual(result.metadata.get("generation_status"), "answered")
        self.assertEqual(calls["n"], 2)  # retried once, recovered

    def test_persistent_empty_is_typed_failure(self) -> None:
        client, calls = self._client([self._completion([])])
        result = self._generate(client)
        self.assertEqual(result.text, "")
        self.assertEqual(result.metadata.get("generation_status"), "failed")
        self.assertGreaterEqual(calls["n"], 2)  # exhausted retries, no crash
        self.assertTrue(
            _llm_payload_needs_rerun(
                {
                    "mock_used": result.mock_used,
                    "metadata": result.metadata,
                }
            )
        )

    def test_nested_empty_completion_marks_run_for_rerun(self) -> None:
        run_metadata = {
            "run_status": "completed",
            "artifact_records": [
                {
                    "llm": {
                        "mock_used": False,
                        "metadata": {
                            "empty_completion": True,
                            "failure_category": "empty_completion",
                            "generation_status": "failed",
                        },
                    }
                }
            ],
        }

        self.assertTrue(_metadata_needs_rerun(run_metadata))

    def test_answered_completion_does_not_mark_run_for_rerun(self) -> None:
        payload = {
            "mock_used": False,
            "metadata": {
                "generation_status": "answered",
            },
        }

        self.assertFalse(_llm_payload_needs_rerun(payload))

    def test_usable_first_try_no_wasted_retries(self) -> None:
        client, calls = self._client(
            [self._completion([self._choice(self._msg(content="Berlin."))])]
        )
        result = self._generate(client)
        self.assertEqual(result.metadata.get("generation_status"), "answered")
        self.assertEqual(calls["n"], 1)

    def test_tool_call_turn_counts_as_usable(self) -> None:
        client, _ = self._client([self._completion([])])
        tc = types.SimpleNamespace(
            id="1",
            type="function",
            function=types.SimpleNamespace(name="search", arguments='{"query":"x"}'),
        )
        usable = self._completion([self._choice(self._msg(content="", tool_calls=[tc]))])
        self.assertTrue(client._completion_has_usable_choice(usable))
        self.assertFalse(client._completion_has_usable_choice(self._completion([])))


if __name__ == "__main__":
    unittest.main()
