# BrowseComp Benchmark Adapter

This folder contains the BrowseComp-Plus benchmark integration for MAS Analyzer.

## What This Benchmark Does

- Uses queries from `Tevatron/browsecomp-plus` (OpenAI BrowseComp derived).
- Runs MAS/SAS agents on each query.
- Evaluates outputs by:
  - answer correctness (`substring` or `llm_judge`),
  - retrieval recall against evidence/gold qrels,
  - citation precision/recall from cited docids in final answer.

## Folder Layout

- `__init__.py`: adapter implementation and evaluation logic.
- `data/browsecomp_plus_decrypted.jsonl`: decrypted benchmark dataset (local).
- `topics-qrels/qrel_evidence.txt`: evidence relevance labels.
- `topics-qrels/qrel_golds.txt`: gold relevance labels.

## Tools Exposed To Agents

When `browsecomp.enable_tools = true`, the adapter provides per-task tools:

- `search(query)`: returns top-k doc snippets with `docid`, `score`, `snippet`.
- `get_document(docid)`: returns a single document payload by id (optional).

These are tracked in traces and run metadata:

- trace events: `tool_call`, `tool_result`
- run metadata: `tool_call_counts`, `tool_calls_total`, `retrieved_docids`

## Evaluation Modes

- `substring`:
  - fast, no judge API cost
  - lower-fidelity correctness proxy
- `llm_judge`:
  - uses official-style grader prompt
  - requires valid judge model/API setup

## Typical Config

```toml
[browsecomp]
decrypted_path = "benchmark/browsecomp/data/browsecomp_plus_decrypted.jsonl"
qrel_evidence_path = "benchmark/browsecomp/topics-qrels/qrel_evidence.txt"
qrel_golds_path = "benchmark/browsecomp/topics-qrels/qrel_golds.txt"
eval_mode = "substring"
enable_tools = true
tool_k = 5
include_get_document = true
tool_snippet_max_tokens = 512
max_tool_iterations = 8
```

## Official Assets

With `browsecomp.auto_download = true` (the default) the adapter fetches and decrypts the
dataset from Hugging Face on first use, writing it to `browsecomp.decrypted_path`. Set
`HF_TOKEN` in `.env` if the source repo requires authentication.

The qrel files in `topics-qrels/` are vendored in this folder and come from
[`texttron/BrowseComp-Plus`](https://github.com/texttron/BrowseComp-Plus); refresh them from
upstream if the benchmark is revised.

## Run Helper Script

You can run SAS or MAS directly with:

```bash
bash scripts/run_browsecomp_experiment.sh --mode sas --task-limit 3
bash scripts/run_browsecomp_experiment.sh --mode mas --number-of-agents 6 --levels 3
```
