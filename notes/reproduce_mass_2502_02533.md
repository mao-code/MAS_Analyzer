# Reproduce Notes: MASS (`arXiv:2502.02533`)

Paper: [Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies](https://arxiv.org/abs/2502.02533)

## Status

- Working branch: `reproduce/arxiv-2502-02533`
- As of 2026-06-18, I did not find an official public code release for MASS itself.

## What the paper actually optimizes

MASS is a 3-stage search procedure:

1. Block-level prompt optimization
2. Workflow topology optimization
3. Workflow-level prompt optimization

The paper says their fixed topology construction order is:

`[summarize, reflect, debate, aggregate]`

and they use these search dimensions:

- `Summarize`: `N_s in {0,1,2,3,4}`
- `Aggregate`: `N_a in {1,3,5,7,9}`
- `Reflect`: `N_r in {0,1,2,3,4}`
- `Debate`: `N_d in {0,1,2,3,4}`
- `Execute`: `N_t in {0,1}`

## Reproduce-critical experiment specs

The paper reports using sampled validation/test subsets to save compute:

- `MATH`: val `60`, test `100`
- `DROP`: val `60`, test `200`
- `HotpotQA`: val `50`, test `100`
- `MuSiQue`: val `50`, test `100`
- `2WikiMQA`: val `50`, test `100`
- `MBPP`: val `60`, test `200`
- `HumanEval`: val `50`, test `100`
- `LiveCodeBench` test-output-prediction: val `100`, test `200`

Reported search spaces by task:

- `MATH`, `DROP`: `{Aggregate, Reflect, Debate}`
- `HotpotQA`, `MuSiQue`, `2WikiMQA`: `{Summarize, Aggregate, Reflect, Debate}`
- `MBPP`, `HumanEval`, `LiveCodeBench`: `{Aggregate, Reflect, Debate, Executor}`

Reported best MASS topologies on Gemini 1.5 Pro:

- `MATH`: `{9, 0, 0}`
- `DROP`: `{5, 0, 0}`
- `HotpotQA`: `{0, 5, 0, 1}`
- `MuSiQue`: `{0, 3, 0, 2}`
- `2WikiMQA`: `{0, 3, 0, 1}`
- `MBPP`: `{1, 4, 0, 1}`
- `HumanEval`: `{1, 3, 0, 1}`
- `LiveCodeBench`: `{3, 1, 1, 1}`

## Baseline settings worth matching

From the appendix:

- `CoT`: zero-shot "think step by step"
- `SC`: `SC@9`, temperature `0.8`
- `Self-Refine`: max `5` reflection rounds
- `Multi-Agent Debate`: `3` agents, `3` debate rounds, then aggregator
- `ADAS`: `30` search rounds, validation repeated `3` times per round
- `AFlow`: `20` rounds, `5` validation runs per round, `k=3`

The paper also notes their AFlow reproduction used:

- AFlow's original optimizer with `Claude 3.5 Sonnet`
- `Gemini 1.5 Pro` as the executor during reproduction

So their AFlow comparison is not perfectly apples-to-apples.

## Resource links

### Primary paper pages

- arXiv abstract: https://arxiv.org/abs/2502.02533
- arXiv PDF: https://arxiv.org/pdf/2502.02533
- OpenReview PDF: https://openreview.net/pdf?id=I05H9RUzHB
- Google Research page: https://research.google/pubs/multi-agent-design-optimizing-agents-with-better-prompts-and-topologies/

### Closest implementation resources

- ADAS: https://github.com/ShengranHu/ADAS
- AFlow: https://github.com/FoundationAgents/AFlow
- GPTSwarm: https://github.com/metauto-ai/GPTSwarm
- DSPy MIPROv2 docs: https://dspy.ai/api/optimizers/MIPROv2/
- DSPy optimizer docs: https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md

### Benchmark and dataset resources

- LongBench: https://github.com/THUDM/LongBench
- LiveCodeBench: https://github.com/livecodebench/livecodebench
- LiveCodeBench project page: https://livecodebench.github.io/
- HumanEval: https://github.com/openai/human-eval
- MBPP: https://github.com/google-research/google-research/tree/master/mbpp
- HotpotQA: https://aclanthology.org/D18-1259/
- MuSiQue: https://aclanthology.org/2022.tacl-1.31/
- 2WikiMultiHopQA: https://aclanthology.org/2020.coling-main.580/
- DROP: https://aclanthology.org/N19-1246/
- MATH: https://arxiv.org/abs/2103.03874

## Practical starting plan for this repo

Given this repository does not already expose `MATH`, `DROP`, `LongBench`, `MBPP`, `HumanEval`, or `LiveCodeBench` runners, the fastest path is:

1. Reproduce the coding subset first: `MBPP` and `HumanEval`
2. Implement only the paper's executor + reflector + aggregator pattern
3. Add prompt optimization later, after a fixed-topology baseline runs end-to-end
4. Use `AFlow`, `ADAS`, and `DSPy/MIPRO` as implementation references, not as ground truth

Why start there:

- the executor loop is clearly specified in the paper
- MBPP/HumanEval have accessible public tests
- the paper explicitly says it uses public MBPP/HumanEval tests for executor feedback

## My recommendation

If we want a realistic first milestone, target this:

- single-task reproduce on `HumanEval`
- one fixed topology first: `{Aggregate=1, Reflect=3, Debate=0, Execute=1}`
- then add Stage 1 prompt optimization
- then decide whether to implement full MASS search or a narrower ablation
