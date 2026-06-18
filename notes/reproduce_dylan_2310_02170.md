# Reproduce Notes: DyLAN (`arXiv:2310.02170`)

Paper: [A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration](https://arxiv.org/abs/2310.02170)

Official repo: https://github.com/SALT-NLP/DyLAN

Branch prepared: `reproduce/dylan-2310-02170`

## What DyLAN Is

DyLAN is a dynamic LLM agent network with two major ideas:

- Team optimization: select useful agents using an unsupervised Agent Importance Score.
- Task solving: selected agents collaborate over multiple rounds with dynamic communication and early stopping.

The code calls the framework `LLMLP`, treating LLM agents like neurons in a multi-layer network.

## Official Repo Status

The official repo is available and has a single `main` branch.

Top-level structure:

- `code/demo`: quick demo
- `code/MATH`: MATH reproduction code
- `code/MMLU`: MMLU reproduction code
- `code/HumanEval`: HumanEval reproduction code
- `exp`: experiment records for verifying reported results

The README says existing experiment records can be verified with commands such as:

```bash
python code/MATH/eval_math.py exp/MATH/CoT None
python code/MATH/eval_math.py exp/MATH/Complex None
python code/MMLU/eval_mmlu.py exp/MMLU/mmlu_optimal7_7 None
```

## Dependencies

The repo uses older dependencies:

- Python-era code around `openai==0.27.6`
- `numpy==1.22.4`
- `pandas==1.5.3`
- `human-eval`

This should probably run in an isolated environment instead of this repo's current `.venv`.

## Fit For Our Benchmarks

DyLAN is not a generic plug-in framework in its current implementation. It is organized around MATH, MMLU, HumanEval, and a demo.

Best reproduction path for our benchmarks:

1. Keep official repo as reference implementation.
2. Extract the reusable LLMLP ideas:
   - fixed candidate role pool
   - multiple communication rounds
   - listwise ranking / activation
   - consensus early stopping
   - backward Agent Importance Score
3. Build a standalone adapter similar to the MASS runner, rather than importing the old OpenAI 0.27 code directly.

## First Experiment Recommendation

Start with a lightweight DyLAN-style runner over one existing benchmark:

- benchmark: `stabletoolbench` or `workbench`
- agents: 4
- rounds: 3
- role pool: repeated `Assistant` first, then domain roles later
- activation: listwise ranker
- stopping: consensus if more than two-thirds agree

This is more useful for our project than trying to force the official MATH/MMLU/HumanEval scripts into our benchmark interface.
