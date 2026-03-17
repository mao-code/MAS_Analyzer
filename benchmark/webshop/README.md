# WebShop

Adapter for the official WebShop benchmark:
- repo: `https://github.com/princeton-nlp/WebShop`
- paper: `https://arxiv.org/abs/2207.01206`

What this adapter keeps:
- official product assets and human instructions
- the WebShop page/action loop (`search[...]`, `click[...]`)
- the reward structure used by the benchmark

What is different:
- it runs in-process inside `MAS_Analyzer`
- it avoids the upstream `pyserini` / old-`gym` stack so it works with this repo's runtime
- the default execution path uses the official `small` asset set for manageable smoke tests

## Config

```toml
[webshop]
data_mode = "small"   # or "full"
split = "test"
auto_download = true
human_goals = true
max_steps = 15
history_window = 4
```

## Notes

- `small` mode downloads the official 1k-product asset bundle.
- `full` mode downloads the full WebShop corpus and is much heavier on disk/runtime.
- Evaluation score is the final WebShop reward in `[0, 1]`.
- `success=true` means an exact successful purchase (`reward == 1.0`).
