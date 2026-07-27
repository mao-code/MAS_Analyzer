# AGENTS.md

Repository guidance for coding agents lives in a single file: **[CLAUDE.md](CLAUDE.md)**.

Read that file. It covers what this repo is, the commands, the architecture, and the
invariants that must not be broken (metric contract, prompt instruction priority,
relay-packet rules, termination ordering).

This pointer exists so agents that look for `AGENTS.md` by convention find the right
document. Do not duplicate the content here — the previous copy of it in this file silently
drifted out of date and started contradicting the real behaviour.
