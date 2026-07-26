# Upstream Attribution

This ADAS reproduction adapter is based on the method and public implementation
from:

- Repository: `https://github.com/ShengranHu/ADAS`
- Paper: `Automated Design of Agentic Systems`
- Upstream commit inspected: `2702bee8fefda42255efc5be9f60e3bd3db96ae4`
- Upstream license: Apache License 2.0

The original repository organizes Meta Agent Search as separate domain-specific
scripts. This adapter reimplements the method for MAS_Analyzer's benchmark
interfaces and artifact contract. It preserves the core algorithmic structure:
archive initialization, meta-agent code proposal, two reflexion prompts,
validation scoring, debug refinement, archive update, and final selected-agent
evaluation.

Generated agent code is executed under a restricted namespace and routed through
MAS_Analyzer's existing OpenRouter client, benchmark tools, judges, checkpointing,
and trace serialization.
