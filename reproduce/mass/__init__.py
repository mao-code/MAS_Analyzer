"""Lightweight MASS-style framework for reproduction experiments."""

from .adapters import TemplateBenchmarkAdapter
from .executor import MASSCandidateExecutor
from .framework import MASSFramework
from .interfaces import (
    BenchmarkAdapter,
    BenchmarkExample,
    OptimizerProtocol,
)
from .models import (
    AgentPromptBundle,
    CandidateEvaluation,
    MASSCandidate,
    MASSConfig,
    SearchSpace,
    StageResult,
    WorkflowSpec,
)
from .optimizer import (
    DSPyMIPROAdapter,
    IdentityPromptOptimizer,
    MIPROLikeConfig,
    MIPROLikePromptOptimizer,
)
from .runtime_runner import MASSRuntimeRunner

__all__ = [
    "AgentPromptBundle",
    "BenchmarkAdapter",
    "BenchmarkExample",
    "CandidateEvaluation",
    "DSPyMIPROAdapter",
    "IdentityPromptOptimizer",
    "MASSCandidateExecutor",
    "MASSCandidate",
    "MASSConfig",
    "MASSFramework",
    "MASSRuntimeRunner",
    "MIPROLikeConfig",
    "MIPROLikePromptOptimizer",
    "OptimizerProtocol",
    "SearchSpace",
    "StageResult",
    "TemplateBenchmarkAdapter",
    "WorkflowSpec",
]
