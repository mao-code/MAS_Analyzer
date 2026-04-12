from __future__ import annotations


def classify_regime(gain_g: float, coordination_cost_k: float) -> str:
    """Classify collaboration regime in the gain-cost plane."""
    if coordination_cost_k < 0:
        return "efficiency_driven_behavior"
    if gain_g > 0 and gain_g > coordination_cost_k:
        return "productive_collaboration"
    if gain_g > 0 and gain_g < coordination_cost_k:
        return "overpriced_collaboration"
    if gain_g < 0 and coordination_cost_k > 0:
        return "wasteful_collaboration"
    return "boundary_or_mixed"
