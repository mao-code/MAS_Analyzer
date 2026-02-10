from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OpenRouterConfig:
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str | None = None
    x_title: str | None = None
    timeout_s: float = 60.0

    def validate(self) -> None:
        if not self.base_url:
            raise ValueError("openrouter.base_url must be a non-empty string")
        if self.timeout_s <= 0:
            raise ValueError("openrouter.timeout_s must be > 0")


@dataclass
class MASConfig:
    levels: int
    intra_level_link_ratio: float
    full_linked: bool
    number_of_agents: int | None = None
    agents_per_level: List[int] | None = None
    agent_types: List[str] = field(default_factory=lambda: ["general"])
    communication_count_internally: int = 1
    turn_mode: str = "single_turn"
    max_turns: int = 1

    def validate(self) -> None:
        if self.levels < 1:
            raise ValueError("mas.levels must be >= 1")

        if not 0.0 <= self.intra_level_link_ratio <= 1.0:
            raise ValueError("mas.intra_level_link_ratio must be between 0.0 and 1.0")

        if self.communication_count_internally < 0:
            raise ValueError("mas.communication_count_internally must be >= 0")

        if self.turn_mode not in {"single_turn", "multi_turn"}:
            raise ValueError("mas.turn_mode must be one of: single_turn, multi_turn")

        if self.max_turns < 1:
            raise ValueError("mas.max_turns must be >= 1")

        if self.agents_per_level is not None:
            if len(self.agents_per_level) != self.levels:
                raise ValueError(
                    "mas.agents_per_level length must equal mas.levels "
                    f"({len(self.agents_per_level)} != {self.levels})"
                )
            for idx, count in enumerate(self.agents_per_level):
                if count < 1:
                    raise ValueError(
                        f"mas.agents_per_level[{idx}] must be >= 1 (got {count})"
                    )

        if self.number_of_agents is not None and self.number_of_agents < 1:
            raise ValueError("mas.number_of_agents must be >= 1 when provided")

        if self.agents_per_level is None and self.number_of_agents is None:
            raise ValueError(
                "Either mas.number_of_agents or mas.agents_per_level must be provided"
            )

        if self.agents_per_level is not None and self.number_of_agents is not None:
            total = sum(self.agents_per_level)
            if total != self.number_of_agents:
                raise ValueError(
                    "mas.number_of_agents does not match sum(mas.agents_per_level) "
                    f"({self.number_of_agents} != {total})"
                )

        if not self.agent_types:
            raise ValueError("mas.agent_types must contain at least one agent type")
        if any(not item.strip() for item in self.agent_types):
            raise ValueError("mas.agent_types entries must be non-empty strings")

    @property
    def total_agents(self) -> int:
        if self.agents_per_level is not None:
            return int(sum(self.agents_per_level))
        assert self.number_of_agents is not None
        return int(self.number_of_agents)

    def resolved_agents_per_level(self) -> List[int]:
        if self.agents_per_level is not None:
            return list(self.agents_per_level)

        assert self.number_of_agents is not None
        base = self.number_of_agents // self.levels
        remainder = self.number_of_agents % self.levels
        per_level = [base] * self.levels
        for idx in range(remainder):
            per_level[idx] += 1
        for idx, count in enumerate(per_level):
            if count < 1:
                raise ValueError(
                    "Derived mas.agents_per_level contains a zero-sized level at index "
                    f"{idx}. Increase mas.number_of_agents or lower mas.levels."
                )
        return per_level


@dataclass
class ExperimentRuntimeConfig:
    output_dir: str = "outputs"
    runs_per_task: int = 3
    seed: int = 42
    task_limit: int | None = None

    def validate(self) -> None:
        if self.runs_per_task < 1:
            raise ValueError("experiment.runs_per_task must be >= 1")
        if self.seed < 0:
            raise ValueError("experiment.seed must be >= 0")
        if self.task_limit is not None and self.task_limit < 1:
            raise ValueError("experiment.task_limit must be >= 1 when provided")


@dataclass
class ExperimentConfig:
    openrouter: OpenRouterConfig
    mas: MASConfig
    experiment: ExperimentRuntimeConfig
    models: Dict[str, str] = field(default_factory=dict)
    browsecomp: Dict[str, Any] = field(default_factory=dict)
    finance_agent: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        self.openrouter.validate()
        self.mas.validate()
        self.experiment.validate()

        if "default" not in self.models:
            raise ValueError("models.default is required in the config")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    data = tomllib.loads(path.read_text(encoding="utf-8"))

    openrouter_raw = _as_dict(data.get("openrouter"), "[openrouter]")
    experiment_raw = _as_dict(data.get("experiment"), "[experiment]")
    mas_raw = _as_dict(data.get("mas"), "[mas]")
    models_raw = _as_dict(data.get("models"), "[models]")

    env_api_key = os.getenv("OPENROUTER_API_KEY")
    api_key = env_api_key if env_api_key else _opt_str(openrouter_raw.get("api_key"))

    openrouter = OpenRouterConfig(
        api_key=api_key,
        base_url=_opt_str(openrouter_raw.get("base_url"))
        or "https://openrouter.ai/api/v1",
        http_referer=_opt_str(openrouter_raw.get("http_referer")),
        x_title=_opt_str(openrouter_raw.get("x_title")),
        timeout_s=float(openrouter_raw.get("timeout_s", 60.0)),
    )

    agents_per_level = mas_raw.get("agents_per_level")
    if agents_per_level is not None:
        if not isinstance(agents_per_level, list):
            raise ValueError("mas.agents_per_level must be a list of integers")
        parsed_agents_per_level = [int(value) for value in agents_per_level]
    else:
        parsed_agents_per_level = None

    agent_types = mas_raw.get("agent_types", ["general"])
    if isinstance(agent_types, str):
        agent_types = [agent_types]
    if not isinstance(agent_types, list):
        raise ValueError("mas.agent_types must be a list of strings")

    mas = MASConfig(
        levels=int(mas_raw.get("levels", 1)),
        intra_level_link_ratio=float(mas_raw.get("intra_level_link_ratio", 1.0)),
        full_linked=bool(mas_raw.get("full_linked", True)),
        number_of_agents=(
            int(mas_raw["number_of_agents"]) if "number_of_agents" in mas_raw else None
        ),
        agents_per_level=parsed_agents_per_level,
        agent_types=[str(item) for item in agent_types],
        communication_count_internally=int(mas_raw.get("communication_count_internally", 1)),
        turn_mode=str(mas_raw.get("turn_mode", "single_turn")),
        max_turns=int(mas_raw.get("max_turns", 1)),
    )

    experiment = ExperimentRuntimeConfig(
        output_dir=str(experiment_raw.get("output_dir", "outputs")),
        runs_per_task=int(experiment_raw.get("runs_per_task", 3)),
        seed=int(experiment_raw.get("seed", 42)),
        task_limit=(
            int(experiment_raw["task_limit"]) if "task_limit" in experiment_raw else None
        ),
    )

    models = {str(key): str(value) for key, value in models_raw.items()}

    config = ExperimentConfig(
        openrouter=openrouter,
        mas=mas,
        experiment=experiment,
        models=models,
        browsecomp=_as_dict(data.get("browsecomp", {}), "[browsecomp]"),
        finance_agent=_as_dict(data.get("finance_agent", {}), "[finance_agent]"),
    )
    config.validate()
    return config


def _as_dict(value: Any, section_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{section_name} must be a table in TOML")
    return dict(value)


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value
