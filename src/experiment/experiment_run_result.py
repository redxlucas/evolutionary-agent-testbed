from dataclasses import dataclass

from evolution import GenerationMetrics


@dataclass
class ExperimentRunResult:
    """Metrics and final-population diagnostics for one seeded run."""

    seed: int
    metrics: list[GenerationMetrics]
    diagnostics: dict[str, object] | None = None
