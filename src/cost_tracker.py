"""
Cost tracking and projection.

Real Gemini API responses include `usage_metadata` (prompt_token_count,
candidates_token_count) — we use that whenever it's available, since it's
the actual billed token count. For calls where the SDK doesn't return
usage (e.g. the embedding endpoint), we fall back to a rough ~4-chars-per-
token estimate and mark that event as `estimated: true` so it's clear
which numbers are exact vs. approximate.

Pricing is hardcoded from the published Gemini API rates as of Jul 2026
(https://ai.google.dev/gemini-api/docs/pricing). Prices change — re-check
that page and update GEMINI_PRICING if your numbers look stale.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# $ per 1,000,000 tokens.
GEMINI_PRICING: Dict[str, Dict[str, float]] = {
    # Gemini 2.5 family — deprecated for new API keys as of June 17, 2026.
    # Kept here only so old cost logs / existing users on this tier still price correctly.
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # Current (Jul 2026) GA lineup.
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},  # cheapest current multimodal model — good default
    "gemini-3-flash": {"input": 0.50, "output": 3.00},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},       # frontier-tier Flash, ~6x the cost of 3.1 Flash-Lite
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    # Embeddings have historically been free on the Gemini API — verify at
    # the pricing link above before relying on this for a large-scale bill.
    "models/text-embedding-004": {"input": 0.00, "output": 0.00},
}
DEFAULT_PRICING = {"input": 0.25, "output": 1.50}  # fall back to gemini-3.1-flash-lite rates if model unknown


def _rough_token_estimate(text: str) -> int:
    """~4 characters/token, used only when a real usage_metadata isn't available."""
    return max(1, len(text) // 4)


@dataclass
class CostEvent:
    stage: str  # "vision_extraction" | "embedding" | "correction"
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    estimated: bool = False  # True if token counts were estimated, not read from usage_metadata


@dataclass
class CostTracker:
    """Accumulates CostEvents for a single image's run through the pipeline."""

    image_path: Optional[str] = None
    events: List[CostEvent] = field(default_factory=list)

    def record(self, stage: str, model: str, input_tokens: int, output_tokens: int, estimated: bool = False) -> CostEvent:
        pricing = GEMINI_PRICING.get(model, DEFAULT_PRICING)
        cost = (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
        event = CostEvent(
            stage=stage,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 8),
            estimated=estimated,
        )
        self.events.append(event)
        return event

    def record_from_response(self, stage: str, model: str, response) -> Optional[CostEvent]:
        """Pull real token counts from a Gemini response's usage_metadata, if present."""
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return None
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        return self.record(stage, model, input_tokens, output_tokens, estimated=False)

    def record_estimated(self, stage: str, model: str, input_text: str, output_text: str = "") -> CostEvent:
        return self.record(
            stage,
            model,
            input_tokens=_rough_token_estimate(input_text),
            output_tokens=_rough_token_estimate(output_text),
            estimated=True,
        )

    @property
    def total_cost_usd(self) -> float:
        return round(sum(e.cost_usd for e in self.events), 8)

    def breakdown_by_stage(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for e in self.events:
            out[e.stage] = round(out.get(e.stage, 0.0) + e.cost_usd, 8)
        return out

    def to_dict(self) -> Dict:
        return {
            "image_path": self.image_path,
            "total_cost_usd": self.total_cost_usd,
            "breakdown_by_stage": self.breakdown_by_stage(),
            "events": [asdict(e) for e in self.events],
        }

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def project_cost_at_scale(avg_cost_per_image: float, volumes=(1_000, 10_000, 100_000)) -> Dict[str, float]:
    return {str(v): round(avg_cost_per_image * v, 2) for v in volumes}


def summarize_costs(cost_dicts: List[Dict]) -> Dict:
    """Aggregate a list of per-image cost dicts (as produced by CostTracker.to_dict())."""
    if not cost_dicts:
        return {"num_images": 0, "avg_cost_per_image_usd": 0.0, "projection_usd": {}}

    total = sum(c["total_cost_usd"] for c in cost_dicts)
    avg = total / len(cost_dicts)

    stage_totals: Dict[str, float] = {}
    for c in cost_dicts:
        for stage, cost in c.get("breakdown_by_stage", {}).items():
            stage_totals[stage] = round(stage_totals.get(stage, 0.0) + cost, 8)

    return {
        "num_images": len(cost_dicts),
        "total_cost_usd": round(total, 6),
        "avg_cost_per_image_usd": round(avg, 6),
        "breakdown_by_stage_total_usd": stage_totals,
        "projection_usd": project_cost_at_scale(avg),
    }
