import json
import logging
import os
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_PRICING_PATH = Path(__file__).parent.parent / "config" / "model_pricing.json"


def _to_decimal(value) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError(f"Invalid price value: {value!r}")


@lru_cache(maxsize=1)
def load_model_pricing() -> dict[str, dict[str, Decimal]]:
    pricing_path = Path(os.getenv("MODEL_PRICING_PATH", str(_DEFAULT_PRICING_PATH)))
    if not pricing_path.exists():
        logger.warning("Model pricing file not found: %s", pricing_path)
        return {}

    try:
        raw = json.loads(pricing_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load model pricing from %s: %s", pricing_path, exc)
        return {}

    pricing: dict[str, dict[str, Decimal]] = {}
    for model_name, values in raw.items():
        if not isinstance(values, dict):
            logger.warning("Skipping malformed pricing entry for model %s", model_name)
            continue
        try:
            input_price = _to_decimal(values["input_per_1m_tokens"])
            output_price = _to_decimal(values["output_per_1m_tokens"])
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping invalid pricing entry for model %s: %s", model_name, exc)
            continue
        pricing[model_name] = {
            "input": input_price,
            "output": output_price,
        }
    return pricing


def calculate_cost_for_model(model_name: str, input_tokens: int, output_tokens: int) -> float:
    pricing = load_model_pricing().get(model_name)
    if not pricing:
        logger.warning("No pricing configured for model: %s", model_name)
        return 0.0

    total = (
        Decimal(int(input_tokens)) * pricing["input"]
        + Decimal(int(output_tokens)) * pricing["output"]
    ) / Decimal("1000000")
    return float(total)


def reset_pricing_cache() -> None:
    load_model_pricing.cache_clear()
