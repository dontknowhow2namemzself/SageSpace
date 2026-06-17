"""Question planner (pipeline node, PR4).

One cheap LLM call that decides whether a reader's question is COMPOUND
(asks about several distinct things that each deserve their own retrieval)
or SIMPLE (one thing), and, when compound, splits it into a few focused
sub-questions. The `retrieve` node then fans out one ReAct branch per
sub-question (design §3/§5), gathers all the evidence, and synthesizes a
single coherent answer.

Design guardrail (§8 Q3): ~90% of questions are simple and MUST pass
through untouched -- over-decomposing a simple question wastes latency and
fragments the answer. The prompt is biased hard toward "simple"; only a
genuinely multi-target question is split.

Cost: the planner also emits a rough `cost_estimate` (sub-question count +
ballpark retrieval calls / latency). The deterministic cost-confirm gate
that would consume it is DEFERRED this round (it only earns its keep once
expensive out-of-book tools like web search exist); the estimate is kept
because it is ~free and feeds the Token-Usage panel + a future gate.
"""
from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)

# Hard ceiling on fan-out width (bounds parallel retrieval + token cost).
MAX_SUBQUESTIONS = 4


class _PlanSchema(BaseModel):
    """Whether the question is compound, and its sub-questions."""

    is_compound: bool = Field(
        ...,
        description=(
            "True ONLY when the user asks about 2+ SEPARATE targets that each "
            "need their own search (e.g. 'Compare X and Y', 'What happens to A "
            "and what does B mean?'). A single broad topic ('main themes', 'how "
            "Alice changes') is NOT compound. When unsure, choose false."
        ),
    )
    subquestions: list[str] = Field(
        default_factory=list,
        description=(
            "When is_compound: 2-4 focused, self-contained sub-questions, each "
            "answerable on its own, in the user's language. When NOT compound: "
            "a single element -- the original question unchanged."
        ),
    )


_SYSTEM_PROMPT = """\
You split a reader's question for a book Q&A system into sub-questions for
retrieval -- but ONLY when it genuinely asks about multiple distinct things.

MOST questions are SIMPLE: return is_compound=false and subquestions=[the
original question, unchanged]. Be conservative -- a single broad topic is
still simple.

Mark is_compound=true ONLY for 2+ separate targets that would each be
searched differently, and split into 2-4 focused, self-contained
sub-questions (each answerable alone). Never exceed 4. Use the question's
language.

Examples:
  "Who is the Cheshire Cat?"                      -> simple
  "What are the main themes of the book?"         -> simple (one broad topic)
  "How does Alice change throughout the story?"   -> simple (one arc)
  "比较红心皇后和公爵夫人"                          -> COMPOUND
     -> ["红心皇后是谁，她是什么样的人?", "公爵夫人是谁，她是什么样的人?"]
  "What happens to the White Rabbit, and what does the Cheshire Cat mean?"
     -> COMPOUND
     -> ["What happens to the White Rabbit in the story?",
         "What does the Cheshire Cat symbolize?"]
"""


def _build_planner() -> ChatOpenAI:
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )


def decompose_question(question: str) -> dict:
    """Plan a question into sub-questions.

    Returns ``{"is_compound": bool, "subquestions": [str, ...]}``. Always
    non-empty (at least the original question). On any LLM failure, degrades
    to a SIMPLE single-sub-question plan -- a compound question merely loses
    its fan-out, never the turn.
    """
    if not question or not question.strip():
        return {"is_compound": False, "subquestions": [question or ""]}
    try:
        structured = _build_planner().with_structured_output(_PlanSchema)
        result = structured.invoke(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}"},
            ]
        )
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        return _coerce_plan(data, question)
    except Exception as exc:
        logger.warning("decompose_question failed: %s. Treating as simple.", str(exc)[:160])
        return {"is_compound": False, "subquestions": [question]}


def estimate_cost(n_subquestions: int) -> dict:
    """Rough, informational cost of fanning out `n` sub-questions. Retrieval
    runs in parallel, so latency grows sub-linearly; calls grow with width.
    Consumed by the (deferred) cost gate + the Token-Usage panel."""
    n = max(1, n_subquestions)
    return {
        "n_subq": n,
        # ~3 retrieval tool calls per sub-question (bounded by MAX_ITERATIONS).
        "est_calls": n * 3,
        # Parallel fan-out: ~one sub-question's latency + a small per-branch tax.
        "est_latency_s": 8 + 3 * (n - 1),
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _coerce_plan(data: dict, question: str) -> dict:
    """Normalize the model output: clean sub-questions, enforce the cap, and
    keep is_compound consistent with what actually survived."""
    raw = data.get("subquestions") or []
    subs = [str(s).strip() for s in raw if str(s).strip()]
    subs = subs[:MAX_SUBQUESTIONS]

    is_compound = bool(data.get("is_compound")) and len(subs) >= 2
    if not is_compound:
        # Simple (or the model marked compound but gave <2 usable parts):
        # collapse to the original question as the single sub-question.
        return {"is_compound": False, "subquestions": [question]}
    return {"is_compound": True, "subquestions": subs}
