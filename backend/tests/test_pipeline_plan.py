"""Tests for the question planner (PR4 step 1).

decompose_question must be CONSERVATIVE: simple questions pass straight
through (one sub-question = the original), and only a genuinely compound
question is split — capped, blank-filtered, and degrading to simple on any
LLM failure. Plus the plan_node graph wrapper (passthrough for non-agent
intents) and the informational cost estimate.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from core.graph import nodes as G
from core.pipeline import plan as plan_mod
from core.pipeline.plan import MAX_SUBQUESTIONS, decompose_question, estimate_cost
from core.pipeline.types import IntentDecision


def _patch(monkeypatch, payload):
    structured = MagicMock()
    if isinstance(payload, Exception):
        structured.invoke.side_effect = payload
    else:
        structured.invoke.return_value = payload
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    monkeypatch.setattr(plan_mod, "_build_planner", lambda: llm)


# ── decompose_question ───────────────────────────────────────────────────────


def test_simple_passes_through(monkeypatch):
    _patch(monkeypatch, {"is_compound": False, "subquestions": ["Who is the Cheshire Cat?"]})
    out = decompose_question("Who is the Cheshire Cat?")
    assert out == {"is_compound": False, "subquestions": ["Who is the Cheshire Cat?"]}


def test_compound_decomposed(monkeypatch):
    _patch(monkeypatch, {"is_compound": True,
                         "subquestions": ["Who is the Queen?", "Who is the Duchess?"]})
    out = decompose_question("Compare the Queen and the Duchess")
    assert out["is_compound"] is True
    assert out["subquestions"] == ["Who is the Queen?", "Who is the Duchess?"]


def test_compound_with_one_part_coerced_to_simple(monkeypatch):
    # Model flagged compound but produced <2 usable parts -> not actionable.
    _patch(monkeypatch, {"is_compound": True, "subquestions": ["only one part"]})
    out = decompose_question("the original question")
    assert out == {"is_compound": False, "subquestions": ["the original question"]}


def test_subquestions_capped(monkeypatch):
    _patch(monkeypatch, {"is_compound": True, "subquestions": [f"q{i}" for i in range(7)]})
    out = decompose_question("a very compound question")
    assert len(out["subquestions"]) == MAX_SUBQUESTIONS == 4


def test_blank_subquestions_filtered(monkeypatch):
    _patch(monkeypatch, {"is_compound": True, "subquestions": ["q1", "   ", "", "q2"]})
    out = decompose_question("q")
    assert out["subquestions"] == ["q1", "q2"]


def test_llm_failure_degrades_to_simple(monkeypatch):
    _patch(monkeypatch, RuntimeError("boom"))
    out = decompose_question("anything at all")
    assert out == {"is_compound": False, "subquestions": ["anything at all"]}


def test_empty_question_is_simple():
    assert decompose_question("") == {"is_compound": False, "subquestions": [""]}


# ── cost estimate ────────────────────────────────────────────────────────────


def test_estimate_cost():
    assert estimate_cost(1) == {"n_subq": 1, "est_calls": 3, "est_latency_s": 8}
    c3 = estimate_cost(3)
    assert c3["n_subq"] == 3 and c3["est_calls"] == 9 and c3["est_latency_s"] == 14
    assert estimate_cost(0)["n_subq"] == 1  # floored to 1


# ── plan_node (graph wrapper) ────────────────────────────────────────────────


def test_plan_node_passthrough_for_non_agent_intents():
    for kind in ("reading_progress", "export_notes", "smalltalk", "chapter_summary"):
        state = {"intent": IntentDecision(kind=kind), "message": "x"}
        assert G.plan_node(state) == {}


def test_plan_node_decomposes_search(monkeypatch):
    monkeypatch.setattr(
        G, "decompose_question",
        lambda q: {"is_compound": True, "subquestions": ["a", "b"]},
    )
    # compound -> plan_node emits fanout_start via the stream writer; stub it
    # (this unit test calls the node directly, outside a graph stream).
    monkeypatch.setattr(G, "get_stream_writer", lambda: (lambda payload: None))
    state = {"intent": IntentDecision(kind="search", search_query="x"),
             "message": "compare a and b"}
    out = G.plan_node(state)
    assert out["plan"] == {"is_compound": True, "subquestions": ["a", "b"]}
    assert out["cost_estimate"]["n_subq"] == 2


def test_plan_node_uses_resolved_question(monkeypatch):
    seen = {}

    def fake_decompose(q):
        seen["q"] = q
        return {"is_compound": False, "subquestions": [q]}

    monkeypatch.setattr(G, "decompose_question", fake_decompose)
    state = {
        "intent": IntentDecision(kind="search", search_query="x"),
        "message": "他怎么样了?",
        "clarification": {"question": "你指的是谁?", "answer": "疯帽子"},
    }
    G.plan_node(state)
    assert "疯帽子" in seen["q"]  # clarify answer folded into the planned question
