"""Tests for the bounded ReAct retrieve agent (PR2 step 3).

Drives the compiled agent<->tools subgraph with a scripted FakeLLM and
mocked tools (no network), asserting:
  * the loop calls a requested tool, feeds the observation back, and stops
    when the LLM stops requesting tools,
  * the HARD iteration cap stops a runaway agent (never spins),
  * the safety net runs a semantic pass when the agent gathers nothing,
  * evidence dedup merges semantic∩keyword into origin "both",
  * tool dispatch + observation formatting.
"""
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import core.graph.retrieve_agent as ra


def _doc(cid, origin="keyword", text="passage text"):
    return Document(page_content=text, metadata={"chunk_id": cid, "origin": origin})


class FakeLLM:
    """Returns scripted AIMessages in order; repeats the last one forever.

    Each returned message is a fresh copy with a unique id — real LLMs
    always return new messages, and langgraph's add_messages dedups by id,
    so reusing one object would replace rather than append it.
    """

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.i = 0

    def invoke(self, messages):
        msg = self.scripted[min(self.i, len(self.scripted) - 1)]
        self.i += 1
        return msg.model_copy(update={"id": f"ai-{self.i}"})


def _patch_llm(monkeypatch, scripted):
    """Patch _agent_llm to return ONE shared FakeLLM so its script counter
    advances across the agent's successive calls (the real factory makes a
    new client each call, but the mock must persist the scripted position)."""
    fake = FakeLLM(scripted)
    monkeypatch.setattr(ra, "_agent_llm", lambda: fake)
    return fake


def _tc(name, args, id="c1"):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


# ── loop behavior ────────────────────────────────────────────────────────────


def test_agent_calls_tool_then_stops(monkeypatch):
    scripted = [
        AIMessage(content="", tool_calls=[_tc("keyword_search", {"terms": "Cheshire Cat"})]),
        AIMessage(content="I have enough.", tool_calls=[]),
    ]
    _patch_llm(monkeypatch, scripted)
    monkeypatch.setattr(ra, "tool_keyword_search", lambda terms, book_id, vs: [_doc("chk_A")])
    monkeypatch.setattr(ra, "tool_semantic_search", lambda q, vs: [])

    ev = ra.gather_evidence("who is the cheshire cat", "bk", "Alice", vectorstore=None)
    assert [d.metadata["chunk_id"] for d in ev] == ["chk_A"]


def test_agent_can_chain_two_tools(monkeypatch):
    scripted = [
        AIMessage(content="", tool_calls=[_tc("semantic_search", {"query": "cat"}, "a")]),
        AIMessage(content="", tool_calls=[_tc("keyword_search", {"terms": "Cheshire"}, "b")]),
        AIMessage(content="done", tool_calls=[]),
    ]
    _patch_llm(monkeypatch, scripted)
    monkeypatch.setattr(ra, "tool_semantic_search", lambda q, vs: [_doc("c_sem", "semantic")])
    monkeypatch.setattr(ra, "tool_keyword_search", lambda t, b, vs: [_doc("c_kw", "keyword")])

    ev = ra.gather_evidence("q", "bk", "Alice", None)
    assert {d.metadata["chunk_id"] for d in ev} == {"c_sem", "c_kw"}


def test_hard_iteration_cap_stops_runaway(monkeypatch):
    # LLM ALWAYS asks for another tool -> must stop at MAX_ITERATIONS rounds.
    always = AIMessage(content="", tool_calls=[_tc("semantic_search", {"query": "x"})])
    _patch_llm(monkeypatch, [always])
    calls = {"n": 0}

    def fake_sem(q, vs):
        calls["n"] += 1
        return [_doc(f"chk_{calls['n']}", "semantic")]

    monkeypatch.setattr(ra, "tool_semantic_search", fake_sem)
    ev = ra.gather_evidence("q", "bk", "Alice", None)
    assert calls["n"] == ra.MAX_ITERATIONS       # exactly the cap, no spin
    assert len(ev) == ra.MAX_ITERATIONS


def test_safety_net_when_agent_gathers_nothing(monkeypatch):
    # Agent stops immediately with no tool call -> fallback semantic pass.
    _patch_llm(monkeypatch, [AIMessage(content="done", tool_calls=[])])
    monkeypatch.setattr(ra, "tool_semantic_search", lambda q, vs: [_doc("fallback", "semantic")])
    ev = ra.gather_evidence("q", "bk", "Alice", None)
    assert [d.metadata["chunk_id"] for d in ev] == ["fallback"]


def test_tool_error_does_not_crash_loop(monkeypatch):
    scripted = [
        AIMessage(content="", tool_calls=[_tc("keyword_search", {"terms": "x"})]),
        AIMessage(content="done", tool_calls=[]),
    ]
    _patch_llm(monkeypatch, scripted)

    def boom(*a, **k):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(ra, "tool_keyword_search", boom)
    monkeypatch.setattr(ra, "tool_semantic_search", lambda q, vs: [_doc("fb", "semantic")])
    ev = ra.gather_evidence("q", "bk", "Alice", None)  # must not raise
    assert [d.metadata["chunk_id"] for d in ev] == ["fb"]  # empty -> safety net


# ── helpers ──────────────────────────────────────────────────────────────────


def test_merge_evidence_dedups_and_marks_both():
    merged = ra._merge_evidence([_doc("c1", "semantic")],
                                [_doc("c1", "keyword"), _doc("c2", "keyword")])
    assert [d.metadata["chunk_id"] for d in merged] == ["c1", "c2"]
    assert merged[0].metadata["origin"] == "both"        # semantic ∩ keyword
    assert merged[1].metadata["origin"] == "keyword"


def test_merge_origin_rules():
    assert ra._merge_origin("semantic", "keyword") == "both"
    assert ra._merge_origin("keyword", "chapter") == "both"
    assert ra._merge_origin("semantic", "semantic") == "semantic"
    assert ra._merge_origin(None, "keyword") == "keyword"
    assert ra._merge_origin("chapter", "neighbor") == "chapter"  # neither is keyword


def test_dispatch_tool_routes(monkeypatch):
    monkeypatch.setattr(ra, "tool_get_chapter", lambda n, book_id, vs, query="": [_doc(f"ch{n}", "chapter")])
    out = ra._dispatch_tool("get_chapter", {"printed_number": "6"}, "bk", None)
    assert out[0].metadata["chunk_id"] == "ch6"      # str coerced to int
    assert ra._dispatch_tool("nonexistent", {}, "bk", None) == []


def test_format_observation():
    obs = ra._format_observation("keyword_search", [_doc("chk_X", "keyword", "Some passage")])
    assert "keyword_search: 1 passage" in obs
    assert "chunk_id=chk_X" in obs
    assert ra._format_observation("semantic_search", []) == "semantic_search: no passages found."


def test_agent_emits_tool_frames(monkeypatch):
    """The tools node writes tool_start/tool_end to the stream (the UI +
    LangSmith trajectory). Stream the agent subgraph and assert the frames."""
    from langchain_core.messages import HumanMessage

    scripted = [
        AIMessage(content="", tool_calls=[_tc("keyword_search", {"terms": "x"})]),
        AIMessage(content="done", tool_calls=[]),
    ]
    _patch_llm(monkeypatch, scripted)
    monkeypatch.setattr(ra, "tool_keyword_search", lambda t, b, vs: [_doc("c1")])

    agent = ra.get_retrieve_agent()
    init = {"messages": [HumanMessage("q")], "evidence": [], "book_id": "bk",
            "vectorstore": None, "iterations": 0}
    frames = list(agent.stream(init, {"recursion_limit": 12}, stream_mode="custom"))
    assert {"type": "tool_start", "tool": "keyword_search"} in frames
    assert {"type": "tool_end", "tool": "keyword_search"} in frames
