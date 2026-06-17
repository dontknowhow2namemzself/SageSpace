import core.database as db
import core.pipeline.finalize as finalize_module
from core.pipeline.finalize import (
    _map_facts_to_chunks,
    _parse_attribution_mapping,
    inject_fact_attribution,
)


def test_inject_fact_attribution_adds_fact_metadata(monkeypatch):
    """Without retrieval docs (smalltalk/progress paths) no mapper call is
    made and every fact stays unattributed: empty data-chunk-ids, no icon.
    The turn-level unions stay at the payload top level for SSE consumers."""
    answer = '<fact>First factual sentence.</fact><commentary>Aside.</commentary><fact>Second factual sentence.</fact>'
    attribution = {
        'retrieval_event_ids': ['evt_1'],
        'chunk_ids': ['chunk_0001', 'chunk_0002'],
        'raptor_ids': ['raptor_l1_0001'],
    }

    def _boom():
        raise AssertionError("mapper LLM must not be built without docs")
    monkeypatch.setattr(finalize_module, "_build_attribution_llm", _boom)

    enriched, payload = inject_fact_attribution(answer, attribution)

    assert 'data-fact-id="f1"' in enriched
    assert 'data-fact-id="f2"' in enriched
    assert 'data-chunk-ids=""' in enriched
    # Per-fact raptor attribution was removed (2026-06-08): the only
    # raptor list is the turn-level union, which misled as per-fact
    # evidence. It must not appear on <fact> tags or in facts[i].
    assert 'data-raptor-ids' not in enriched
    assert payload is not None
    assert len(payload['facts']) == 2
    assert payload['facts'][0]['fact_id'] == 'f1'
    assert payload['facts'][0]['chunk_ids'] == []
    assert payload['facts'][1]['chunk_ids'] == []
    assert 'raptor_ids' not in payload['facts'][0]
    # The turn-level unions stay in the payload (SSE answer_attribution
    # consumers, e.g. sage-eval, read them from there).
    assert payload['chunk_ids'] == ['chunk_0001', 'chunk_0002']
    assert payload['raptor_ids'] == ['raptor_l1_0001']


def test_attach_event_answer_attribution_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', tmp_path / 'test.db')
    db.init_db()

    book_id = db.create_book('Test Book', 'Author', '/tmp/test.pdf')
    session_id = db.create_session(book_id)
    event_id = db.create_retrieval_event(
        session_id=session_id,
        book_id=book_id,
        query_text='query',
        multi_query_variants_json='[]',
        hyde_hypothesis='',
        raw_hits_count=1,
        new_raw_hits_count=1,
        summary_hits_count=0,
    )

    payload = {
        'retrieval_event_ids': [event_id],
        'chunk_ids': ['chunk_0001'],
        'raptor_ids': [],
        'facts': [
            {
                'fact_id': 'f1',
                'text': 'fact text',
                'chunk_ids': ['chunk_0001'],
                'retrieval_event_ids': [event_id],
            }
        ],
    }
    db.attach_event_answer_attribution(event_id, payload)

    event = db.get_retrieval_events(session_id)[0]
    assert 'answer_attribution_json' in event
    assert 'fact text' in (event['answer_attribution_json'] or '')


# ── LLM-mapped per-fact source attribution ────────────────────────────────


_DOCS = [
    {"chunk_id": "chk_intro", "text": "Alice fell down the rabbit hole",
     "raptor_level": 0},
    {"chunk_id": "chk_cat", "text": "the Cheshire Cat appeared and grinned",
     "raptor_level": 0},
    {"chunk_id": "chk_tea", "text": "Alice attended a Mad Tea-Party",
     "raptor_level": 0},
]

_ATTRIB = {"chunk_ids": ["chk_intro", "chk_cat", "chk_tea"], "raptor_ids": [],
           "retrieval_event_ids": ["evt_1"]}


def _mock_mapper(monkeypatch, result):
    calls = []

    def fake(fact_texts, docs):
        calls.append({"facts": list(fact_texts), "docs": list(docs)})
        return result

    monkeypatch.setattr(finalize_module, "_map_facts_to_chunks", fake)
    return calls


def test_per_fact_attribution_uses_mapper_result(monkeypatch):
    """Each fact's data-chunk-ids is exactly what the mapper returned —
    different facts route to different chunks, multi-id facts allowed."""
    calls = _mock_mapper(
        monkeypatch, [["chk_intro"], ["chk_cat", "chk_tea"], []]
    )
    answer = (
        '<fact>Alice fell into Wonderland.</fact>'
        '<fact>She met the Cat at the tea party.</fact>'
        '<fact>Totally ungrounded claim.</fact>'
    )
    enriched, payload = inject_fact_attribution(
        answer, _ATTRIB, retrieval_docs=_DOCS
    )

    per_fact = [f["chunk_ids"] for f in payload["facts"]]
    assert per_fact == [["chk_intro"], ["chk_cat", "chk_tea"], []]
    assert 'data-chunk-ids="chk_intro"' in enriched
    assert 'data-chunk-ids="chk_cat,chk_tea"' in enriched
    # The ungrounded fact gets an EMPTY attribute — no shared-list fallback.
    assert 'data-chunk-ids=""' in enriched
    # Mapper was called once with the fact texts in order.
    assert len(calls) == 1
    assert calls[0]["facts"] == [
        "Alice fell into Wonderland.",
        "She met the Cat at the tea party.",
        "Totally ungrounded claim.",
    ]


def test_per_fact_attribution_passes_summaries_to_the_mapper(monkeypatch):
    """RAPTOR summary nodes ARE mapper candidates (2026-06-10): facts
    grounded only in a chapter summary cite the summary (popup labels
    it; the Reading Map filters raptor ids out of its lit set)."""
    calls = _mock_mapper(monkeypatch, [["raptor_l1_sec_x"]])
    docs = [
        {"chunk_id": "raptor_l1_sec_x", "text": "Chapter summary",
         "raptor_level": 1},
        {"chunk_id": "chk_real", "text": "Alice meets the Cheshire Cat",
         "raptor_level": 0},
    ]
    answer = '<fact>Alice meets the Cheshire Cat.</fact>'
    _, payload = inject_fact_attribution(answer, _ATTRIB, retrieval_docs=docs)

    assert [d["chunk_id"] for d in calls[0]["docs"]] == [
        "raptor_l1_sec_x", "chk_real",
    ]
    assert payload["facts"][0]["chunk_ids"] == ["raptor_l1_sec_x"]


def test_mapper_failure_degrades_to_unattributed(monkeypatch):
    """If the mapper LLM call blows up, every fact gets [] — the answer
    still renders, just without citation icons."""
    class _BoomLLM:
        def invoke(self, messages):
            raise RuntimeError("provider down")

    monkeypatch.setattr(
        finalize_module, "_build_attribution_llm", lambda: _BoomLLM()
    )
    answer = '<fact>Alice fell into Wonderland.</fact>'
    enriched, payload = inject_fact_attribution(
        answer, _ATTRIB, retrieval_docs=_DOCS
    )
    assert payload["facts"][0]["chunk_ids"] == []
    assert 'data-chunk-ids=""' in enriched


def test_map_facts_to_chunks_happy_path(monkeypatch):
    """End-to-end through the real parser with a faked LLM response."""
    class _FakeLLM:
        def invoke(self, messages):
            class R:
                content = (
                    '{"mappings": [{"fact": 1, "passages": [2]},'
                    ' {"fact": 2, "passages": []}]}'
                )
            return R()

    monkeypatch.setattr(
        finalize_module, "_build_attribution_llm", lambda: _FakeLLM()
    )
    out = _map_facts_to_chunks(["fact one", "fact two"], _DOCS)
    assert out == [["chk_cat"], []]


# ── _parse_attribution_mapping unit ───────────────────────────────────────


def test_parse_mapping_drops_invalid_passage_numbers():
    raw = '{"mappings": [{"fact": 1, "passages": [0, 99, 2, "x", 2]}]}'
    out = _parse_attribution_mapping(raw, 1, _DOCS)
    # 0 and 99 out of range, "x" non-int, duplicate 2 deduped.
    assert out == [["chk_cat"]]


def test_parse_mapping_caps_ids_per_fact():
    raw = '{"mappings": [{"fact": 1, "passages": [1, 2, 3]}]}'
    out = _parse_attribution_mapping(raw, 1, _DOCS)
    assert out == [["chk_intro", "chk_cat"]]  # capped at 2


def test_parse_mapping_missing_facts_get_empty():
    raw = '{"mappings": [{"fact": 2, "passages": [1]}]}'
    out = _parse_attribution_mapping(raw, 3, _DOCS)
    assert out == [[], ["chk_intro"], []]


def test_parse_mapping_unparseable_returns_none():
    assert _parse_attribution_mapping("not json", 2, _DOCS) is None
    assert _parse_attribution_mapping('["list"]', 2, _DOCS) is None


def test_parse_mapping_non_list_mappings_returns_none():
    """Valid JSON, wrong shape ('mappings' not a list) must degrade,
    not raise — a TypeError here would abort the turn before the token
    frame ships."""
    assert _parse_attribution_mapping('{"mappings": 7}', 1, _DOCS) is None
    assert _parse_attribution_mapping('{"mappings": true}', 1, _DOCS) is None
    assert _parse_attribution_mapping('{"mappings": "x"}', 1, _DOCS) is None


def test_parse_mapping_rejects_json_booleans():
    """bool is a subclass of int in Python: JSON true must not pass as
    passage 1, and a fact key of true must not collide with fact 1."""
    raw = '{"mappings": [{"fact": 1, "passages": [true, 2]}, {"fact": true, "passages": [1]}]}'
    out = _parse_attribution_mapping(raw, 1, _DOCS)
    assert out == [["chk_cat"]]


def test_quote_grounding_overrides_mapper_pick(monkeypatch):
    """A fact quoting the book verbatim must cite the chunk CONTAINING
    that quote — even when the mapper picked a same-scene neighbour
    (live 2026-06-10: the Hatter-riddle fact got cited to the riddle
    follow-up chunk)."""
    docs = [
        {"chunk_id": "chk_followup",
         "text": '"Have you guessed the riddle yet?" the Hatter said.',
         "raptor_level": 0},
        {"chunk_id": "chk_riddle",
         "text": 'The Hatter opened his eyes: "Why is a raven like a writing-desk?"',
         "raptor_level": 0},
    ]
    _mock_mapper(monkeypatch, [["chk_followup"]])  # mapper picks wrong
    answer = '<fact>The Hatter asks, “Why is a raven like a writing-desk?”</fact>'
    attrib = {"chunk_ids": ["chk_followup", "chk_riddle"], "raptor_ids": [],
              "retrieval_event_ids": ["evt_1"]}
    _, payload = inject_fact_attribution(answer, attrib, retrieval_docs=docs)
    assert payload["facts"][0]["chunk_ids"] == ["chk_riddle"]


def test_quote_grounding_nulls_when_quote_absent_from_pool(monkeypatch):
    """When NO pool chunk contains the quote (it grounded via a RAPTOR
    summary), an adjacent-scene citation misleads — the fact goes
    unattributed instead."""
    docs = [
        {"chunk_id": "chk_followup",
         "text": '"Have you guessed the riddle yet?" the Hatter said.',
         "raptor_level": 0},
    ]
    _mock_mapper(monkeypatch, [["chk_followup"]])
    answer = '<fact>The Hatter asks, “Why is a raven like a writing-desk?”</fact>'
    attrib = {"chunk_ids": ["chk_followup"], "raptor_ids": [],
              "retrieval_event_ids": ["evt_1"]}
    _, payload = inject_fact_attribution(answer, attrib, retrieval_docs=docs)
    assert payload["facts"][0]["chunk_ids"] == []


def test_quote_grounding_leaves_unquoted_facts_to_the_mapper(monkeypatch):
    _mock_mapper(monkeypatch, [["chk_cat"]])
    answer = '<fact>Alice meets a grinning cat in a tree.</fact>'
    _, payload = inject_fact_attribution(answer, _ATTRIB, retrieval_docs=_DOCS)
    assert payload["facts"][0]["chunk_ids"] == ["chk_cat"]


def test_quote_grounding_is_verify_not_override(monkeypatch):
    """When the MAPPER's pick contains the quote, it is kept — even if
    another pool doc also contains the same phrase. A containment-first
    search would let the same words from an unrelated chapter hijack
    the citation (live 2026-06-10: 'pack of cards' from the Chapter XI
    trial chunk hijacked a Chapter XII fact)."""
    docs = [
        {"chunk_id": "chk_ch11_trial",
         "text": "the whole pack of cards: the Knave was standing before them",
         "raptor_level": 0},
        {"chunk_id": "raptor_l1_ch12",
         "text": 'Alice declares the court a "pack of cards" and wakes up.',
         "raptor_level": 1},
    ]
    _mock_mapper(monkeypatch, [["raptor_l1_ch12"]])  # mapper picked ch12 summary
    answer = '<fact>She declares the court a “pack of cards,” and wakes.</fact>'
    attrib = {"chunk_ids": ["chk_ch11_trial"], "raptor_ids": ["raptor_l1_ch12"],
              "retrieval_event_ids": ["evt_1"]}
    _, payload = inject_fact_attribution(answer, attrib, retrieval_docs=docs)
    # The mapper's contextual pick survives; the chapter XI string
    # coincidence does NOT override it.
    assert payload["facts"][0]["chunk_ids"] == ["raptor_l1_ch12"]


def test_quote_grounding_strips_edge_punctuation(monkeypatch):
    """“pack of cards,” (comma inside the quotes, American style) must
    match a doc containing “pack of cards:” — edge punctuation is not
    part of the quote."""
    docs = [
        {"chunk_id": "chk_trial",
         "text": "the whole pack of cards: the Knave was standing",
         "raptor_level": 0},
    ]
    _mock_mapper(monkeypatch, [[]])  # mapper found nothing
    answer = '<fact>She declares them a “pack of cards,” at the end.</fact>'
    attrib = {"chunk_ids": ["chk_trial"], "raptor_ids": [],
              "retrieval_event_ids": ["evt_1"]}
    _, payload = inject_fact_attribution(answer, attrib, retrieval_docs=docs)
    assert payload["facts"][0]["chunk_ids"] == ["chk_trial"]


def test_quote_grounding_falls_back_to_summary_when_no_raw_contains(monkeypatch):
    """Quote lives only in a chapter summary (the Hatter-riddle case):
    cite the summary rather than nothing — the popup shows it labeled
    as AI-generated."""
    docs = [
        {"chunk_id": "chk_followup",
         "text": '"Have you guessed the riddle yet?" the Hatter said.',
         "raptor_level": 0},
        {"chunk_id": "raptor_l1_ch7",
         "text": 'the Hatter asking, "Why is a raven like a writing-desk?"',
         "raptor_level": 1},
    ]
    _mock_mapper(monkeypatch, [["chk_followup"]])  # pick lacks the quote
    answer = '<fact>The Hatter asks, “Why is a raven like a writing-desk?”</fact>'
    attrib = {"chunk_ids": ["chk_followup"], "raptor_ids": ["raptor_l1_ch7"],
              "retrieval_event_ids": ["evt_1"]}
    _, payload = inject_fact_attribution(answer, attrib, retrieval_docs=docs)
    assert payload["facts"][0]["chunk_ids"] == ["raptor_l1_ch7"]


def test_map_facts_to_chunks_wrong_shape_degrades(monkeypatch):
    """End-to-end: a valid-JSON wrong-shape mapper response degrades to
    all-empty instead of raising out of _map_facts_to_chunks."""
    class _FakeLLM:
        def invoke(self, messages):
            class R:
                content = '{"mappings": 7}'
            return R()

    monkeypatch.setattr(
        finalize_module, "_build_attribution_llm", lambda: _FakeLLM()
    )
    assert _map_facts_to_chunks(["fact one"], _DOCS) == [[]]
