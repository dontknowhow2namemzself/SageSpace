"""Tests for the section label classifier + printed-number parser.

These are the spec for the new chapter_parse module that PR4 leans
on. Every section label produced by the EPUB / PDF normalizers will
flow through these two helpers, and so does every "is this Chapter N"
lookup downstream (get_chapter_summary, future intent node, frontend
chapter picker).

We separate two contracts:

  parse_printed_number(label) -> int | None
    Extract the author-printed number ("CHAPTER V" -> 5). Pure
    function. Must reject malformed roman numerals and random text.

  classify_section_kind(label) -> str
    Map to one of cover / titlepage / toc / preface / foreword /
    introduction / prologue / chapter / epilogue / afterword /
    appendix / glossary / index / bibliography / front_matter /
    back_matter / other. Order rules:
      * prologue / epilogue beats front/back-matter
      * back-matter keywords beat chapter detection (so
        "Appendix B" classifies as back_matter, not chapter)
"""
from __future__ import annotations

import pytest

from core.canonical.chapter_parse import (
    classify_section_kind,
    parse_printed_number,
)


# ── parse_printed_number: positive forms ───────────────────────────────────


@pytest.mark.parametrize("label,expected", [
    # Arabic, with various prefixes
    ("Chapter 5", 5),
    ("CHAPTER 5", 5),
    ("chapter 5", 5),
    ("Chapter 5: The Caterpillar", 5),
    ("5. The Caterpillar", 5),
    ("5: The Caterpillar", 5),
    ("5", 5),
    ("Chapter 12", 12),
    ("Chapter 108", 108),
    # Roman, with prefix
    ("Chapter V", 5),
    ("CHAPTER V", 5),
    ("CHAPTER V. Pig and Pepper", 5),
    ("Chapter VIII", 8),
    ("Chapter IX. The Mock Turtle's Story", 9),
    ("Chapter XII", 12),
    ("Chapter XIV", 14),
    # Roman, standalone (just the numeral, possibly with period)
    ("V.", 5),
    ("IX", 9),
    ("XII. Alice's Evidence", 12),
    # Chinese
    ("第五章", 5),
    ("第五章 牛刀小试", 5),
    ("第十二章", 12),
    ("第一百零八章", 108),
    ("第二十一章", 21),
    # English number words (small + compound)
    ("Chapter One", 1),
    ("Chapter Five", 5),
    ("CHAPTER FIVE", 5),
    ("Chapter Twenty", 20),
    ("Chapter Twenty-One", 21),
    ("Chapter twenty one", 21),  # space separator
])
def test_parse_printed_number_extracts_expected_int(label, expected):
    assert parse_printed_number(label) == expected


# ── parse_printed_number: negative forms ───────────────────────────────────


@pytest.mark.parametrize("label", [
    None,
    "",
    "   ",
    "Cover",
    "Title Page",
    "Contents",
    "Index",
    "Glossary",
    "Bibliography",
    "About the Author",
    "Preface to the Second Edition",
    "Foreword",
    # Body text-like strings -- the parser must not greedily pluck a
    # roman numeral out of unrelated words.
    "In the beginning",
    "It was a dark and stormy night",
    "Down the Rabbit-Hole",
    # Roman lookalikes that should NOT validate (round-trip rejects)
    "Chapter IIII",   # malformed, canonical is IV
    "Chapter VV",     # malformed, canonical is X
    "Chapter LL",     # malformed
    # Standalone roman in body text
    "I am Alice",
    # Bare number followed by plain words (no heading punctuation) --
    # Gutenberg publisher-advert address lines must NOT become chapters
    # (Soap Manufacturer bug: these parsed as chapters 8 and 30).
    "8 BROADWAY, LUDGATE HILL, LONDON, E.C.",
    "30 GEORGE SQUARE, GLASGOW.",
    "8 Broadway",
])
def test_parse_printed_number_returns_none_for_non_chapter_labels(label):
    assert parse_printed_number(label) is None


def test_parse_printed_number_zero_or_negative_rejected():
    """Author-printed numbers are always positive."""
    assert parse_printed_number("Chapter 0") is None


# ── classify_section_kind: front-matter ────────────────────────────────────


@pytest.mark.parametrize("label", [
    "Cover",
    "Title Page",
    "Half Title",
    "Copyright",
    "Copyright Page",
    "Contents",
    "Table of Contents",
    "Dedication",
    "Foreword",
    "Preface",
    "Preface to the Second Edition",
    "Introduction",
    "Acknowledgments",
    "目录",
    "前言",
    "自序",
    "译者序",
    "版权",
    "扉页",
])
def test_classify_section_kind_front_matter(label):
    assert classify_section_kind(label) == "front_matter"


# ── classify_section_kind: prologue / epilogue (specific) ──────────────────


@pytest.mark.parametrize("label", [
    "Prologue",
    "PROLOGUE",
    "Prologue: The Beginning",
    "引子",
    "楔子",
])
def test_classify_section_kind_prologue(label):
    assert classify_section_kind(label) == "prologue"


@pytest.mark.parametrize("label", [
    "Epilogue",
    "EPILOGUE",
    "Epilogue: The End",
    "尾声",
    "终章",
])
def test_classify_section_kind_epilogue(label):
    assert classify_section_kind(label) == "epilogue"


# ── classify_section_kind: chapter ─────────────────────────────────────────


@pytest.mark.parametrize("label", [
    "Chapter 1",
    "Chapter V",
    "CHAPTER VI. Pig and Pepper",
    "第五章",
    "第十二章",
    "Chapter Twenty-One",
    "5. The Caterpillar",
    # Explicit chapter marker outranks keyword matches: a folded
    # Gutenberg label must stay a chapter even though "introduction"
    # is a front-matter keyword and "notes" a back-matter one.
    "CHAPTER I. INTRODUCTION.",
    "Chapter 3: Notes on Method",
])
def test_classify_section_kind_chapter(label):
    assert classify_section_kind(label) == "chapter"


@pytest.mark.parametrize("label", [
    # Publisher-advert address lines from Gutenberg back matter: bare
    # leading numbers must not make these chapters.
    "8 BROADWAY, LUDGATE HILL, LONDON, E.C.",
    "30 GEORGE SQUARE, GLASGOW.",
])
def test_classify_section_kind_address_lines_are_other(label):
    assert classify_section_kind(label) == "other"


# ── classify_section_kind: back-matter (must beat "looks like chapter") ───


@pytest.mark.parametrize("label", [
    "Appendix",
    "Appendix A",       # would otherwise extract roman A->1, but kind wins
    "Appendix B",
    "Appendix B: Notes on the Translation",
    "Afterword",
    "Glossary",
    "Index",
    "Bibliography",
    "Colophon",
    "Notes",
    "References",
    "About the Author",
    "附录",
    "索引",
    "参考文献",
    "后记",
])
def test_classify_section_kind_back_matter(label):
    assert classify_section_kind(label) == "back_matter"


# ── classify_section_kind: fallthroughs ────────────────────────────────────


@pytest.mark.parametrize("label", [
    None,
    "",
    "   ",
    "Some Untitled Block",
    "The First Friday",
    "A Brief Aside",
])
def test_classify_section_kind_other(label):
    assert classify_section_kind(label) == "other"


# ── Combined: Alice in Wonderland ToC layout ──────────────────────────────


def test_alice_in_wonderland_front_matter_offset_resolves_correctly():
    """The user-reported bug: Alice's section list begins with several
    front-matter slots, so sections[chapter_num - 1] picks the wrong
    section when chapter_num is small. With kind + printed_number, the
    lookup picks the section whose author labeled it Chapter N
    regardless of how many front-matter slots came before.

    Note: labels here are what classify_section_kind can distinguish
    from `label` alone. Some real EPUB front-matter sections carry the
    book title as their label (e.g. order_idx 0 = "Alice's Adventures
    in Wonderland") -- those cases need the EPUB epub:type signal,
    which normalize_epub.py threads through separately. The classifier
    here is the fallback when no semantic markup is available.
    """
    sections = [
        ("Cover", "front_matter", None),
        ("Title Page", "front_matter", None),
        ("Copyright", "front_matter", None),
        ("Contents", "front_matter", None),
        ("CHAPTER I. Down the Rabbit-Hole", "chapter", 1),
        ("CHAPTER II. The Pool of Tears", "chapter", 2),
        ("CHAPTER III. A Caucus-Race and a Long Tale", "chapter", 3),
        ("CHAPTER IV. The Rabbit Sends in a Little Bill", "chapter", 4),
        ("CHAPTER V. Advice from a Caterpillar", "chapter", 5),
        ("CHAPTER VI. Pig and Pepper", "chapter", 6),
        ("CHAPTER VII. A Mad Tea-Party", "chapter", 7),
    ]
    for label, expected_kind, expected_num in sections:
        assert classify_section_kind(label) == expected_kind, label
        assert parse_printed_number(label) == expected_num, label
