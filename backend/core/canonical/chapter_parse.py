"""Section label classification + printed-chapter-number extraction.

This module turns a free-form `Section.label` string (e.g.
"CHAPTER VI. Pig and Pepper", "第五章 牛刀小试", "Appendix B", "Cover")
into two structured values:

  * `kind` (str enum):
      cover / titlepage / toc / preface / foreword / introduction /
      prologue / chapter / epilogue / afterword / appendix /
      glossary / index / bibliography / other

  * `printed_number` (int | None):
      The number the author printed on the chapter heading -- 5 for
      "CHAPTER V" / "Chapter 5" / "第五章" / "5. Title". `None` for any
      label that does not look like a chapter at all.

Why both: front-matter sections (cover, ToC, preface) consume
order_idx slots in the canonical sections list, but they are NOT what
a user means when they say "chapter N". The (kind, printed_number)
pair lets us route "chapter 5" to the section whose author labeled it
chapter 5, regardless of how many front-matter slots came before it.

Supported number forms (most specific first; parse_printed_number
tries them in that order):

  Chinese:   第五章, 第十二章, 第一百零八章
  Arabic:    Chapter 5, 5., 5.0 (decimal stripped)
  Roman:     CHAPTER V, Chapter VIII, IX.  (round-trip validated to
             reject malformed forms like "IIII")
  English:   Chapter Five, Chapter Twenty-One   (small + compound words)

Negative cases (parse returns None / classify returns 'other'):
  "Cover", "Contents", "Index", "Glossary", random body text
"""
from __future__ import annotations

import re


# ── kind classification keywords ────────────────────────────────────────────


# (substring match, lowercase, on the label). Order matters: more
# specific labels (prologue) before more general ones (front_matter).
_PROLOGUE_KEYWORDS = ("prologue", "引子", "楔子")
_EPILOGUE_KEYWORDS = ("epilogue", "尾声", "终章")
_FRONT_MATTER_KEYWORDS = (
    # English
    "cover", "title page", "titlepage", "half title", "copyright",
    "contents", "table of contents", "toc", "dedication",
    "foreword", "preface", "introduction", "acknowledgments",
    "frontispiece", "list of figures", "list of tables",
    # Chinese
    "前言", "序言", "自序", "译者", "目录", "版权", "献辞", "导言",
    "扉页", "插图目录",
)
_BACK_MATTER_KEYWORDS = (
    # English
    "afterword", "appendix", "glossary", "index", "bibliography",
    "colophon", "notes", "references", "about the author",
    "further reading",
    # Chinese
    "附录", "索引", "参考文献", "后记", "作者简介", "延伸阅读",
)


def classify_section_kind(label: str | None) -> str:
    """Classify a section label into one of the kind enum values.

    The classifier is intentionally simple: substring match on a
    lowercased copy of the label, in order from most specific
    (prologue / epilogue) to most general (chapter via printed-number
    presence). A label that matches nothing is "other".

    Robustness notes:
      * EPUB-3 ingests should prefer `epub:type` (set externally by
        the normalizer) over this heuristic when both are available.
      * "Appendix A" / "Appendix B" will classify as `appendix` first
        (back_matter keyword wins) even though parse_printed_number
        would extract a roman ordinal -- this is correct.
    """
    if not label:
        return "other"
    label_lower = label.lower()

    # An explicit chapter marker at the START outranks keyword matches:
    # "CHAPTER I. INTRODUCTION." (Gutenberg marker + folded title) is a
    # chapter even though "introduction" is a front-matter keyword.
    if _EXPLICIT_CHAPTER_PREFIX_RE.match(label):
        return "chapter"

    if any(kw in label_lower for kw in _PROLOGUE_KEYWORDS):
        return "prologue"
    if any(kw in label_lower for kw in _EPILOGUE_KEYWORDS):
        return "epilogue"
    if any(kw in label_lower for kw in _BACK_MATTER_KEYWORDS):
        return "back_matter"
    if any(kw in label_lower for kw in _FRONT_MATTER_KEYWORDS):
        return "front_matter"

    # Last-resort heuristic: if we can pull a printed chapter number
    # out of the label, it is almost certainly a body-matter chapter.
    if parse_printed_number(label) is not None:
        return "chapter"

    return "other"


# ── printed_number extraction ──────────────────────────────────────────────


# Chinese: 第N章, where N can be Arabic digits or Chinese numerals.
_CHINESE_CHAPTER_RE = re.compile(
    r"第\s*([0-9零一二三四五六七八九十百千两]+)\s*章"
)

# Roman numerals anchored to the start of the label (optionally
# after "CHAPTER" / "Part" / "Book" prefix). Trailing word boundary
# protects against catching "I" out of "In the beginning..." etc.
_ROMAN_RE = re.compile(
    r"^\s*(?:CHAPTER|Chapter|chapter|PART|Part|part|BOOK|Book|book)?"
    r"\s*([IVXLCDM]+)\b\.?",
    re.IGNORECASE,
)

# Arabic digits. Two forms, mirroring the roman-numeral rules below:
#   * prefixed ("Chapter 5", "Part 2: Title") -- anything may follow;
#   * standalone ("5", "5.", "5: Title") -- the number must be the whole
#     label or be followed by heading punctuation. A bare number followed
#     by plain words ("8 BROADWAY, LUDGATE HILL") is NOT a chapter: TOCs
#     of scanned books carry address/advert lines that start with digits.
_ARABIC_PREFIXED_RE = re.compile(
    r"^\s*(?:CHAPTER|PART|BOOK)\s*(\d+)\b",
    re.IGNORECASE,
)
_ARABIC_BARE_RE = re.compile(
    r"^\s*(\d+)(?:\s*$|\s*[.:：、—–-].*$)"
)

# Explicit chapter marker at the start of a label: "CHAPTER I.",
# "Chapter 5: ...", "PART II". Used by classify_section_kind to outrank
# the keyword heuristics, and by the EPUB normalizer to detect bare
# markers whose title lives in a nested TOC child.
_EXPLICIT_CHAPTER_PREFIX_RE = re.compile(
    r"^\s*(?:CHAPTER|PART|BOOK)\s+(?:\d+|[IVXLCDM]+)\b",
    re.IGNORECASE,
)

# English number words, lowercased. Covers 1..20 + the tens. For
# bigger numbers we fall back to None rather than build a full
# English-number parser -- corpus says this rarely matters past 20.
_ENGLISH_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_ENGLISH_RE = re.compile(
    r"^\s*(?:CHAPTER|Chapter|chapter|PART|Part|part|BOOK|Book|book)\s+"
    r"([a-zA-Z]+(?:[-\s][a-zA-Z]+)?)\b",
    re.IGNORECASE,
)


_CN_NUM_MAP = {
    "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def _parse_chinese_number(text: str) -> int | None:
    """Parse a Chinese-numeral string (e.g. "十", "二十一", "一百零八")
    into an int. Accepts pure-Arabic input too."""
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        value = int(text)
        return value if value > 0 else None
    if text == "十":
        return 10
    if "百" in text:
        parts = text.split("百", 1)
        left = _CN_NUM_MAP.get(parts[0], 1 if parts[0] == "" else None)
        right = _parse_chinese_number(parts[1]) if parts[1] else 0
        if left is None or right is None:
            return None
        return left * 100 + right
    if "十" in text:
        parts = text.split("十", 1)
        left = _CN_NUM_MAP.get(parts[0], 1 if parts[0] == "" else None)
        right = _CN_NUM_MAP.get(parts[1], 0 if parts[1] == "" else None)
        if left is None or right is None:
            return None
        return left * 10 + right
    if len(text) == 1:
        value = _CN_NUM_MAP.get(text)
        return value if value and value > 0 else None
    total = 0
    for char in text:
        value = _CN_NUM_MAP.get(char)
        if value is None:
            return None
        total = total * 10 + value
    return total if total > 0 else None


_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


def _to_roman(n: int) -> str:
    """Canonical roman-numeral form of `n`. Used to round-trip
    validate input so malformed forms like "IIII" or "VV" are rejected."""
    if not 0 < n < 4000:
        return ""
    table = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
    ]
    out = []
    for val, sym in table:
        while n >= val:
            out.append(sym)
            n -= val
    return "".join(out)


def _parse_roman(s: str) -> int | None:
    """Parse a roman numeral string. Returns None for malformed input
    (the round-trip validator catches "IIII", "VV", "LL", etc.)."""
    if not s:
        return None
    s = s.upper()
    if not all(c in _ROMAN_VALUES for c in s):
        return None
    total = 0
    prev = 0
    for c in reversed(s):
        val = _ROMAN_VALUES[c]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    if total <= 0 or _to_roman(total) != s:
        return None
    return total


def _parse_english_words(s: str) -> int | None:
    """Parse "Five", "Twenty-One" into ints. Returns None for unknown
    words. Covers small + compound (tens-units) numbers up to 99."""
    s = (s or "").strip().lower().replace("-", " ")
    if not s:
        return None
    parts = s.split()
    if len(parts) == 1:
        return _ENGLISH_NUM_WORDS.get(parts[0])
    if len(parts) == 2:
        tens = _ENGLISH_NUM_WORDS.get(parts[0])
        units = _ENGLISH_NUM_WORDS.get(parts[1])
        if tens is None or units is None:
            return None
        # Only valid combos are tens (20,30,...,90) + units (1..9)
        if tens >= 20 and tens % 10 == 0 and 1 <= units <= 9:
            return tens + units
    return None


def resolve_kind_and_number(
    label: str | None,
    kind_override: str | None = None,
) -> tuple[str, int | None]:
    """Decide (kind, printed_number) for a section given its label and an
    optional higher-signal kind override from format-specific metadata
    (e.g. EPUB epub:type).

    Priority:
      1. `kind_override` if provided AND non-"other" -- use as-is. This is
         the path EPUB normalizers take when epub:type maps cleanly to
         our kind enum.
      2. Otherwise: classify_section_kind(label) on the label text.

    `printed_number` is always sourced from the label (epub:type does
    not carry chapter numbers). We only populate it for sections whose
    final kind is "chapter" -- a prologue or appendix may have its own
    "number" in the label but those aren't body-matter chapters and
    shouldn't be reachable by a "Chapter N" lookup.
    """
    if kind_override and kind_override != "other":
        kind = kind_override
    else:
        kind = classify_section_kind(label)
    printed_number = parse_printed_number(label) if kind == "chapter" else None
    return kind, printed_number


def parse_printed_number(label: str | None) -> int | None:
    """Extract the author-printed chapter number from a section label.

    Returns the integer when the label clearly carries one (5 for
    "Chapter 5" / "CHAPTER V" / "第五章" / "Chapter Five"), or None.

    Tries in order:
      1. Chinese 第N章 (any position in the label)
      2. Arabic anchored at start
      3. Roman anchored at start (round-trip validated)
      4. English number word (Chapter Five, Chapter Twenty-One)
    """
    if not label:
        return None

    # 1. Chinese
    m = _CHINESE_CHAPTER_RE.search(label)
    if m:
        return _parse_chinese_number(m.group(1))

    # 2. Arabic -- prefixed form first, then the strict standalone form.
    m = _ARABIC_PREFIXED_RE.match(label) or _ARABIC_BARE_RE.match(label)
    if m:
        try:
            value = int(m.group(1))
            return value if value > 0 else None
        except ValueError:
            pass

    # 3. Roman -- only when prefixed by Chapter/Part/Book OR the label
    # is *just* the roman numeral. Otherwise we would catch words like
    # "I" or "MD" in random titles.
    if re.match(r"^\s*(?:CHAPTER|Chapter|chapter|PART|Part|part|BOOK|Book|book)\b",
                label):
        m = _ROMAN_RE.match(label)
        if m:
            value = _parse_roman(m.group(1))
            if value is not None:
                return value
    else:
        # Standalone roman: the numeral must be either the entire label
        # or be followed by a period (signaling "chapter heading"
        # punctuation). Without that rule, body-text labels like
        # "I am Alice" would falsely parse as chapter 1 because of
        # the leading "I".
        m = re.match(r"^\s*([IVXLCDM]+)(?:\s*$|\s*\..*$)", label)
        if m:
            value = _parse_roman(m.group(1))
            if value is not None:
                return value

    # 4. English number words: "Chapter Five", "Chapter Twenty-One"
    m = _ENGLISH_RE.match(label)
    if m:
        value = _parse_english_words(m.group(1))
        if value is not None:
            return value

    return None
