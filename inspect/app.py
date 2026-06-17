"""SageSpace Inspector (v0.1).

Streamlit dev tool for browsing the canonical text layer: Section ->
Block -> Chunk -> RAPTOR tree. Reads SQLite + ChromaDB directly. Never
writes. Run with:

    cd sagespace/inspect
    venv/bin/streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

import lib


st.set_page_config(page_title="SageSpace Inspector", layout="wide")

st.warning(
    "Read-only inspector — never writes data. SQLite opens with `mode=ro`; "
    "ChromaDB is read via get-only queries.",
    icon="🔒",
)


# ── Sidebar: book + section picker ─────────────────────────────────────────

# Section kinds that count as body matter. The rest (cover, toc, license,
# bibliography, etc.) get visually demoted in the picker so the eye lands
# on real chapters first.
_BODY_KINDS = {
    "chapter", "prologue", "epilogue", "appendix",
    "introduction", "preface", "foreword",
}


def _book_label(b: dict) -> str:
    title = b.get("title") or "(untitled)"
    author = b.get("author")
    suffix = f" — {author}" if author else ""
    return f"{title}{suffix}  [{b['id'][:8]}]"


def _section_label(s: dict) -> str:
    kind = (s.get("kind") or "other")
    printed = s.get("printed_number")
    label = (s.get("label") or "").strip() or "(untitled)"
    # Body matter: CAPS prefix + printed number for easy scanning.
    # Non-body:   lowercase prefix, no number, "·" prefix to mute.
    if kind in _BODY_KINDS:
        prefix = kind.upper()
        if printed is not None:
            prefix = f"{prefix} · {printed}"
        return f"{prefix} — {label}"
    return f"· {kind} — {label}"


with st.sidebar:
    st.header("Book")
    try:
        books = lib.list_books()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    if not books:
        st.info("No books in the database.")
        st.stop()

    book_idx = st.selectbox(
        "Select a book",
        range(len(books)),
        format_func=lambda i: _book_label(books[i]),
        label_visibility="collapsed",
    )
    book = books[book_idx]
    st.caption(
        f"chunks: **{book.get('total_chunks')}** · "
        f"chapters: **{book.get('total_chapters')}** · "
        f"status: **{book.get('raptor_status')}**"
    )

    st.divider()
    st.header("Find by id")
    with st.form("lookup_form", clear_on_submit=False, border=False):
        lookup_input = st.text_input(
            "chunk_id or raptor node id",
            value=st.session_state.get("lookup_query", ""),
            placeholder="chk_… / raptor_l1_sec_… / raptor_l2_c…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Find", width="stretch")
    if submitted:
        st.session_state["lookup_query"] = lookup_input
        if lookup_input.strip():
            st.session_state["lookup_result"] = lib.lookup_by_id(
                book["id"], lookup_input
            )
        else:
            st.session_state["lookup_result"] = None

    st.divider()
    st.header("Section")
    sections = lib.load_sections(book["id"])
    if not sections:
        st.info("This book has no canonical sections (v1 ingest?).")
        st.stop()

    # Per-book key so switching books doesn't carry an out-of-range index
    # into a different section list.
    section_key = f"section_radio_{book['id']}"
    section_idx = st.radio(
        "Select a section",
        range(len(sections)),
        format_func=lambda i: _section_label(sections[i]),
        label_visibility="collapsed",
        key=section_key,
    )
    section = sections[section_idx]


# ── Match card (active when a lookup result is in session state) ───────────


def _clear_lookup() -> None:
    st.session_state["lookup_result"] = None
    st.session_state["lookup_query"] = ""


def _jump_to_section(book_id: str, section_idx: int) -> None:
    st.session_state[f"section_radio_{book_id}"] = section_idx
    _clear_lookup()


def _type_label(level: int) -> str:
    return {
        0: "level-0 chunk",
        1: "level-1 section summary",
    }.get(level, f"level-{level} cluster summary")


def _section_index(sections: list[dict], section_id: str) -> int | None:
    for i, s in enumerate(sections):
        if s["section_id"] == section_id:
            return i
    return None


def _render_match_card(result: dict, book: dict, sections: list[dict]) -> None:
    with st.container(border=True):
        st.markdown(f"### `{result['id']}`")
        bits = [_type_label(result["level"])]
        if result.get("cluster_size") is not None:
            bits.append(f"cluster_size **{result['cluster_size']}**")
        if result["block_ids"]:
            bits.append(f"blocks **{len(result['block_ids'])}**")
        if result.get("page") is not None:
            bits.append(f"page **{result['page']}**")
        st.caption(" · ".join(bits))

        sec = result.get("section")
        if sec:
            printed = sec.get("printed_number")
            printed_str = f"#{printed} · " if printed is not None else ""
            st.markdown(
                f"**Section** · `{sec['kind']}` · {printed_str}"
                f"{sec.get('label') or '(unlabeled)'}  "
                f"`{sec['section_id']}`"
            )
        elif result.get("sections"):
            spans = result["sections"]
            st.markdown(f"**Spans {len(spans)} sections**")
            for s in spans[:10]:
                printed = s.get("printed_number")
                printed_str = f"#{printed} · " if printed is not None else ""
                st.markdown(
                    f"- `{s['kind']}` · {printed_str}"
                    f"{s.get('label') or '(unlabeled)'}"
                )
            if len(spans) > 10:
                st.caption(f"… and {len(spans) - 10} more")

        if result.get("primary_block_id"):
            st.markdown(f"**Primary block** · `{result['primary_block_id']}`")

        st.markdown("**Text**")
        st.code(result["text"] or "(empty)", language=None, wrap_lines=True)

        col_jump, col_clear = st.columns([1, 1])
        with col_jump:
            target_section_id = None
            if sec:
                target_section_id = sec["section_id"]
            elif result.get("sections"):
                target_section_id = result["sections"][0]["section_id"]
            idx = (
                _section_index(sections, target_section_id)
                if target_section_id
                else None
            )
            st.button(
                "Show in tree",
                width="stretch",
                disabled=idx is None,
                help=(
                    "Switch the sidebar to this node's section."
                    if idx is not None
                    else "No single section to jump to."
                ),
                on_click=_jump_to_section,
                args=(book["id"], idx if idx is not None else 0),
                key="match_show_in_tree",
            )
        with col_clear:
            st.button(
                "Clear search",
                width="stretch",
                on_click=_clear_lookup,
                key="match_clear",
            )


_lookup_query_active = bool(
    (st.session_state.get("lookup_query") or "").strip()
)
_lookup_result = st.session_state.get("lookup_result")
if _lookup_query_active and _lookup_result is None:
    st.error(
        f"No id matches `{st.session_state['lookup_query'].strip()}` "
        f"in **{book.get('title') or book['id']}**.",
        icon="🔎",
    )
elif _lookup_result is not None:
    _render_match_card(_lookup_result, book, sections)


# ── Main: section header + 3 tabs ──────────────────────────────────────────

st.title(section.get("label") or "(unlabeled section)")
st.caption(
    f"kind: **{section.get('kind')}** · "
    f"printed_number: **{section.get('printed_number')}** · "
    f"order_idx: **{section.get('order_idx')}** · "
    f"level: **{section.get('level')}** · "
    f"section_id: `{section['section_id']}`"
)

tab_blocks, tab_chunks, tab_raptor = st.tabs(["Blocks", "Chunks", "RAPTOR tree"])


# ── Tab 1: Blocks ──────────────────────────────────────────────────────────

def _locator_page(loc: dict) -> object:
    if not loc:
        return ""
    for key in ("page", "print_page", "spine_idx"):
        if key in loc and loc[key] is not None:
            return loc[key]
    return ""


with tab_blocks:
    blocks = lib.load_blocks(book["id"], section["section_id"])
    if not blocks:
        st.info("No blocks in this section.")
    else:
        st.caption(f"{len(blocks)} blocks · ordered by `order_idx`")
        rows = []
        for b in blocks:
            text = b.get("text") or ""
            preview = text if len(text) <= 200 else text[:200] + "…"
            rows.append({
                "block_id": b["block_id"][:12],
                "order_idx": b["order_idx"],
                "kind": b["kind"],
                "page": _locator_page(b.get("locator") or {}),
                "chars": len(text),
                "preview": preview,
            })
        st.dataframe(rows, width="stretch", hide_index=True)


# ── Tab 2: Chunks ──────────────────────────────────────────────────────────

with tab_chunks:
    chunks = lib.load_chunks(book["id"], section["section_id"])
    if not chunks:
        st.info("No chunks indexed for this section.")
    else:
        # Streamlit re-applies `expanded` on every script run, so the
        # toggle effectively becomes a "default" — flipping it expands
        # or collapses every chunk at once; individual clicks afterwards
        # still work until the next rerun.
        expand_all = st.toggle(
            "Expand all chunks",
            value=False,
            key="chunks_expand_all",
            help="Show full chunk text without clicking each one.",
        )
        st.caption(
            f"{len(chunks)} chunks · level-0 retrieval units · "
            "ordered by reading position (first block's `order_idx`)"
        )
        for idx, c in enumerate(chunks):
            header = (
                f"#{idx} `{c['chunk_id']}` · primary "
                f"`{(c['primary_block_id'] or '')[:12]}` · "
                f"{len(c['block_ids'])} blocks · "
                f"{c['char_length']} chars · page {c['page']}"
            )
            with st.expander(header, expanded=expand_all):
                if c["block_ids"]:
                    st.markdown(
                        f"**Block coverage** ({len(c['block_ids'])}): "
                        + ", ".join(f"`{b[:12]}`" for b in c["block_ids"])
                    )
                else:
                    st.markdown("**Block coverage**: _(none recorded)_")
                # st.code gives a built-in copy button and avoids the
                # Streamlit 'C' hotkey conflict that st.text_area suffers
                # from. wrap_lines keeps long prose readable.
                st.code(c["text"], language=None, wrap_lines=True)


# ── Tab 3: RAPTOR tree ─────────────────────────────────────────────────────

def _render_l1(child: dict) -> None:
    # Always surface node_id alongside the section label so the L1 row
    # matches the L2 row's format (which already shows the node_id).
    label = child.get("section_label") or "(no section label)"
    leaves = child.get("leaf_chunks", [])
    with st.expander(
        f"**L1** `{child['node_id']}` · {label} · "
        f"{child['block_count']} blocks · "
        f"{len(leaves)} L0 chunks"
    ):
        st.markdown("**Summary**")
        st.write(child.get("text") or "_(empty)_")
        if leaves:
            st.markdown(f"**Level-0 chunks** ({len(leaves)})")
            for leaf in sorted(
                leaves, key=lambda x: (x.get("primary_block_id") or "")
            ):
                st.markdown(
                    f"- `{leaf['chunk_id']}` · "
                    f"{leaf['char_length']} chars · "
                    f"primary `{(leaf['primary_block_id'] or '')[:12]}` · "
                    f"{len(leaf['block_ids'])} blocks"
                )


with tab_raptor:
    if not (book.get("raptor_status") or "").startswith("ready"):
        st.info(
            f"RAPTOR tree not ready (status={book.get('raptor_status')!r})."
        )
    else:
        tree = lib.load_raptor_tree(book["id"])
        roots = tree.get("roots", [])
        if not roots:
            st.info("No RAPTOR nodes in Chroma for this book.")
        elif tree["has_top_level"]:
            st.caption(
                f"{len(roots)} top-level KMeans clusters · L_top → L1 → L0"
            )
            for root in roots:
                children = root.get("children", [])
                with st.expander(
                    f"**L{root['level']}** `{root['node_id']}` · "
                    f"covers {root['block_count']} blocks · "
                    f"{len(children)} L1 children"
                ):
                    st.markdown("**Summary**")
                    st.write(root.get("text") or "_(empty)_")
                    if children:
                        st.markdown("---")
                        st.markdown("**Children**")
                        for child in children:
                            _render_l1(child)

            if tree.get("orphans"):
                st.divider()
                st.caption(
                    f"{len(tree['orphans'])} L1 nodes with no L2 parent"
                )
                for child in tree["orphans"]:
                    _render_l1(child)
        else:
            st.caption(
                f"No L2 clusters in this book — showing "
                f"{len(roots)} L1 section summaries"
            )
            for child in roots:
                _render_l1(child)
