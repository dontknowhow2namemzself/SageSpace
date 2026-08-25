"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

Builds a multi-level summarisation tree over book chunks and stores every
level in ChromaDB so queries can hit both fine-grained and abstract nodes.

Tree shape (post-PR6):

  Level 0: raw chunks (verbatim from the chunker)
  Level 1: one deterministic summary per body-matter section (chapter /
           prologue / epilogue / appendix). Skipped for front-matter
           and back-matter sections that the user does not typically
           ask "Chapter N" questions about.
  Level 2+: KMeans clustering across level-1 summaries (theme grouping
           that crosses chapters -- e.g. the trial scenes of Alice).

The pre-PR6 tree was pure KMeans at every level, including level 1.
That made a level-1 node a noisy cluster of "chunks that happened to
embed close together", with no stable section attribution -- a
summary node spanning chapters 6, 8, and 11 would get tagged with
whichever chapter int had the mode count among its leaves, throwing
away the rest. PR6 anchors level 1 to actual book structure, which:

  * makes get_chapter_summary a SQL-direct lookup (no similarity
    search needed for the common case)
  * gives every level-1 node a single, unambiguous section_id
  * lets level 2 do something cleaner: cluster real chapter
    summaries by theme, not raw paragraphs by embedding proximity
"""
import os
import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


from core.paths import DATA_DIR

RAPTOR_MAX_CLUSTERS = 10
CHROMA_DIR = str(DATA_DIR / "chroma_db")

# Section kinds eligible for a level-1 summary. Front-matter (cover /
# ToC / preface) and back-matter (index / bibliography / colophon) get
# no summary -- a user asking "summarize the bibliography" is a degen
# case, and we would rather not pay an LLM call to summarize a copyright
# page. Appendix is included because users do sometimes ask about it.
_BODY_MATTER_KINDS = ("chapter", "prologue", "epilogue", "appendix")

# Maximum input chars per per-section level-1 summary call. Long
# chapters whose total chunk text exceeds this get evenly truncated
# (each chunk gets its share of the budget). Below the budget, each
# chunk is forwarded with its full chunk text to preserve detail.
_LEVEL_1_INPUT_CHAR_BUDGET = 24000


def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )


def build_raptor_index(
    chunks: list[Document],
    book_id: str,
    sections: list[dict] | None = None,
    persist_dir: str = CHROMA_DIR,
    register_block_links: "callable | None" = None,
) -> Chroma:
    """Build RAPTOR tree and store all levels in ChromaDB. Returns vectorstore.

    Args:
        chunks: level-0 chunks from the canonical chunker. Each must
                carry `chunk_id`, `section_id`, and `block_ids` metadata.
        book_id: scopes the Chroma collection.
        sections: list of section dicts (the shape canonical_db.get_sections
                  returns -- includes `section_id`, `kind`, `label`,
                  `printed_number`). When provided, level 1 is built
                  deterministically per body-matter section. When None
                  (legacy path for books ingested before PR4 or for
                  callers that have not been updated), falls back to
                  the original KMeans-at-every-level behavior.
        register_block_links: callback invoked for every summary node
                              (level >= 1) with (node_id, covers_block_ids).
                              Used by api/ingest.py to populate the
                              raptor_node_blocks reverse index.

    Returns the Chroma vectorstore with all levels persisted.
    """
    embeddings = _get_embeddings()
    llm = _get_llm()

    vectorstore = Chroma(
        collection_name=f"book_{book_id}",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    # Level 0: raw chunks (unchanged across PR6).
    for chunk in chunks:
        chunk.metadata["raptor_level"] = 0
    vectorstore.add_documents(chunks)

    # Per-chunk_id -> covers_block_ids map for upward propagation. Level
    # 1+ inherits the union of its children's coverage so summary hits
    # can resolve back to canonical blocks via raptor_node_blocks.
    from core.canonical.chunker import decode_block_ids

    coverage: dict[str, set[str]] = {}
    for ch in chunks:
        cid = ch.metadata.get("chunk_id", "")
        bids = decode_block_ids(ch.metadata.get("block_ids", ""))
        if cid and bids:
            coverage[cid] = set(bids)

    # ── Level 1 ─────────────────────────────────────────────────────────────
    # Preferred path: structural, one summary per body-matter section.
    # Fallback: legacy KMeans (when no sections were passed in).
    level_1_nodes: list[Document] = []
    if sections:
        level_1_nodes = _summarize_per_section(
            chunks=chunks,
            sections=sections,
            llm=llm,
            coverage=coverage,
            register_block_links=register_block_links,
        )
        if level_1_nodes:
            vectorstore.add_documents(level_1_nodes)

    # Decide the input to the next-level KMeans pass.
    #   * If we produced structural level-1 summaries, they are the
    #     input to level 2 (theme clustering across chapters).
    #   * Otherwise we are on the legacy path: input is raw chunks and
    #     KMeans runs starting at level 1, exactly as before PR6.
    if level_1_nodes:
        current_docs = level_1_nodes
        next_level = 2
    else:
        current_docs = chunks
        next_level = 1

    # ── Level 2+ (or level 1+ in the legacy path) ───────────────────────────
    while len(current_docs) > 4 and next_level <= 3:
        summaries, summary_children = _cluster_and_summarize(
            current_docs, llm, embeddings, next_level
        )
        if not summaries:
            break
        for s, children in zip(summaries, summary_children):
            covered: set[str] = set()
            for c in children:
                child_id = c.metadata.get("chunk_id", "")
                if child_id in coverage:
                    covered |= coverage[child_id]
            sid = s.metadata.get("chunk_id", "")
            if sid:
                coverage[sid] = covered
                if register_block_links and covered:
                    register_block_links(sid, covered)
        vectorstore.add_documents(summaries)
        current_docs = summaries
        next_level += 1

    return vectorstore


def get_vectorstore(book_id: str, persist_dir: str = CHROMA_DIR) -> Chroma:
    """Load an existing book's vectorstore."""
    return Chroma(
        collection_name=f"book_{book_id}",
        embedding_function=_get_embeddings(),
        persist_directory=persist_dir,
    )


# ── Level 1: per-section summarization ─────────────────────────────────────


def _summarize_per_section(
    chunks: list[Document],
    sections: list[dict],
    llm: ChatOpenAI,
    coverage: dict[str, set[str]],
    register_block_links: "callable | None",
) -> list[Document]:
    """Build one level-1 summary node per body-matter section.

    Per-section node id is `raptor_l1_<section_id>` (deterministic). The
    section_id metadata lets get_chapter_summary do a direct .get() in
    Chroma without similarity search. The chapter / chapter_label /
    section_label metadata fields are populated so the existing
    consumers (source_refs, faithfulness fallback) keep working.

    Sections with no chunks (e.g. a section that exists in the canonical
    layer but had no level-0 content land in it) are silently skipped.
    """
    # Group chunks by section_id, preserving order within each section
    # for prompt stability.
    chunks_by_section: dict[str, list[Document]] = {}
    for ch in chunks:
        sid = ch.metadata.get("section_id") or ""
        if not sid:
            continue
        chunks_by_section.setdefault(sid, []).append(ch)

    out: list[Document] = []
    for section in sections:
        kind = section.get("kind") or "other"
        if kind not in _BODY_MATTER_KINDS:
            continue
        section_id = section.get("section_id") or ""
        if not section_id:
            continue
        section_chunks = chunks_by_section.get(section_id, [])
        if not section_chunks:
            continue

        node_id = f"raptor_l1_{section_id}"
        section_label = section.get("label") or ""
        printed_number = section.get("printed_number") or 0

        summary_text = _summarize_section_chunks(
            section_label or section_id, section_chunks, llm
        )

        out.append(
            Document(
                page_content=summary_text,
                metadata={
                    "raptor_level": 1,
                    "section_id": section_id,
                    "section_label": section_label,
                    "chapter_label": section_label,
                    "chapter": printed_number,
                    "chunk_id": node_id,
                    "source": section_chunks[0].metadata.get("source", ""),
                    "cluster_size": len(section_chunks),
                },
            )
        )

        # Block coverage union for the raptor_node_blocks reverse index.
        covered: set[str] = set()
        for c in section_chunks:
            cid = c.metadata.get("chunk_id", "")
            if cid in coverage:
                covered |= coverage[cid]
        if covered:
            coverage[node_id] = covered
            if register_block_links:
                register_block_links(node_id, covered)

    return out


def _summarize_section_chunks(
    section_label: str, section_chunks: list[Document], llm: ChatOpenAI
) -> str:
    """Concatenate section chunks under a fair per-chunk budget, then ask
    the LLM for a 200-word summary. Returns the model's response text.

    Budget math: each chunk gets at least 400 chars; if all chunks would
    fit under _LEVEL_1_INPUT_CHAR_BUDGET at their full length, we send
    full text. Otherwise each chunk gets an equal share of the budget
    so long chapters don't truncate just the tail.
    """
    n = len(section_chunks)
    if n == 0:
        return ""
    total_chars = sum(len(c.page_content) for c in section_chunks)
    if total_chars <= _LEVEL_1_INPUT_CHAR_BUDGET:
        per_chunk_chars = None  # full text
    else:
        per_chunk_chars = max(400, _LEVEL_1_INPUT_CHAR_BUDGET // n)

    pieces = []
    for c in section_chunks:
        text = c.page_content if per_chunk_chars is None else c.page_content[:per_chunk_chars]
        pieces.append(text)
    combined = "\n\n".join(pieces)

    prompt = (
        f"Summarize the following content from \"{section_label}\" in 200 words "
        f"or less. Preserve key arguments, characters, and concepts. Be "
        f"specific; do not generalize away the section's distinctive "
        f"details.\n\n{combined}"
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", "") or ""


# ── Level 2+: KMeans across summary nodes ───────────────────────────────────


def _cluster_and_summarize(
    docs: list[Document],
    llm: ChatOpenAI,
    embeddings: OpenAIEmbeddings,
    level: int,
) -> tuple[list[Document], list[list[Document]]]:
    """K-means cluster docs, then summarise each cluster.

    Used at level 2+ (or at level 1+ when sections were not provided,
    i.e. the legacy path for callers that have not adopted the PR6
    sections-passing contract yet).

    Returns (summaries, children_per_summary). The children list lets
    build_raptor_index propagate block coverage upward.
    """
    if len(docs) <= 2:
        return [], []

    texts = [d.page_content for d in docs]
    vecs = np.array(embeddings.embed_documents(texts))
    vecs = normalize(vecs)

    n_clusters = min(RAPTOR_MAX_CLUSTERS, max(2, len(docs) // 4))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(vecs)

    summaries: list[Document] = []
    children_per_summary: list[list[Document]] = []
    for cluster_id in range(n_clusters):
        cluster_docs = [docs[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
        if not cluster_docs:
            continue

        combined = "\n\n".join(d.page_content[:600] for d in cluster_docs)
        prompt = (
            "Please create a concise structured summary (200 words or less) of the following content, preserving key arguments and concepts:\n\n"
            + combined
        )
        summary_text = llm.invoke(prompt).content

        chapters = [d.metadata.get("chapter", 1) for d in cluster_docs]
        dominant_chapter = max(set(chapters), key=chapters.count)

        summaries.append(
            Document(
                page_content=summary_text,
                metadata={
                    "raptor_level": level,
                    "chapter": dominant_chapter,
                    "chunk_id": f"raptor_l{level}_c{cluster_id:03d}",
                    "source": cluster_docs[0].metadata.get("source", ""),
                    "cluster_size": len(cluster_docs),
                },
            )
        )
        children_per_summary.append(cluster_docs)
    return summaries, children_per_summary
