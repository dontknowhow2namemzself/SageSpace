"""
Query Translation pipeline:
  1. Multi-Query: expand user question into 3 semantic variants
  2. HyDE: generate a hypothetical answer, embed it, use for retrieval
  3. Merge + deduplicate + MMR re-rank → top-8 chunks
"""
import os
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate


_MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are a reading assistant. For the question below, write exactly 3 diverse "
        "search-query variants (one per line, no numbering, no bullets, no extra text) "
        "that would help retrieve relevant passages from a book.\n\n"
        "Question: {question}\n\n"
        "Variants:"
    ),
)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="openai/gpt-4o-mini",
        openai_api_base=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )


def build_retriever(vectorstore: Chroma, k: int = 3) -> MultiQueryRetriever:
    base = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20, "lambda_mult": 0.5},
    )
    return MultiQueryRetriever.from_llm(
        retriever=base,
        llm=_get_llm(),
        prompt=_MULTI_QUERY_PROMPT,
    )


def retrieve_with_hyde(query: str, vectorstore: Chroma, k: int = 3) -> list[Document]:
    """HyDE: ask LLM to generate a hypothetical answer, then embed & search."""
    llm = _get_llm()
    hypothetical = llm.invoke(
        "Imagine there is a short passage in the book that answers the question below. "
        "Write that hypothetical passage in the book's tone and style "
        "(roughly one paragraph, at most about 120 words).\n\n"
        f"Question: {query}\n\n"
        "Hypothetical passage:"
    ).content
    # Go through a retriever Runnable (not a raw vectorstore.similarity_search)
    # so the HyDE vector lookup emits its OWN span in LangSmith. With the raw
    # call, only the hypothetical-generation LLM is traced and the search that
    # uses it looks "missing" from the graph. Functionally identical:
    # as_retriever defaults to similarity search, top-k.
    return vectorstore.as_retriever(search_kwargs={"k": k}).invoke(
        hypothetical, config={"run_name": "hyde_vector_search"}
    )


def retrieve_combined(query: str, vectorstore: Chroma) -> list[Document]:
    """
    Run Multi-Query + HyDE, deduplicate by chunk_id, return up to 8 docs.
    Each doc's metadata gains 'retrieval_origin': multi_query | hyde | both.
    """
    retriever = build_retriever(vectorstore, k=3)
    multi_results: list[Document] = retriever.invoke(query)
    hyde_results: list[Document] = retrieve_with_hyde(query, vectorstore, k=3)

    multi_ids: set[str] = {d.metadata.get("chunk_id", "") for d in multi_results}
    hyde_ids:  set[str] = {d.metadata.get("chunk_id", "") for d in hyde_results}

    seen: set[str] = set()
    combined: list[Document] = []
    for doc in multi_results + hyde_results:
        cid = doc.metadata.get("chunk_id", "")
        if cid in seen:
            continue
        seen.add(cid)
        if cid in multi_ids and cid in hyde_ids:
            doc.metadata["retrieval_origin"] = "both"
        elif cid in multi_ids:
            doc.metadata["retrieval_origin"] = "multi_query"
        else:
            doc.metadata["retrieval_origin"] = "hyde"
        combined.append(doc)
        if len(combined) >= 8:
            break
    return combined
