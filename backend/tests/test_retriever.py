from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from core.retriever import retrieve_combined


def make_doc(chunk_id: str, content: str = "test content") -> Document:
    return Document(page_content=content, metadata={"chunk_id": chunk_id, "chapter": 1})


def test_retrieve_combined_deduplicates():
    doc_a = make_doc("chunk_0001", "philosophy text")
    doc_b = make_doc("chunk_0002", "ethics content")
    doc_dup = make_doc("chunk_0001", "philosophy text duplicate")

    with patch("core.retriever.build_retriever") as mock_build, \
         patch("core.retriever.retrieve_with_hyde") as mock_hyde:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [doc_a, doc_b]
        mock_build.return_value = mock_retriever
        mock_hyde.return_value = [doc_dup]

        results = retrieve_combined("test query", MagicMock())

    ids = [d.metadata["chunk_id"] for d in results]
    assert len(ids) == len(set(ids))
    assert "chunk_0001" in ids
    assert "chunk_0002" in ids


def test_retrieve_combined_max_8():
    docs = [make_doc(f"chunk_{i:04d}") for i in range(6)]

    with patch("core.retriever.build_retriever") as mock_build, \
         patch("core.retriever.retrieve_with_hyde") as mock_hyde:
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = docs[:6]
        mock_build.return_value = mock_retriever
        mock_hyde.return_value = [make_doc(f"chunk_{i:04d}") for i in range(6, 12)]

        results = retrieve_combined("test", MagicMock())

    assert len(results) <= 8


def test_retrieve_combined_attaches_origin():
    mock_vs = MagicMock()

    multi_docs = [make_doc("chunk_0001"), make_doc("chunk_0002")]
    hyde_docs  = [make_doc("chunk_0002"), make_doc("chunk_0003")]

    with patch("core.retriever.build_retriever") as mock_build, \
         patch("core.retriever.retrieve_with_hyde", return_value=hyde_docs):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = multi_docs
        mock_build.return_value = mock_retriever

        results = retrieve_combined("test query", mock_vs)

    origins = {d.metadata["chunk_id"]: d.metadata["retrieval_origin"] for d in results}
    assert origins["chunk_0001"] == "multi_query"
    assert origins["chunk_0002"] == "both"
    assert origins["chunk_0003"] == "hyde"
