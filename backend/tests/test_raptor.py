import pytest
import numpy as np
from unittest.mock import MagicMock
from langchain_core.documents import Document
from core.raptor import _cluster_and_summarize


def make_docs(n: int) -> list[Document]:
    return [
        Document(
            page_content=f"Content of document {i}. " * 20,
            metadata={"chapter": (i // 5) + 1, "chunk_id": f"chunk_{i:04d}", "raptor_level": 0, "source": "test.pdf"},
        )
        for i in range(n)
    ]


def test_cluster_produces_fewer_docs():
    docs = make_docs(20)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="摘要内容")
    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = np.random.rand(20, 128).tolist()

    summaries, children = _cluster_and_summarize(docs, mock_llm, mock_emb, level=1)
    assert 0 < len(summaries) < len(docs)
    # Children list aligns 1:1 with summaries and partitions the original docs.
    assert len(children) == len(summaries)
    flat_children = [c for group in children for c in group]
    assert len(flat_children) == len(docs)
    for s in summaries:
        assert s.metadata["raptor_level"] == 1
        assert s.metadata["chunk_id"].startswith("raptor_l1_")


def test_cluster_too_few_docs_returns_empty():
    docs = make_docs(2)
    summaries, children = _cluster_and_summarize(docs, MagicMock(), MagicMock(), level=1)
    assert summaries == []
    assert children == []


def test_cluster_dominant_chapter():
    docs = [
        Document(page_content="text", metadata={"chapter": 2, "chunk_id": f"c{i}", "raptor_level": 0, "source": ""})
        for i in range(10)
    ]
    # Override one to chapter 3
    docs[0].metadata["chapter"] = 3
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="summary")
    mock_emb = MagicMock()
    # All in one cluster
    mock_emb.embed_documents.return_value = np.zeros((10, 128)).tolist()

    summaries, _children = _cluster_and_summarize(docs, mock_llm, mock_emb, level=1)
    assert len(summaries) >= 1
    # Most docs are chapter 2, so dominant chapter should be 2
    assert summaries[0].metadata["chapter"] == 2
