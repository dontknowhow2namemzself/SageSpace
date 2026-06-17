"""LangGraph chat-turn workflow (agentic upgrade, design §3).

Public surface:
  * `build_chat_graph(checkpointer=None)` -- assemble + compile the graph
    (tests inject an isolated checkpointer).
  * `get_chat_graph()` -- the process-wide compiled singleton used by
    api/chat.py.
  * `GraphState` -- the typed state object flowing through the graph.
"""
from core.graph.build import build_chat_graph, checkpoint_db_path, get_chat_graph
from core.graph.state import GraphState

__all__ = [
    "build_chat_graph",
    "get_chat_graph",
    "checkpoint_db_path",
    "GraphState",
]
