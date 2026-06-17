"""Chat-turn pipeline modules.

The eventual home for the Intent / Retrieve / Synthesize / Finalize nodes
that will replace the ReAct agent (see docs/ARCHITECTURE.md §6.5 and the
PR-series notes). PR3 lands `finalize` alone so the fast path and the
slow path can share their per-turn closing semantics today, with no
risk to the agent code itself.
"""
