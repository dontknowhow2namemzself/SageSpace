"""Tests for the recommendations DB layer (design §B)."""
import pytest

import core.database as db_module
from core.database import (
    init_db,
    insert_recommendation,
    list_recommendations,
    get_recommendation,
    set_recommendation_status,
    recommended_titles,
    recommendation_stats,
)


@pytest.fixture(autouse=True)
def use_test_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    init_db()


def _seed(title, status="suggested", author="A"):
    return insert_recommendation(
        title=title, author=author, blurb="b",
        reason="因为你在读X", which_interest="X", status=status,
    )


def test_insert_and_get():
    rec_id = _seed("Meditations")
    row = get_recommendation(rec_id)
    assert row["title"] == "Meditations"
    assert row["status"] == "suggested"
    assert row["which_interest"] == "X"


def test_get_missing_returns_none():
    assert get_recommendation("nope") is None


def test_list_filters_by_status():
    _seed("A", status="suggested")
    _seed("B", status="added")
    _seed("C", status="suggested")
    assert {r["title"] for r in list_recommendations(status="suggested")} == {"A", "C"}
    assert {r["title"] for r in list_recommendations(status="added")} == {"B"}
    assert len(list_recommendations()) == 3  # no filter -> all


def test_set_status_transitions_and_reports_missing():
    rec_id = _seed("A")
    assert set_recommendation_status(rec_id, "added") is True
    assert get_recommendation(rec_id)["status"] == "added"
    assert set_recommendation_status("ghost", "added") is False  # -> API 404


def test_set_status_rejects_unknown_status():
    rec_id = _seed("A")
    with pytest.raises(ValueError):
        set_recommendation_status(rec_id, "bogus")


def test_recommended_titles_spans_all_statuses():
    _seed("Owned", status="dismissed")
    _seed("Seen", status="seen")
    _seed("Live", status="suggested")
    # Exclude set must include EVERY title ever shown, regardless of status.
    assert recommended_titles() == {"Owned", "Seen", "Live"}


def test_recommendation_stats_groups_by_status():
    _seed("A", status="suggested")
    _seed("B", status="added")
    _seed("C", status="added")
    _seed("D", status="dismissed")
    assert recommendation_stats() == {"suggested": 1, "added": 2, "dismissed": 1}


def test_recommendation_stats_empty():
    assert recommendation_stats() == {}
