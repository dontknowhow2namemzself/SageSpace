import core.database as db_module
from core.pipeline.finalize import _persist_usage_from_callback
from core.pricing import reset_pricing_cache


class DummyUsageCallback:
    def __init__(self, usage_metadata):
        self.usage_metadata = usage_metadata


def test_persist_usage_from_callback_updates_session_totals(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text(
        '{"openai/gpt-5.4-mini-20260317":{"input_per_1m_tokens":0.6,"output_per_1m_tokens":2.4}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("MODEL_PRICING_PATH", str(pricing_file))
    reset_pricing_cache()
    db_module.init_db()

    book_id = db_module.create_book("Test Book", "Author", "/tmp/fake.pdf")
    session_id = db_module.create_session(book_id)

    cb = DummyUsageCallback(
        {
            "openai/gpt-5.4-mini-20260317": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "total_tokens": 1500,
            }
        }
    )

    _persist_usage_from_callback(session_id, cb)
    session = db_module.get_session(session_id)

    assert session is not None
    assert session["total_tokens_in"] == 1000
    assert session["total_tokens_out"] == 500
    assert session["total_cost_usd"] > 0


def test_unknown_model_records_tokens_with_zero_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "test.db")
    pricing_file = tmp_path / "pricing.json"
    pricing_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MODEL_PRICING_PATH", str(pricing_file))
    reset_pricing_cache()
    db_module.init_db()

    book_id = db_module.create_book("Test Book", "Author", "/tmp/fake.pdf")
    session_id = db_module.create_session(book_id)

    cb = DummyUsageCallback(
        {
            "unknown/model": {
                "input_tokens": 200,
                "output_tokens": 50,
                "total_tokens": 250,
            }
        }
    )

    _persist_usage_from_callback(session_id, cb)
    session = db_module.get_session(session_id)

    assert session is not None
    assert session["total_tokens_in"] == 200
    assert session["total_tokens_out"] == 50
    assert session["total_cost_usd"] == 0
