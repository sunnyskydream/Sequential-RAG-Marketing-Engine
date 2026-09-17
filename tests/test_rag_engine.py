"""Unit tests for rag_engine.

No network calls: the OpenAI client is never constructed here, and the functions
under test are the deterministic ones. Run from the repository root:

    python -m pip install -r requirements-dev.txt
    python -m pytest
"""
import numpy as np
import pandas as pd
import pytest

import rag_engine


# ── Synthetic data ───────────────────────────────────────────────────────────

def test_clickstream_is_deterministic_for_a_fixed_seed():
    a = rag_engine.generate_synthetic_clickstream(n_users=5, seed=42)
    b = rag_engine.generate_synthetic_clickstream(n_users=5, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_demographics_schema_and_user_mapping():
    clicks = rag_engine.generate_synthetic_clickstream(n_users=5, seed=42)
    users = clicks["user_id"].unique()
    demo = rag_engine.generate_demographics(users, seed=42)

    expected = {"user_id", "age", "gender", "location", "primary_interest", "annual_income"}
    assert expected.issubset(set(demo.columns))
    # exactly one demographic record per user, and no strays
    assert set(demo["user_id"]) == set(users)
    assert len(demo) == len(users)


# ── Semantic context ─────────────────────────────────────────────────────────

def _one_profile(duration_s=742):
    """A single clickstream row plus its matching demographic record."""
    click_row = pd.Series(
        {
            "user_id": "u1",
            "timestamp": pd.Timestamp("2026-09-16 10:30:00"),
            "session_duration_s": duration_s,
            "device_type": "Desktop",
            "journey_stage": "Purchase Intent",
        }
    )
    demographics_df = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "age": 34,
                "gender": "Female",
                "location": "San Francisco",
                "primary_interest": "Tech",
                "annual_income": "High",
            }
        ]
    )
    return click_row, demographics_df


def test_context_reports_session_duration_not_timestamp():
    """Regression test for the defect where the event timestamp was emitted as seconds.

    The sentence reads "on this step for approximately N seconds", so N must be the
    session duration. Before the fix this produced
    "for approximately 2026-09-16 10:30:00 seconds".
    """
    click_row, demographics_df = _one_profile(duration_s=742)
    context = rag_engine.stringify_user_context(click_row, demographics_df)

    assert "approximately 742 seconds" in context
    # the timestamp must not leak into the duration slot in any form
    assert "2026-09-16" not in context
    assert "10:30:00" not in context


def test_context_returns_none_when_demographics_are_missing():
    click_row, _ = _one_profile()
    empty = pd.DataFrame(columns=["user_id", "age", "gender", "location",
                                  "primary_interest", "annual_income"])
    assert rag_engine.stringify_user_context(click_row, empty) is None


# ── Retrieval ────────────────────────────────────────────────────────────────

def test_cosine_similarity_handles_zero_vectors():
    zero = np.zeros(3, dtype=np.float32)
    other = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert rag_engine._cosine_similarity(zero, other) == 0.0
    assert rag_engine._cosine_similarity(zero, zero) == 0.0
    assert rag_engine._cosine_similarity(other, other) == pytest.approx(1.0)


def test_search_respects_threshold_ordering_and_top_k():
    store = rag_engine.InMemoryVectorStore()
    store.add("exact", "exact match", [1.0, 0.0])
    store.add("partial", "partial match", [0.7071, 0.7071])   # ~0.707 similarity
    store.add("orthogonal", "no match", [0.0, 1.0])           # ~0.0 similarity
    assert len(store) == 3

    results = store.search([1.0, 0.0], top_k=3, threshold=0.4)

    # the orthogonal record falls below the threshold
    assert [r["user_id"] for r in results] == ["exact", "partial"]
    # descending by similarity
    assert results[0]["similarity"] >= results[1]["similarity"]
    # top_k truncates
    assert len(store.search([1.0, 0.0], top_k=1, threshold=0.4)) == 1
    # an empty store returns nothing rather than raising
    assert rag_engine.InMemoryVectorStore().search([1.0, 0.0]) == []
