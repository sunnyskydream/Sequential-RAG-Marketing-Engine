"""
rag_engine.py — Core logic for the Sequential RAG Marketing Engine.

Replicates the notebook pipeline locally:
  synthetic data → semantic stringification → OpenAI embeddings →
  in-memory vector store → cosine-similarity retrieval → GPT-4o content generation
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Callable, List, Optional

import numpy as np
import pandas as pd


# ── Data generation ──────────────────────────────────────────────────────────

def generate_synthetic_clickstream(
    n_users: int = 200,
    seed: int = 42,
    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate synthetic e-commerce clickstream data mimicking the Kaggle dataset."""
    random.seed(seed)
    np.random.seed(seed)

    devices = ["Desktop", "Mobile", "Tablet"]
    stages = ["Awareness", "Consideration", "Purchase Intent", "Conversion"]
    product_categories = categories or [
        "Electronics", "Home & Lifestyle", "Fashion", "Sports & Outdoors", "Beauty & Health"
    ]

    user_ids = [f"CUST_{i:04d}" for i in range(1, n_users + 1)]
    base_time = datetime(2025, 1, 1)
    rows = []

    for uid in user_ids:
        n_events = random.randint(2, 5)
        for _ in range(n_events):
            ts = base_time + timedelta(
                days=random.randint(0, 120),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
                seconds=random.randint(0, 59),
            )
            rows.append(
                {
                    "user_id": uid,
                    "timestamp": ts,
                    "device_type": random.choice(devices),
                    "journey_stage": random.choice(stages),
                    "session_duration_s": random.randint(60, 1800),
                    "pages_viewed": random.randint(1, 20),
                    "category": random.choice(product_categories),
                }
            )

    return pd.DataFrame(rows)


def generate_demographics(
    user_ids,
    seed: int = 42,
    interests: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate synthetic user demographic profiles for the given user IDs."""
    random.seed(seed)

    income_levels = ["Low", "Medium", "High", "Ultra High"]
    primary_interests = interests or ["Lifestyle", "DIY", "Professional", "Fitness", "Tech"]
    genders = ["Male", "Female", "Non-binary"]
    locations = [
        "San Jose", "San Francisco", "New York",
        "Austin", "Seattle", "Los Angeles", "Chicago",
    ]

    data = []
    for uid in user_ids:
        data.append(
            {
                "user_id": uid,
                "age": random.randint(18, 55),
                "gender": random.choice(genders),
                "location": random.choice(locations),
                "annual_income": random.choice(income_levels),
                "loyalty_score": random.randint(1, 10),
                "primary_interest": random.choice(primary_interests),
            }
        )
    return pd.DataFrame(data)


# ── Semantic stringification ─────────────────────────────────────────────────

_PRONOUNS = {
    "Male": ("He", "his"),
    "Female": ("She", "her"),
    "Non-binary": ("They", "their"),
}

_STAGE_DESCRIPTIONS = {
    "Awareness": "in the early exploration stage of product discovery",
    "Consideration": (
        "comparing multiple products and reviewing detailed specifications, "
        "indicating a mid-to-high purchase intent"
    ),
    "Purchase Intent": (
        "revisiting the same products multiple times and checking pricing and reviews, "
        "indicating strong purchase intent"
    ),
    "Conversion": "completing a purchase transaction, showing high conversion readiness",
}


def stringify_user_context(
    click_row: pd.Series, demographics_df: pd.DataFrame
) -> Optional[str]:
    """
    Convert one clickstream row + its matching demographic record into a natural-language
    semantic description — the format used by the original notebook for embedding.
    """
    uid = click_row["user_id"]
    demo = demographics_df[demographics_df["user_id"] == uid]
    if demo.empty:
        return None

    demo = demo.iloc[0]
    subject, possessive = _PRONOUNS.get(demo["gender"], ("They", "their"))
    stage_desc = _STAGE_DESCRIPTIONS.get(click_row["journey_stage"], "browsing the site")

    ts = click_row["timestamp"]
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)

    return (
        f"This is a {demo['age']}-year-old {demo['gender']} living in {demo['location']} "
        f"with a primary interest in {demo['primary_interest']} "
        f"and a {demo['annual_income']} income level. "
        f"{subject} is currently using a {click_row['device_type']} device and has been "
        f"on this step for approximately {ts_str} seconds. "
        f"Based on {possessive} behavior pattern, {subject.lower()} is {stage_desc}."
    )


# ── Embeddings ───────────────────────────────────────────────────────────────

def get_embedding(text: str, client) -> list[float]:
    """Convert text to a 1 536-dim vector via OpenAI text-embedding-3-small."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small",
    )
    return response.data[0].embedding


# ── In-memory vector store ───────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


class InMemoryVectorStore:
    """Lightweight vector store that replaces Supabase/pgvector for local use."""

    def __init__(self) -> None:
        self._vectors: list[np.ndarray] = []
        self._contents: list[str] = []
        self._user_ids: list[str] = []

    def add(self, user_id: str, content: str, vector: list[float]) -> None:
        self._user_ids.append(user_id)
        self._contents.append(content)
        self._vectors.append(np.array(vector, dtype=np.float32))

    def search(
        self, query_vector: list[float], top_k: int = 3, threshold: float = 0.4
    ) -> list[dict]:
        """Return top-k records with cosine similarity ≥ threshold."""
        if not self._vectors:
            return []

        qv = np.array(query_vector, dtype=np.float32)
        scores = [_cosine_similarity(qv, v) for v in self._vectors]

        results = [
            {"user_id": self._user_ids[i], "content": self._contents[i], "similarity": round(s, 4)}
            for i, s in enumerate(scores)
            if s >= threshold
        ]
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def __len__(self) -> int:
        return len(self._vectors)


# ── Index builder ─────────────────────────────────────────────────────────────

def build_vector_store(
    df: pd.DataFrame,
    client,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> InMemoryVectorStore:
    """
    Embed every row's 'semantic_context' and load it into an InMemoryVectorStore.
    df must have columns: user_id, semantic_context.
    """
    store = InMemoryVectorStore()
    valid = df[df["semantic_context"].notna()].reset_index(drop=True)
    total = len(valid)

    for i, (_, row) in enumerate(valid.iterrows()):
        vector = get_embedding(row["semantic_context"], client)
        store.add(row["user_id"], row["semantic_context"], vector)
        if progress_callback:
            progress_callback(i + 1, total)

    return store


# ── Content generation ────────────────────────────────────────────────────────

def generate_marketing_content(
    context: str,
    client,
    brand_name: str = "Our Brand",
    brand_context: str = "",
    product_categories: Optional[List[str]] = None,
) -> str:
    """Generate intent-aware personalized marketing copy using GPT-4o."""
    category_hint = (
        f"Product categories: {', '.join(product_categories)}. "
        if product_categories
        else ""
    )
    brand_description = (
        f"{brand_context} " if brand_context else f"a brand called {brand_name}. "
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a seasoned senior digital marketing expert at {brand_name}, "
                    "skilled at generating precise, behavior-driven personalized marketing content."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Based on the profile of this similar user:\n\n{context}\n\n"
                    f"Generate the following three marketing outputs for {brand_name} — {brand_description}"
                    f"{category_hint}\n\n"
                    "1. **Email Subject Line** — A high click-rate subject tailored to this user's purchase intent.\n"
                    f"2. **Product Recommendation** — Recommend one specific {brand_name} product "
                    "with a brief rationale.\n"
                    "3. **Call to Action (CTA)** — One compelling short copy line to convert this user.\n\n"
                    "Format your response with clear section headers."
                ),
            },
        ],
    )
    return response.choices[0].message.content
