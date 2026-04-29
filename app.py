"""
app.py — Streamlit frontend for the Sequential RAG Marketing Engine.

Workflow:
  1. Setup tab  → generate synthetic data + build in-memory vector index
  2. Query tab  → enter a user behavior description → retrieve similar users → GPT-4o marketing copy
  3. Explore tab → browse the raw demographics, clickstream, and semantic contexts
"""

import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # picks up OPENAI_API_KEY from a local .env file if present

from rag_engine import (
    InMemoryVectorStore,
    build_vector_store,
    generate_demographics,
    generate_marketing_content,
    generate_synthetic_clickstream,
    get_embedding,
    stringify_user_context,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sequential RAG Marketing Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────

for key in ("vector_store", "context_df", "demographics_df", "clickstream_df"):
    if key not in st.session_state:
        st.session_state[key] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚀 Sequential RAG\nMarketing Engine")
    st.caption("Personalized Corsair marketing via behavior-aware RAG")
    st.divider()

    api_key = st.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-...",
        help="Set OPENAI_API_KEY in a local .env file, or paste it here. Never committed to git.",
    )

    st.divider()
    st.markdown("#### Settings")
    n_users = st.slider("Synthetic users to generate", 50, 500, 200, step=50)
    top_k = st.slider("Top-K retrieval results", 1, 5, 3)
    sim_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.40, step=0.05)

    st.divider()
    if st.session_state.vector_store is not None:
        st.success(f"Index ready — {len(st.session_state.vector_store)} vectors")
    else:
        st.info("No index built yet.\nSee the **Setup** tab.")

# ── Helper ────────────────────────────────────────────────────────────────────

def make_client() -> OpenAI:
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        st.stop()
    return OpenAI(api_key=api_key)

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_setup, tab_query, tab_explore = st.tabs(
    ["⚙️ Setup & Index", "🔍 Query & Generate", "📊 Explore Data"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Setup & Index
# ─────────────────────────────────────────────────────────────────────────────

with tab_setup:
    st.header("Setup & Index")
    st.markdown(
        "**Step 1** generates synthetic users that mirror the Kaggle e-commerce "
        "clickstream dataset (no Kaggle account needed). "
        "**Step 2** embeds each user's semantic profile and builds an in-memory vector index."
    )

    col_gen, col_idx = st.columns(2)

    # ── Step 1: generate data ──────────────────────────────────────────────
    with col_gen:
        if st.button("1️⃣  Generate Synthetic Data", use_container_width=True):
            with st.spinner(f"Generating {n_users} users…"):
                clickstream_df = generate_synthetic_clickstream(n_users=n_users)
                demographics_df = generate_demographics(
                    clickstream_df["user_id"].unique(), seed=42
                )

                # One context row per user: their most recent clickstream event
                latest = (
                    clickstream_df.sort_values("timestamp")
                    .groupby("user_id")
                    .last()
                    .reset_index()
                )
                latest["semantic_context"] = latest.apply(
                    lambda row: stringify_user_context(row, demographics_df), axis=1
                )

                st.session_state.clickstream_df = clickstream_df
                st.session_state.demographics_df = demographics_df
                st.session_state.context_df = latest
                st.session_state.vector_store = None  # reset stale index

            st.success(
                f"Generated **{n_users} users** across "
                f"**{len(clickstream_df)} clickstream events**."
            )

    # ── Step 2: build index ────────────────────────────────────────────────
    with col_idx:
        index_disabled = st.session_state.context_df is None
        if st.button(
            "2️⃣  Build Vector Index",
            use_container_width=True,
            disabled=index_disabled,
            help="Generate data first, then add your API key.",
        ):
            client = make_client()
            df = st.session_state.context_df

            progress_bar = st.progress(0.0, text="Starting…")

            def _update(current: int, total: int) -> None:
                pct = current / total
                progress_bar.progress(pct, text=f"Embedding user {current}/{total}…")

            try:
                store = build_vector_store(df, client, progress_callback=_update)
                st.session_state.vector_store = store
                progress_bar.progress(1.0, text="Done!")
                st.success(f"Index built — **{len(store)} vectors** stored in memory.")
            except Exception as exc:
                st.error(f"Error building index: {exc}")

    # ── Sample preview ─────────────────────────────────────────────────────
    if st.session_state.context_df is not None:
        st.divider()
        st.subheader("Sample Semantic Contexts")
        sample = st.session_state.context_df[["user_id", "semantic_context"]].head(5)
        for _, row in sample.iterrows():
            with st.expander(f"User: {row['user_id']}"):
                st.write(row["semantic_context"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Query & Generate
# ─────────────────────────────────────────────────────────────────────────────

with tab_query:
    st.header("Query & Generate Marketing Content")

    if st.session_state.vector_store is None:
        st.warning("Build the vector index in the **Setup** tab first.")
    else:
        # Preset queries taken directly from the notebook
        PRESETS: dict[str, str] = {
            "Early Exploration — Gaming, Mobile, Low income": (
                "This is a 38-year-old Male living in New York "
                "with a primary interest in Gaming and a Low income level. "
                "He is currently using a Mobile device and has been on this step "
                "for approximately 2025-03-31 11:11:46 seconds. "
                "Based on his behavior pattern, he is in the early exploration stage of product discovery."
            ),
            "Consideration Stage — Gaming, Desktop, Medium income": (
                "This is a 29-year-old Male living in California "
                "with a primary interest in Gaming and a Medium income level. "
                "He is currently using a Desktop device and has been on this step "
                "for approximately 2025-04-10 20:45:12 seconds. "
                "Based on his behavior pattern, he is comparing multiple products and reviewing "
                "detailed specifications, indicating a mid-to-high purchase intent."
            ),
            "High Intent — Professional, Desktop, High income": (
                "This is a 42-year-old Female living in Texas "
                "with a primary interest in Professional use and a High income level. "
                "She is currently using a Desktop device and has been on this step "
                "for approximately 2025-05-18 09:32:55 seconds. "
                "Based on her behavior pattern, she has revisited the same product multiple times "
                "and is checking pricing and reviews, indicating strong purchase intent."
            ),
            "Ultra-High Income Prospect — Professional, Tablet": (
                "This is a 40-year-old Female living in San Francisco "
                "with a primary interest in Professional use and an Ultra High income level. "
                "She is currently using a Tablet device and has been on this step "
                "for approximately 2025-02-26 12:57:10 seconds. "
                "Based on her behavior pattern, she is in the early exploration stage of product discovery."
            ),
            "Esports Enthusiast — Desktop, High income, Conversion-ready": (
                "This is a 24-year-old Male living in Seattle "
                "with a primary interest in Esports and a High income level. "
                "He is currently using a Desktop device and has been on this step "
                "for approximately 2025-06-05 18:22:33 seconds. "
                "Based on his behavior pattern, he is completing a purchase transaction, "
                "showing high conversion readiness."
            ),
        }

        col_left, col_right = st.columns([4, 2])
        with col_right:
            preset_choice = st.selectbox(
                "Load a preset query", ["— Custom —"] + list(PRESETS.keys())
            )
        with col_left:
            default_val = PRESETS.get(preset_choice, "")
            query_text = st.text_area(
                "User Behavior Description",
                value=default_val,
                height=130,
                placeholder="Describe a user's demographics and current browsing behavior…",
            )

        run_disabled = not query_text.strip()
        if st.button("🔍  Search & Generate", type="primary", disabled=run_disabled):
            client = make_client()

            with st.spinner("Vectorizing query and searching index…"):
                try:
                    qv = get_embedding(query_text.strip(), client)
                    results = st.session_state.vector_store.search(
                        qv, top_k=top_k, threshold=sim_threshold
                    )
                except Exception as exc:
                    st.error(f"Search error: {exc}")
                    results = []

            if not results:
                st.warning(
                    "No users found above the similarity threshold. "
                    "Try lowering the **Similarity threshold** in the sidebar."
                )
            else:
                st.subheader(f"Retrieved {len(results)} similar user(s)")

                for i, match in enumerate(results):
                    expanded = i == 0
                    label = (
                        f"Match #{i+1} — User {match['user_id']} "
                        f"(similarity: {match['similarity']:.4f})"
                    )
                    with st.expander(label, expanded=expanded):
                        st.markdown("**Retrieved context:**")
                        st.info(match["content"])

                        if i == 0:
                            st.markdown("---")
                            with st.spinner("Generating Corsair marketing copy via GPT-4o…"):
                                try:
                                    advice = generate_marketing_content(match["content"], client)
                                    st.markdown("#### Corsair AI Marketing Recommendations")
                                    st.markdown(advice)
                                except Exception as exc:
                                    st.error(f"Content generation error: {exc}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Explore Data
# ─────────────────────────────────────────────────────────────────────────────

with tab_explore:
    st.header("Explore Generated Data")

    if st.session_state.demographics_df is None:
        st.info("Generate synthetic data in the **Setup** tab to browse it here.")
    else:
        sub_demo, sub_click, sub_ctx = st.tabs(
            ["👤 Demographics", "🖱️ Clickstream", "📝 Semantic Contexts"]
        )

        with sub_demo:
            df_d = st.session_state.demographics_df
            st.caption(f"{len(df_d)} users")

            # Summary charts
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.bar_chart(df_d["gender"].value_counts(), use_container_width=True)
                st.caption("Gender distribution")
            with col_b:
                st.bar_chart(df_d["primary_interest"].value_counts(), use_container_width=True)
                st.caption("Primary interest")
            with col_c:
                st.bar_chart(df_d["annual_income"].value_counts(), use_container_width=True)
                st.caption("Income level")

            st.divider()
            st.dataframe(df_d, use_container_width=True, hide_index=True)

        with sub_click:
            df_c = st.session_state.clickstream_df
            st.caption(f"{len(df_c)} events across {df_c['user_id'].nunique()} users")

            col_x, col_y = st.columns(2)
            with col_x:
                st.bar_chart(df_c["journey_stage"].value_counts(), use_container_width=True)
                st.caption("Journey stage distribution")
            with col_y:
                st.bar_chart(df_c["device_type"].value_counts(), use_container_width=True)
                st.caption("Device type distribution")

            st.divider()
            st.dataframe(df_c, use_container_width=True, hide_index=True)

        with sub_ctx:
            df_s = st.session_state.context_df[
                ["user_id", "device_type", "journey_stage", "semantic_context"]
            ]
            st.caption(f"{len(df_s)} semantic contexts (one per user, latest event)")
            st.dataframe(df_s, use_container_width=True, hide_index=True)
