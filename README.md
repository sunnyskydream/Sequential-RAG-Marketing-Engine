**🚀 Sequential RAG Marketing Engine**\
AI-powered personalization engine combining clickstream + demographic data for intent-aware marketing

📌 Overview\
This project explores how marketing teams can move beyond static segmentation by combining user demographics with behavioral signals (clickstream data).

Using a Retrieval-Augmented Generation (RAG) approach, the system transforms raw user data into semantic understanding, enabling dynamic personalization across:
Email targeting,
Product recommendations,
Landing page personalization,
Campaign messaging.


🎯 Key Features\
🔍 Semantic user representation via embeddings\
⚡ In-memory vector retrieval — no cloud database required\
✨ RAG-based personalized content generation\
🖥️ Interactive Streamlit UI with Setup, Query, and Explore tabs\
🧪 Original prototyping notebook included for reference


🏗️ Architecture\
User Data (Synthetic Clickstream + Demographics)\
        ↓\
Semantic Stringification\
        ↓\
Embedding Layer (OpenAI text-embedding-3-small)\
        ↓\
In-Memory Vector Store (NumPy cosine similarity)\
        ↓\
Retrieval (Relevant User Context)\
        ↓\
RAG Content Generation (GPT-4o)
<img width="1536" height="1024" alt="RAG marketing engine" src="https://github.com/user-attachments/assets/a118d70d-9139-47bb-ab78-1649b8a85652" />

📊 Dataset\
Clickstream Data: Synthetic data generated locally — mirrors the structure of the Kaggle E-commerce Customer Journey dataset (no Kaggle account required)\
Demographic Data: Synthesized and joined via user_id, simulating CDP-like user profiles


🛠️ Tech Stack\
Embeddings: OpenAI text-embedding-3-small\
Vector Store: In-memory NumPy cosine similarity (replaces Supabase/pgvector)\
UI: Streamlit (local)\
Language: Python


⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/sunnyskydream/Sequential-RAG-Marketing-Engine.git
cd Sequential-RAG-Marketing-Engine

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your OpenAI API key
cp .env.example .env
# Edit .env and replace the placeholder with your real key

# 4. Launch the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

**API key:** you can also skip the `.env` step and paste your key directly into the sidebar when the app loads.


🎬 Demo

<video src="RAG%20demo.mp4" width="100%" controls title="Sequential RAG Marketing Engine Demo"></video>

> [▶️ Click here to watch the demo](RAG%20demo.mp4) if the video does not render above.


🖥️ Using the App

| Tab | What it does |
| --- | --- |
| ⚙️ Setup & Index | Configure your brand → generate synthetic users → embed their profiles into the vector store |
| 📊 Explore Data | Browse demographics, clickstream events, and semantic context strings |
| 🔍 Query & Generate | Describe a user's behavior → retrieve similar profiles → generate tailored marketing copy |


💡 Example Use Case\
Input:\
High-income professional user\
Browsing high-end products with long dwell time

Output:\
Tailored product recommendation\
Intent-aware messaging (exploration vs purchase stage)


⚡ Before vs After: Traditional vs AI-Driven Personalization\
❌ Traditional Marketing (Static Segmentation)\
Approach:\
Define audience rules manually\
"Female, 30–50, high income"\
Pre-write fixed messaging\
Apply same content to entire segment

Limitations\
Ignores real-time behavior\
Slow to update\
Requires manual campaign setup\
Low personalization depth

✅ AI-Driven (RAG-Based Personalization)\
Approach:\
Combine demographic + clickstream data\
Use embeddings to represent user intent\
Retrieve relevant context dynamically\
Generate content in real time

Advantages:\
Behavior + intent-aware messaging\
Dynamic personalization at scale\
Faster iteration cycles\
Reduces manual segmentation effort

🔍 Example Comparison
| Scenario                                   | Traditional Output               | RAG-Based Output                                                         |
| ------------------------------------------ | -------------------------------- | ------------------------------------------------------------------------ |
| High-income user browsing premium products | Generic "Explore our collection" | "Upgrade your setup with premium performance designed for professionals" |
| User early in journey                      | Same promo messaging             | Educational / discovery-focused messaging                                |
| User near conversion                       | Same messaging                   | Urgency + product-specific recommendation                                |


🧠 Key Takeaway\
👉 Traditional marketing asks:
"Which segment does this user belong to?"\
👉 This system asks:
"What is this user trying to do right now?"


⚠️ Limitations\
Synthetic demographic data (no real CDP integration)\
No real-time streaming pipeline\
In-memory vector store resets on app restart (not persistent)\
Embedding model optimized for cost over maximum accuracy


🔮 Future Improvements\
Real-time event streaming integration\
CDP integration\
Persistent vector database (e.g. Supabase, Pinecone)\
Advanced prompt optimization & evaluation\
GA4 MCP integration for live agentic workflows


⚙️ Development Notes\
Original prototype built in Google Colab — see `Sequential_RAG_Marketing_Engine.ipynb`\
Local version replaces Supabase/pgvector with an in-memory NumPy store for zero-config setup\
API key is entered at runtime via the Streamlit sidebar and is never stored in code


📋 What Changed in This Version

**v2 — Brand-Agnostic Update (May 2025)**

| Area | Change |
| --- | --- |
| Brand configuration | Sidebar now accepts any brand name, brand context description, and product categories — no longer hardcoded to Corsair |
| Copy generation | `generate_marketing_content()` takes `brand_name`, `brand_context`, and `product_categories` as inputs; GPT-4o tailors all copy to the specified brand |
| Synthetic data | `generate_synthetic_clickstream()` and `generate_demographics()` accept custom product categories and interest lists — defaults are now brand-neutral |
| Tab order | Reordered to: Setup & Index → Explore Data → Query & Generate for a more natural workflow |
| Preset queries | Removed Corsair-specific references; presets are now generic behavioral profiles |
| Demo video | Added `RAG demo.mp4` walkthrough |

**v1 — Streamlit UI (April 2025)**

| Section | Before | After |
| --- | --- | --- |
| Key Features | "notebook environment" | Streamlit UI with 3 tabs |
| Architecture | Supabase vector DB | In-memory NumPy cosine similarity |
| Dataset | Kaggle (requires account) | Synthetic local data, no account needed |
| Tech Stack | Supabase + Google Colab | In-memory store + local Streamlit |
| Quick Start | *(not present)* | Full install + run instructions added |
| App Tab Guide | *(not present)* | Setup / Query / Explore tab table added |
