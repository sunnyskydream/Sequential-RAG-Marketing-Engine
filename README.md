**🚀 Sequential RAG Engine**\
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
⚡ Vector-based retrieval of user context\
✨ RAG-based personalized content generation\
🧪 Interactive prototyping via notebook environment\

🏗️ Architecture\
User Data (Clickstream + Demographics)\
        ↓\
Embedding Layer (OpenAI)\
        ↓\
Vector Database (Supabase)\
        ↓\
Retrieval (Relevant Context)\
        ↓\
RAG (Content Generation)
<img width="1536" height="1024" alt="RAG marketing engine" src="https://github.com/user-attachments/assets/a118d70d-9139-47bb-ab78-1649b8a85652" />

📊 Dataset\
Clickstream Data (Source: Kaggle E-commerce customer journey dataset)\
Demographic Data (Synthesized and joined via user_id
Simulates CDP-like user profiles)\

🛠️ Tech Stack\
Embeddings: OpenAI (text-embedding-3-small)\
Vector DB: Supabase\
Prototyping: Google Colab\
Language: Python

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
“Female, 30–50, high income”\
Pre-write fixed messaging\
Apply same content to entire segment\

Limitations\
Ignores real-time behavior\
Slow to update\
Requires manual campaign setup\
Low personalization depth\

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
| High-income user browsing premium products | Generic “Explore our collection” | “Upgrade your setup with premium performance designed for professionals” |
| User early in journey                      | Same promo messaging             | Educational / discovery-focused messaging                                |
| User near conversion                       | Same messaging                   | Urgency + product-specific recommendation                                |


🧠 Key Takeaway\
👉 Traditional marketing asks:
“Which segment does this user belong to?”\
👉 This system asks:
“What is this user trying to do right now?”


⚠️ Limitations\
Synthetic demographic data\
No real-time streaming pipeline\
UI layer (Streamlit) not deployed in current version\
Embedding model optimized for cost over accuracy\

🔮 Future Improvements\
Deploy full UI (Streamlit / web app)\
Real-time event streaming integration\
CDP integration\
Advanced vector DB scaling\
Prompt optimization & evaluation

⚙️ Development Notes\
Prototyped in Google Colab due to local environment dependency constraints\
Streamlit UI was initially planned for interactive simulation, but not fully implemented due to package compatibility limitations in Colab\
Current version focuses on core RAG pipeline and personalization logic
