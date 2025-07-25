import streamlit as st
import pandas as pd
import ollama
import re
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from streamlit_javascript import st_javascript


# --- Sticky Full-Width Banner Header ---
def get_base64_logo(path):
    import base64
    with open(path, "rb") as f:
        data = f.read()
        return base64.b64encode(data).decode()

logo_base64 = get_base64_logo("logo_b.png")

st.markdown(f"""
    <style>
    .sticky-header {{
        position: sticky;
        top: 0;
        z-index: 100;
        background-color: white;
        padding-bottom: 10px;
    }}
    .sticky-header img {{
        width: 100%;
        max-height: 120px;
        object-fit: contain;
    }}
    </style>
    <div class="sticky-header">
        <img src="data:image/png;base64,{logo_base64}" alt="Progenics Header" />
    </div>
""", unsafe_allow_html=True)
# --- App Title ---
#st.title("🧬 Genomics Sales Assistant")

#with header_col2:
    #st.markdown("<h2 style='margin-top: 25px;'>🧬 Genomics Sales Assistant</h2>", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("genomics_tests_fixed.xlsx", sheet_name="Sheet1", engine="openpyxl")
    df.columns = [col.strip().lower() for col in df.columns]
    df.fillna('', inplace=True)
    df['combined'] = df.apply(lambda r: f"{r['test name']} {r['panel details']} {r['methodology']} {r['sample type']}", axis=1)
    return df

df = load_data()

# --- Embeddings Setup ---
@st.cache_resource
def setup_embeddings(data):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(data['combined'].tolist(), show_progress_bar=True)
    nn = NearestNeighbors(n_neighbors=3, metric='cosine')
    nn.fit(embeddings)
    return model, nn, embeddings

embed_model, embed_index, embed_vectors = setup_embeddings(df)

# --- Semantic Search ---
def semantic_search(query, top_k=3):
    query_vec = embed_model.encode([query])
    distances, indices = embed_index.kneighbors(query_vec, n_neighbors=top_k)
    return df.iloc[indices[0]]

# --- Context Formatting ---
def format_row(row):
    fields = ["test name", "service code", "panel details", "methodology", "sample type", "tat", "mrp"]
    info = []
    for col in fields:
        if col in row and row[col]:
            val = f"₹{float(row[col]):,.0f}" if col == "mrp" and pd.notna(row[col]) else row[col]
            info.append(f"**{col.replace('_',' ').title()}:** {val}")
    return "\n".join(info)

def format_context(rows):
    return "\n\n".join([format_row(row) for _, row in rows.iterrows()])

# --- LLM Interaction ---
def ask_llm(prompt, context, history=[]):
    system_prompt = f"""
You are a friendly genomics test sales assistant. You will help with understanding test details, price (₹), sample type, methodology, service code, and TAT. Use only the provided context. Be polite, helpful, and ask one clarification question if needed.

Context:
{context}
"""
    messages = [{"role": "system", "content": system_prompt}]
    for h in history[-4:]:
        messages.append({"role": "user", "content": h['user']})
        messages.append({"role": "assistant", "content": h['assistant']})
    messages.append({"role": "user", "content": prompt})
    response = ollama.chat(model="gemma3n:latest", messages=messages, options={"host": "http://localhost:11434"})
    return re.sub(r'<think>.*?</think>', '', response['message']['content'], flags=re.DOTALL).strip()

# --- Chat Interface ---
if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if query := st.chat_input("Ask about a test, cost, TAT, or sample type..."):
    st.chat_message("user").markdown(query)
    st.session_state.chat.append({"role": "user", "content": query})

    matches = semantic_search(query, top_k=3)
    context = format_context(matches)
    history = [
        dict(user=st.session_state.chat[i]['content'], assistant=st.session_state.chat[i+1]['content'])
        for i in range(0, len(st.session_state.chat)-1, 2)
    ]

    with st.spinner("Thinking..."):
        if not matches.empty:
            reply = ask_llm(query, context, history)
        else:
            reply = "Sorry, I couldn't find that test. Could you try using a different keyword or service code?"

    st.chat_message("assistant").markdown(reply)
    st.session_state.chat.append({"role": "assistant", "content": reply})

# --- Sidebar ---
st.sidebar.header("🧪 Training Mode & Filters")
st.sidebar.write(f"Total tests: {len(df)}")
if 'mrp' in df.columns:
    st.sidebar.metric("Avg Price", f"₹{df['mrp'].mean():,.0f}")

if st.sidebar.checkbox("Enable Training Mode"):
    st.subheader("Sales Quiz")
    sample = df.sample(1).iloc[0]
    question = f"What is the TAT for **{sample['test name']}**?"
    user_answer = st.text_input(question)
    if user_answer:
        correct = str(sample['tat']).strip().lower()
        if user_answer.strip().lower() == correct:
            st.success("✅ Correct!")
        else:
            st.error(f"❌ Nope! It's: {correct}")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat = []
    st.rerun()

