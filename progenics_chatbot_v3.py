import streamlit as st
import pandas as pd
import ollama
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import base64

# --- Page Configuration ---
st.set_page_config(page_title="Genomics Sales Assistant", layout="wide")

# --- Force Light Mode and Custom Styling ---
st.markdown("""
<style>
/* Title */
h3 {
    color: #334A7D;
    font-weight: 600;
    font-size: 1.5rem;
    text-align: center;
    margin-top: 0.5em;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #EAF4FA;
}
.css-1d391kg, .css-10trblm, .css-1v0mbdj {
    color: #334A7D !important;
}

/* Buttons and Inputs */
button {
    border: 1px solid #83BBE2 !important;
    background-color: #ffffff !important;
    color: #334A7D !important;
}
button:hover {
    background-color: #83BBE2 !important;
    color: white !important;
}
.stTextInput>div>div>input {
    border: 1px solid #83BBE2 !important;
    background-color: white !important;
    color: #334A7D !important;
}

/* Metrics */
.css-1dp5vir { color: #7F8284; }
.css-1p05y26 { color: #334A7D; }

/* Chat bubbles */
.chat-message {
    background-color: white;
    border-left: 5px solid #83BBE2;
    padding: 0.8em;
    border-radius: 8px;
    margin-bottom: 1em;
    color: #334A7D !important;
}
</style>
""", unsafe_allow_html=True)

# --- Header Banner ---
def get_base64_logo(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_logo("logo_b.png")

st.markdown(f"""
<div class="sticky-header">
    <img src="data:image/png;base64,{logo_base64}" alt="Progenics Banner" />
</div>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h3>Genomics Sales Assistant</h3>", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_excel("genomics_tests_fixed.xlsx", sheet_name="Sheet1", engine="openpyxl")
    df.columns = [col.strip().lower() for col in df.columns]
    df.fillna('', inplace=True)
    df['combined'] = df.apply(lambda r: f"{r['test name']} {r['panel details']} {r['methodology']} {r['sample type']}", axis=1)
    return df

df = load_data()

# --- Setup Embeddings ---
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

# --- Chat UI ---
if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg['role']):
        st.markdown(f"<div class='chat-message'>{msg['content']}</div>", unsafe_allow_html=True)

if query := st.chat_input("Ask about a test, cost, TAT, or sample type..."):
    st.chat_message("user").markdown(f"<div class='chat-message'>{query}</div>", unsafe_allow_html=True)
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

    st.chat_message("assistant").markdown(f"<div class='chat-message'>{reply}</div>", unsafe_allow_html=True)
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

