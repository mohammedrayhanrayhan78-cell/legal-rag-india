import os
import streamlit as st
import pdfplumber
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️")
st.title("⚖️ Indian Constitution AI Assistant")
st.caption("Ask anything about the Indian Constitution")

@st.cache_resource
def load_rag():
    full_text = ""
    with pdfplumber.open("CONSTITUTION OF INDIA.pdf") as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)
    return chunks

def simple_search(chunks, question, k=3):
    question_words = set(question.lower().split())
    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(question_words & chunk_words)
        scores.append(score)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in top_indices]

with st.spinner("Loading Constitution..."):
    chunks = load_rag()

st.success("Ready! Ask your question below.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask about the Indian Constitution..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    relevant_chunks = simple_search(chunks, question, k=3)
    context = "\n".join(relevant_chunks)

    client = Groq()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"You are a legal assistant for the Indian Constitution. Answer based only on this context:\n{context}"},
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)