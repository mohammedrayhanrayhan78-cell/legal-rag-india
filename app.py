import os
import pdfplumber
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
# Step 1 - Load PDF
print("Loading PDF...")
full_text = ""
with pdfplumber.open("CONSTITUTION OF INDIA.pdf") as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"

# Step 2 - Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(full_text)

# Step 3 - Vector database
print("Creating vector database...")
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_texts(chunks, embeddings)
print("Database ready!")
print("\n--- Legal AI Assistant ---")
print("Ask any question about the Indian Constitution.")
print("Type 'exit' to quit.\n")

client = Groq()

# Step 4 - Chat loop
while True:
    question = input("You: ")
    
    if question.lower() == "exit":
        print("Goodbye!")
        break
    
    if question.strip() == "":
        continue

    # Retrieve relevant chunks
    docs = db.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # Get answer from AI
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"You are a legal assistant for the Indian Constitution. Answer based only on this context:\n{context}"},
            {"role": "user", "content": question}
        ]
    )

    print(f"\nAI: {response.choices[0].message.content}\n")