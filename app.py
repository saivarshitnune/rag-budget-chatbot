from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from sentence_transformers import CrossEncoder
import os

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (modify for security)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google Gemini embeddings & LLM
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key="")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, api_key="")

# pdf_loader = PyPDFLoader(r"C:\Users\vahin\OneDrive\Desktop\rag-budget\budget_speech.pdf")
# documents = pdf_loader.load()
pdf_folder = r"C:\Users\vahin\OneDrive\Desktop\rag-budget\data-pdfs"

# Get all PDF file paths in the folder
pdf_files = [os.path.join(pdf_folder, file) for file in os.listdir(pdf_folder) if file.endswith(".pdf")]

# Extract text from all PDFs
documents = []
for pdf_file in pdf_files:
    pdf_loader = PyPDFLoader(pdf_file)
    documents.extend(pdf_loader.load())

# Extract raw text from the documents
raw_text = "\n\n".join([doc.page_content for doc in documents])

# Define headers for splitting based on Markdown structure
headers_to_split_on = [
    ("#", "Main Title"),
    ("##", "Section"),
    ("###", "Subsection"),
]

# Use MarkdownHeaderTextSplitter
text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
split_docs = text_splitter.split_text(raw_text)  # Correct method usage
# Load pre-existing vector store
vector_store = Chroma.from_documents(split_docs, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# Cross-Encoder for reranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_documents(query, retrieved_docs):
    """Re-rank retrieved documents using Cross-Encoder."""
    doc_texts = [doc.page_content for doc in retrieved_docs]
    model_inputs = [[query, doc] for doc in doc_texts]
    scores = cross_encoder.predict(model_inputs)
    doc_score_pairs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in doc_score_pairs]

@app.post("/chat")
async def chat(query: dict):
    global retriever
    
    user_query = query["query"]
    retrieved_docs = retriever.get_relevant_documents(user_query)

    # Re-rank documents
    reranked_docs = rerank_documents(user_query, retrieved_docs)

    # Generate response using RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    response = qa_chain.invoke({"query": f"{user_query}\nContext: {reranked_docs}"})
    
    return {"response": response["result"]}
