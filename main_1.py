from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import MarkdownHeaderTextSplitter
# from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os

# Set up Google API key (Ensure you have it set in your environment variables)
#os.environ["GOOGLE_API_KEY"] = "your_google_api_key"

# Load and process PDF documents


# Load the PDF
pdf_loader = PyPDFLoader(r"C:\Users\vahin\OneDrive\Desktop\rag-budget\budget_speech.pdf")
documents = pdf_loader.load()

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
# # print(split_docs)

# print(split_docs[0].page_content)

# Initialize embeddings model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",api_key="AIzaSyB4U_WEGXADoGvetHg4cf4tMrBTDBVALqQ")

# Store embeddings in FAISS
# vector_store = FAISS.from_documents(docs, embedding_model)
vector_store = Chroma.from_documents(split_docs, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query, retrieved_docs):
    """Re-rank retrieved documents using Cross-Encoder."""
    doc_texts = [doc.page_content for doc in retrieved_docs]  # Extract text
    model_inputs = [[query, doc] for doc in doc_texts]  # Prepare input pairs
    scores = cross_encoder.predict(model_inputs)  # Get relevance scores
    doc_score_pairs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in doc_score_pairs]


# Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a financial AI assistant specializing in government budgets and financial reports. 
    Your task is to analyze the provided budget speech excerpts and generate a concise, fact-based response.

    **Context:**  
    {context}  

    **User's Question:**  
    {question}  

    **Instructions:**  
    - **Extract key insights** from the provided context that directly answer the question.  
    - **Include financial figures, percentages, or key policies** if relevant.  
    - **Avoid speculation**—only use the information given in the context.  
    - **If the context lacks sufficient details, say: "The provided information does not contain enough details to answer this question."**  
    - **Structure your response logically** to enhance readability.  

    **Final Response:**
    """
)

# Initialize LLM (Google Gemini 2.0 Flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7,api_key="AIzaSyB4U_WEGXADoGvetHg4cf4tMrBTDBVALqQ")

# Set up memory for conversation persistence
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define RAG pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    memory=memory,
    chain_type_kwargs={"prompt": prompt_template}
)

# Chatbot interaction loop
def chat():
    print("Welcome to the RAG Chatbot! Type 'exit' to quit.")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        retrieved_docs = retriever.get_relevant_documents(query)

        # Re-rank documents for better relevance
        reranked_docs = rerank_documents(query, retrieved_docs)

        # Pass only re-ranked docs to LLM
        response = qa_chain.invoke({"query": f"{query}\nContext: {reranked_docs}"})
        # response = qa_chain.run(query)
        #response = qa_chain.invoke({"query": query})
        print("Bot:", response['result'])

# Run chatbot
if __name__ == "__main__":
    chat()
