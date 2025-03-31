from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.vectorstores import FAISS
from langchain_chroma import Chroma
# from langchain.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os

Set up Google API key (Ensure you have it set in your environment variables)
os.environ["GOOGLE_API_KEY"] = "your_google_api_key"

Load and process PDF documents
pdf_loader = PyPDFLoader(r"C:\Users\vahin\OneDrive\Desktop\rag-budget\budget_speech.pdf")  # Replace with your PDF file path
documents = pdf_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50)
docs = text_splitter.split_documents(documents)
print(docs)

# # Initialize embeddings model
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",api_key="AIzaSyB4U_WEGXADoGvetHg4cf4tMrBTDBVALqQ")

# Store embeddings in FAISS
# vector_store = FAISS.from_documents(docs, embedding_model)
vector_store = Chroma.from_documents(documents, embedding_model)
retriever = vector_store.as_retriever()

#Define prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a helpful AI assistant. Use the following retrieved context to answer the question.
    Context: {context}
    Question: {question}
    Provide a concise and accurate response.
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
        # response = qa_chain.run(query)
        response = qa_chain.invoke({"query": query})
        print("Bot:", response['result'])

# Run chatbot
if __name__ == "__main__":
    chat()
