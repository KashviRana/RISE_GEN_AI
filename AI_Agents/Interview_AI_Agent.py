from dotenv import load_dotenv
import os

# Document Loader + Vector DB
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LCEL Components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


# Load ENV
load_dotenv()

# LLM (Groq)
from langchain_groq import ChatGroq


# Loading LLM (AI model)
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# KNOWLEDGE BASE (RAG DATA)
text_data = """
Artificial Intelligence is simulation of human intelligence.
Machine Learning is subset of AI that learns from data.
Deep Learning uses neural networks with many layers.
Overfitting means model works well on training data but poorly on test data.
CNN is used for image processing tasks.
RNN is used for sequential data like text.
NLP helps machines understand language.
LangChain is a framework for building LLM applications.
RAG means Retrieval Augmented Generation.
FAISS is a vector database used for similarity search.
"""

# Convert text into Document
documents = [Document(page_content=text_data)]

# Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)


# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Vector Store
vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# PROMPT
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI Interview Assistant. "
        "Answer ONLY using the given context. "
        "If answer is not in context, say 'I don't know'."
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}"
    )
])


# Context function
def get_context(inp: dict) -> str:
    docs = retriever.invoke(inp["question"])
    return "\n\n".join(d.page_content for d in docs)


# LCEL RAG CHAIN
rag_chain = (
    RunnablePassthrough.assign(context=get_context)
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)


# CHAT LOOP
print("\nAI Interview RAG Chatbot Ready...")
print("Type 'exit' to stop\n")

while True:
    query = input("Ask: ")

    if query.lower() == "exit":
        break

    result = rag_chain.invoke({"question": query})

    print("\nAnswer:")
    print(result)
    print()