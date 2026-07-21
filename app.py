from dotenv import load_dotenv
load_dotenv()

import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_anthropic import ChatAnthropic
from langchain_classic.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr

# Load documents
def load_documents():
    loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200,separators=["\n\n", "\n", " ", ""])
    documents = text_splitter.split_documents(documents)

    #Debug
    print(f"Loaded {len(documents)} document chunks")
    for i, doc in enumerate(documents[:3]):
        print(f"---Doc {i} ---")
        print(f"Length: {len(doc.page_content)} characters")
        print(f"Preview: {doc.page_content[:200]}")
        
    return documents

# Create vector store
def create_vector_store(documents):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Load or create vectorstore
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local(
        "faiss_index",
        SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True,
    )
else:
    print("Index not found. Creating new index...")
    docs = load_documents()
    vectorstore = create_vector_store(docs)
    vectorstore.save_local("faiss_index")

# Claude model instead of GPT-2
llm = ChatAnthropic(
    model="claude-sonnet-5",
    max_tokens=1024,
    system=(
        "You are a legal AI assistant specializing in Sri Lankan law. "
        "Answer only using the provided context from the documents. "
        "If the answer isn't in the context, say you don't know rather than guessing. "
        "Be precise, consistent, and avoid speculative or creative phrasing."
    ),
)

# Memory for conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)

# Build the retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    return_source_documents=True,
)

# Gradio-facing function
def chat_interface(query, chat_history):
    result = qa_chain.invoke({"question": query})
    answer = result["answer"]
    chat_history = chat_history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer},
    ]
    return chat_history, chat_history


with gr.Blocks() as demo:
    gr.Markdown("# 🇱🇰 Legal AI Sri Lanka")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask me anything about Sri Lankan law...")
    state = gr.State([])

    msg.submit(chat_interface, [msg, state], [chatbot, state])
    msg.submit(lambda: "", None, msg)
    
demo.launch()