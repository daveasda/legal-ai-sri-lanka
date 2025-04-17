import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from langchain.memory import ConversationBufferMemory
import gradio as gr

# Load documents
def load_documents():
    loader = DirectoryLoader("docs", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Create vector store
def create_vector_store(documents):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

# Load or create vectorstore
if os.path.exists("faiss_index"):
    vectorstore = FAISS.load_local("faiss_index", SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"), allow_dangerous_deserialization=True)
else:
    print("Index not found. Creating new index...")
    docs = load_documents()
    vectorstore = create_vector_store(docs)
    vectorstore.save_local("faiss_index")

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Setup memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define the function to generate response using GPT-2
def generate_response(query, chat_history=[]):
    # Concatenate chat history and query to form input
    input_text = "\n".join([f"User: {q[0]}\nBot: {q[1]}" for q in chat_history]) + f"\nUser: {query}\nBot:"
    
    # Tokenize the input
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)
    
    # Generate output from GPT-2
    output = model.generate(inputs, max_length=1024, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract the response (the part after the last "Bot:" token)
    response = output_text.split("Bot:")[-1].strip()
    
    # Append the query and response to the chat history
    chat_history.append((query, response))
    
    return chat_history, chat_history

# Gradio UI
def chat_interface(query, chat_history=[]):
    return generate_response(query, chat_history)

with gr.Blocks() as demo:
    gr.Markdown("# 🇱🇰 Legal AI Sri Lanka")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask me anything about Sri Lankan law...")
    state = gr.State([])

    msg.submit(chat_interface, [msg, state], [chatbot, state])

demo.launch()
