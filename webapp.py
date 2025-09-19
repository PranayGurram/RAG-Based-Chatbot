import os
import google.generativeai as genai
from langchain.vectorstores import FAISS  # Vector DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import faiss
import streamlit as st
from pdfextractor import text_extractor_pdf

# App title
st.set_page_config(page_title="AI RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 :blue[AI-Powered RAG Chatbot]")

tips = """
### 📌 How to use:
1. 📄 Upload a PDF from the sidebar  
2. 💬 Ask a question in the chat box  
3. 🤖 Get AI-powered responses with context from your document  
"""
st.markdown(tips)

# Sidebar file upload
st.sidebar.title("📂 Upload PDF")
file_uploader = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

if file_uploader:
    # Extract text
    file_text = text_extractor_pdf(file_uploader)

    # Configure LLM
    key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=key)
    llm_model = genai.GenerativeModel("gemini-2.5-flash-lite")

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_text(file_text)

    # Create FAISS Vector Store
    vector_store = FAISS.from_texts(chunks, embedding_model)

    # Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Function to generate response
    def generate_response(query):
        relevant_docs = retriever.get_relevant_documents(query=query)
        context = " ".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        You are an AI assistant using RAG.
        Context: {context}
        User query: {query}
        """
        content = llm_model.generate_content(prompt)
        return content.text

    # Initialize session state
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(
                f"""
                <div style='text-align: right; background-color:#DCF8C6;
                            padding:10px; border-radius:12px; margin:5px;'>
                    👤 <b>You:</b> {msg['text']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style='text-align: left; background-color:#E6E6FA;
                            padding:10px; border-radius:12px; margin:5px;'>
                    🤖 <b>Chatbot:</b> {msg['text']}
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("💬 Enter your query:")
        send = st.form_submit_button("Send")

    if user_input and send:
        # Add user input
        st.session_state.history.append({"role": "user", "text": user_input})

        # Get bot response
        model_output = generate_response(user_input)
        st.session_state.history.append({"role": "bot", "text": model_output})

        st.rerun()
