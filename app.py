import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- Page Config ---
st.set_page_config(page_title="PDF Chat AI", layout="centered")
st.title("📄 PDF Retrieval Augmented Generation [RAG]")
st.subheader("Talk to your documents in real-time")

# --- 1. Setup API Key ---
# You can also use st.secrets for production
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    st.info("Waiting for API Key from Streamlit Secrets...")
# --- 2. Sidebar for PDF Uploading ---
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# --- 3. The RAG Logic (Cached to avoid re-processing every click) ---
@st.cache_resource
def process_pdf(file_path):
    # Step 1: Loading
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Step 2: Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunk_documents = text_splitter.split_documents(docs)
    
    # Step 3: Embeddings & Vector Store
    db = FAISS.from_documents(chunk_documents, OpenAIEmbeddings())
    return db.as_retriever()

# --- 4. Building the Chain ---
if uploaded_file:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Initialize Retriever
    retriever = process_pdf("temp.pdf")

    # Define the Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the following question based only on the provided context.\n"
                   "Think step by step before providing an answer.\n\n"
                   "<context>\n{context}\n</context>"),
        ("human", "{input}")
    ])

    # Setup the LLM and Chains
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    retrieval_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=document_chain)

    # --- 5. The Chat Interface ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if user_query := st.chat_input("Ask a question about your PDF:"):
        # Add user message to state
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Searching document..."):
                response = retrieval_chain.invoke({"input": user_query})
                answer = response['answer']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload a PDF file from the sidebar to start chatting.")