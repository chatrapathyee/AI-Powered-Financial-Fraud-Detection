import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import os


st.set_page_config(page_title="Fraud Detection AI", page_icon="�️", layout="wide")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0E1117;
        background-image: radial-gradient(circle at 50% 0%, #1E293B 0%, #0E1117 70%);
    }
    
    .header-container {
        text-align: center;
        padding: 4rem 1rem 2rem;
        animation: fadeIn 1.5s ease-in-out;
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        color: #94A3B8;
        font-weight: 300;
    }
    
    .stTextArea textarea {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2) !important;
    }
    
    .stButton button {
        background: linear-gradient(90deg, #4F46E5, #06B6D4) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px rgba(79, 70, 229, 0.5);
    }
    
    .result-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        animation: slideUp 0.5s ease-out;
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #F8FAFC;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <div class="header-title">Financial Guardian AI</div>
    <div class="header-subtitle">Advanced Fraud Detection powered by RAG & LLM</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("� Analysis Dashboard")
    user_input = st.text_area("Financial Statement Input", height=100, placeholder="Paste financial text here for real-time fraud analysis...", label_visibility="collapsed")
    
    analyze_button = st.button("Analyze Statement")
@st.cache_resource
def load_llm():
    llm = CTransformers(
        model="TheBloke/Zephyr-7B-Beta-GGUF",
        model_file="zephyr-7b-beta.Q4_K_M.gguf",
        model_type="mistral",
        config={'max_new_tokens': 512, 'temperature': 0.1, 'context_length': 2048}
    )
    return llm

@st.cache_resource
def load_vector_store():
    hg_embeddings = HuggingFaceEmbeddings()
    persist_directory = 'docs/chroma_rag/'
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=hg_embeddings, collection_name="finance_data_new")
    return vectordb

if analyze_button and user_input:
    with col2:
        with st.spinner("🤖 Analyzing patterns and cross-referencing data..."):
            try:
             
                llm = load_llm()
                vectordb = load_vector_store()
                
              
                retriever = vectordb.as_retriever(search_kwargs={"k": 2})
                
                
                template = """
                You are a Fraud Detection Expert in Financial Text Data. Analyze the following statement and predict if it indicates fraud or not.
                If you don't know the answer, just say "Sorry, I Don't Know."
                
                Context: {context}
                
                Question: {question}
                
                Answer:
                """
                PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
                
               
                qa_chain = RetrievalQA.from_chain_type(
                    llm,
                    retriever=retriever,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
              
                result = qa_chain.invoke({"query": user_input})
              
                st.markdown(f"""
                <div class="result-card">
                    <div class="result-title">📊 Analysis Report</div>
                    <div style="color: #CBD5E1; line-height: 1.6;">
                        {result['result']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {e}")

elif analyze_button and not user_input:
    with col2:
        st.warning("⚠️ Please provide a financial statement to analyze.")
