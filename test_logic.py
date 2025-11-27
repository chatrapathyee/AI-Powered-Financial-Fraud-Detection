from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
import os
import time

def test_fraud_detection():
    print("Loading resources...")
    # Load embeddings
    hg_embeddings = HuggingFaceEmbeddings()
    # Load Chroma
    persist_directory = 'docs/chroma_rag/'
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=hg_embeddings, collection_name="finance_data_new")
    
    # Load LLM
    llm = CTransformers(
        model="TheBloke/Zephyr-7B-Beta-GGUF",
        model_file="zephyr-7b-beta.Q4_K_M.gguf",
        model_type="mistral",
        config={'max_new_tokens': 512, 'temperature': 0.1, 'context_length': 2048}
    )
    
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
    
    fraud_statement = "The company reported inflated revenues by including sales that never occurred."
    print(f"\nTesting Fraud Statement: {fraud_statement}")
    result_fraud = qa_chain.invoke({"query": fraud_statement})
    print(f"Result: {result_fraud['result']}")
    
    non_fraud_statement = "Financial records accurately reflect all expenses and liabilities."
    print(f"\nTesting Non-Fraud Statement: {non_fraud_statement}")
    result_non_fraud = qa_chain.invoke({"query": non_fraud_statement})
    print(f"Result: {result_non_fraud['result']}")

if __name__ == "__main__":
    test_fraud_detection()
