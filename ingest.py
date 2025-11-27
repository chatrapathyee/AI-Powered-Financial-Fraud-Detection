import pandas as pd
import random
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

def clean_text(text):
    text = text.encode('ascii', 'ignore').decode()
    
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    text = text.lower()  
 
    tokens = word_tokenize(text)
  
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    cleaned_text = ' '.join(tokens)
    
    return cleaned_text

def main():
    print("Generating data...")
    fraud_statements = [
        "The company reported inflated revenues by including sales that never occurred.",
        "Financial records were manipulated to hide the true state of expenses.",
        "The company failed to report significant liabilities on its balance sheet.",
        "Revenue was recognized prematurely before the actual sales occurred.",
        "The financial statement shows significant discrepancies in inventory records.",
        "The company used off-balance-sheet entities to hide debt.",
        "Expenses were understated by capitalizing them as assets.",
        "There were unauthorized transactions recorded in the financial books.",
        "Significant amounts of revenue were recognized without proper documentation.",
        "The company falsified financial documents to secure a larger loan.",
        "There were multiple instances of duplicate payments recorded as expenses.",
        "The company reported non-existent assets to enhance its financial position.",
        "Expenses were fraudulently categorized as business development costs.",
        "The company manipulated financial ratios to meet loan covenants.",
        "Significant related-party transactions were not disclosed.",
        "The financial statement shows fabricated sales transactions.",
        "There was intentional misstatement of cash flow records.",
        "The company inflated the value of its assets to attract investors.",
        "Revenue from future periods was reported in the current period.",
        "The company engaged in channel stuffing to inflate sales figures."
    ]

    non_fraud_statements = [
        "The company reported stable revenues consistent with historical trends.",
        "Financial records accurately reflect all expenses and liabilities.",
        "The balance sheet provides a true and fair view of the company’s financial position.",
        "Revenue was recognized in accordance with standard accounting practices.",
        "The inventory records are accurate and match physical counts.",
        "The company’s debt is fully disclosed on the balance sheet.",
        "All expenses are properly categorized and recorded.",
        "Transactions recorded in the financial books are authorized and documented.",
        "Revenue recognition is supported by proper documentation.",
        "Financial documents were audited and found to be accurate.",
        "Payments and expenses are recorded accurately without discrepancies.",
        "The assets reported on the balance sheet are verified and exist.",
        "Business development costs are properly recorded as expenses.",
        "Financial ratios are calculated based on accurate data.",
        "All related-party transactions are fully disclosed.",
        "Sales transactions are accurately recorded in the financial statement.",
        "Cash flow records are accurate and reflect actual cash movements.",
        "The value of assets is fairly reported in the financial statements.",
        "Revenue is reported in the correct accounting periods.",
        "Sales figures are accurately reported without manipulation."
    ]

  
    fraud_data = [{"text": statement, "fraud_status": "fraud"} for statement in fraud_statements]
    non_fraud_data = [{"text": random.choice(non_fraud_statements), "fraud_status": "non-fraud"} for _ in range(60)]

    data = fraud_data + non_fraud_data
    random.shuffle(data) 

    df = pd.DataFrame(data)
    
    print("Cleaning text...")
 
    df['Clean_Text'] = df['text'].apply(clean_text)

    documents = []
    for i, row in df.iterrows():
        content = f"id:{i}\nFillings: {row['Clean_Text']}\nFraud_Status: {row['fraud_status']}"
        documents.append(Document(page_content=content))

    print("Creating embeddings and vector store...")
    hg_embeddings = HuggingFaceEmbeddings()
    
    persist_directory = 'docs/chroma_rag/'
    
    langchain_chroma = Chroma.from_documents(
        documents=documents,
        collection_name="finance_data_new",
        embedding=hg_embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Vector store created at {persist_directory}")

if __name__ == "__main__":
    main()
