import os
import pandas as pd
import re
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# --- Configuration ---
# Ensure your CSV file matches this name
CSV_PATH = "data/IMDB Dataset.csv" 
DB_PATH = "./chroma_sentiment_db"
EMBEDDING_MODEL = "mxbai-embed-large"

def clean_text(text):
    """Removes HTML tags and excessive whitespace."""
    if not isinstance(text, str): return ""
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print(f"Loading data from {CSV_PATH}...")
    try:
        # We only expect columns: 'review' and 'sentiment'
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print("Error: CSV file not found. Please check the filename.")
        return

    # 1. Clean Data
    print("Cleaning text...")
    # Drop rows where review is missing
    df = df.dropna(subset=["review"]) 
    df["clean_review"] = df["review"].apply(clean_text)

    # 2. Prepare Documents
    print("Preparing documents...")
    documents = []
    ids = []
    
    # We use 'enumerate' to create a fake ID (0, 1, 2...) since the dataset doesn't have one
    for i, row in df.iterrows():
        
        # The content to embed is just the review text
        content = row['clean_review']
        
        # We store sentiment in metadata so we can filter later
        # e.g., retriever.invoke("...", filter={"sentiment": "positive"})
        meta = {
            "sentiment": str(row["sentiment"])
        }
        
        doc = Document(
            page_content=content,
            metadata=meta,
            id=str(i) # Using the index as the ID
        )
        documents.append(doc)
        ids.append(str(i))

    # 3. Initialize Vector Store
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    if os.path.exists(DB_PATH):
        print(f"Database found at {DB_PATH}. Appending/Updating data...")
    else:
        print(f"Creating new database at {DB_PATH}...")

    vector_store = Chroma(
        collection_name="sentiment_reviews",
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    # 4. Ingest
    print(f"Adding {len(documents)} documents to ChromaDB...")
    vector_store.add_documents(documents=documents, ids=ids)
    
    print("Success! Data ingested.")

if __name__ == "__main__":
    main()