
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# --- Configuration ---
# Must match the path used in your new ingest.py
DB_PATH = "./chroma_sentiment_db" 
EMBEDDING_MODEL = "mxbai-embed-large"
CHAT_MODEL = "llama3"

def format_docs(docs):
    """Joins document content into a single string for the LLM."""
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    # 1. Load the Database
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name="sentiment_reviews",
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    # 2. Create the Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. Initialize the Chat Model
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.3)

    # 4. Create the Prompt Template
    template = """You are a sentiment analysis assistant. 
    Use the following customer reviews to answer the user's question.
    
    Context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 5. Build the Chain with Sources
    # This structure retrieves docs first, then passes them to TWO places:
    #   a. The 'answer' branch (which formats them -> prompt -> llm)
    #   b. The final output (so we can see them)
    rag_chain_with_source = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        .assign(answer=(
            RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
            | prompt
            | llm
            | StrOutputParser()
        ))
    )

    # 6. Interactive Loop
    print(f"Agent ready! Connected to {CHAT_MODEL} and sentiment database.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask about reviews: ")
        if query.lower() in ["exit", "quit", "q"]:
            break
            
        print("Thinking...")
        
        # Invoke the chain
        result = rag_chain_with_source.invoke(query)
        
        print("\n--- Answer ---")
        print(result["answer"])
        
        print("\n[Sources Used]")
        for i, doc in enumerate(result["context"]):
            # We access the 'sentiment' metadata we saved in ingest.py
            sentiment = doc.metadata.get("sentiment", "Unknown")
            # Truncate content if it's too long for display
            preview = doc.page_content[:100].replace("\n", " ") + "..."
            print(f"{i+1}. [{sentiment.upper()}] {preview}")
        print("-" * 30 + "\n")

if __name__ == "__main__":
    main()