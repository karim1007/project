"""
Example usage of the RAG Manager class
"""
import os
from rag_manager import RAGManager

def main():
    # Initialize the RAG Manager
    # Using default Ollama parameters - assumes Ollama is running locally
    rag = RAGManager(
        ollama_model_name="llama3.1",  # Change to your preferred model
        embedding_model_name="mxbai-embed-large"  # Change based on available models in your Ollama instance
    )
    
    # Example 1: Load documents from a URL
    print("Example 1: Loading documents from a URL...")
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    documents = rag.load_from_url(url)
    print(f"Loaded and processed {len(documents)} document chunks from URL")
    
    # Create the vector store
    vectorstore = rag.create_vectorstore(documents)
    print("Vector store created successfully")
    
    # Set up the QA chain
    qa_chain = rag.setup_qa_chain()
    print("QA chain set up successfully")
    
    # Ask some questions
    questions = [
        "What is artificial intelligence?",
        "What are the main applications of AI?",
        "What are the ethical concerns around AI?"
    ]
    
    for question in questions:
        print(f"\nQuestion: {question}")
        result = rag.query(question)
        print(f"Answer: {result['result']}")
        print("Sources:")
        for i, doc in enumerate(result['source_documents'][:2]):  # Show only first 2 sources
            print(f"Source {i+1}: {doc.page_content[:100]}...")
    
    # Example 2: Save and load the vector store
    save_dir = "vectorstore"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nSaving vector store...")
    rag.save_vectorstore(save_dir)
    print(f"Vector store saved to {save_dir}")
    
    # Create a new instance and load the saved vector store
    print("\nLoading vector store from disk...")
    new_rag = RAGManager()
    new_rag.load_vectorstore(save_dir)
    print("Vector store loaded successfully")
    
    # Set up the QA chain for the new instance
    new_rag.setup_qa_chain()
    
    # Test querying with the loaded vector store
    test_question = "How does machine learning relate to AI?"
    print(f"\nQuestion: {test_question}")
    result = new_rag.query(test_question)
    print(f"Answer: {result['result']}")


if __name__ == "__main__":
    main()