"""
Example usage of the RAG Manager class with PDF files
"""
import os
from rag_manager import RAGManager

def main():
    # Initialize the RAG Manager
    # Using default Ollama parameters - assumes Ollama is running locally
    rag = RAGManager(
        ollama_model_name="llama3.1",
        embedding_model_name="mxbai-embed-large",
        chunk_size=500,  # Smaller chunk size for PDFs can sometimes work better
        chunk_overlap=50
    )
    
    # Path to your PDF file
    # Replace this with the actual path to your PDF file
    pdf_path = "C:\\Users\\PC\\Desktop\\rag for interview\\report.pdf"
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found.")
        return
    
    print(f"Processing PDF file: {pdf_path}")
    
    # Load and process the PDF file
    documents = rag.load_from_file(pdf_path)
    print(f"Loaded and processed {len(documents)} document chunks from PDF")
    
    # Create the vector store
    vectorstore = rag.create_vectorstore(documents)
    print("Vector store created successfully")
    
    # Save the vector store for future use
    save_dir = "pdf_vectorstore"
    os.makedirs(save_dir, exist_ok=True)
    rag.save_vectorstore(save_dir)
    print(f"Vector store saved to {save_dir}")
    
    # Set up the QA chain
    qa_chain = rag.setup_qa_chain()
    print("QA chain set up successfully")
    
    # Interactive question answering
    print("\nEnter your questions about the PDF document (type 'exit' to quit):")
    
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'exit':
            break
            
        result = rag.query(question)
        print(f"\nAnswer: {result['result']}")
        
        # Display sources (optional)
        print("\nSources:")
        for i, doc in enumerate(result['source_documents'][:2]):  # Show only first 2 sources
            print(f"Source {i+1}: {doc.page_content[:150]}...\n")

if __name__ == "__main__":
    main()