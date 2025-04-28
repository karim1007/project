from langsmith import evaluate, Client
from difflib import SequenceMatcher
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

client = Client()
dataset_name = "rag data"

# Initialize Ollama embeddings
embedding_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

# Initialize Ollama LLM
llm = OllamaLLM(
    model="llama3.1",
    base_url="http://localhost:11434"
)

# Path to your vectorstore
vectorstore_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vectorstore")

def retrieve_documents(question):
    """
    Retrieve relevant documents from the FAISS index based on the question
    
    Args:
        question: The query to search for in the vector store
        
    Returns:
        List of documents relevant to the question
    """
    # Load the existing vector store
    vectorstore = FAISS.load_local(vectorstore_path, embedding_model)
    
    # Retrieve documents from the vector store
    docs = vectorstore.similarity_search(question, k=4)
    
    return docs

def generate_response(question, documents):
    """
    Generate a response to the question using the retrieved documents
    
    Args:
        question: The query to answer
        documents: The retrieved documents to use as context
        
    Returns:
        Generated answer to the question
    """
    # Create a context string from the documents
    context = "\n\n".join([doc.page_content for doc in documents])
    
    # Create a prompt template
    prompt_template = """
    You are a helpful assistant. Use the following context to answer the question.
    If you don't know the answer based on the context, just say you don't know.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create an LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain
    response = chain.invoke({"context": context, "question": question})
    
    return response["text"]

def langsmith_rag(question):
    """
    RAG pipeline function that retrieves documents and generates a response
    
    Args:
        question: The query to answer
        
    Returns:
        Dictionary with the generated output
    """
    # Retrieve relevant documents
    documents = retrieve_documents(question)
    
    # Generate response
    answer = generate_response(question, documents)
    
    # Return in the format expected by LangSmith
    return {"output": answer}

def similarity_score(reference_outputs: dict, outputs: dict) -> dict:
    reference = reference_outputs["output"]
    prediction = outputs["output"]
    similarity = SequenceMatcher(None, reference, prediction).ratio()
    return {"key": "similarity", "score": similarity}

def target_function(inputs: dict):
    return langsmith_rag(inputs["question"])

# Only run evaluation if this file is executed directly
if __name__ == "__main__":
    evaluate(
        target_function,
        data=dataset_name,
        evaluators=[similarity_score],
        experiment_prefix="rag data"
    )
