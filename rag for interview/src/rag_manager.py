"""
RAG Manager Class - Handles document processing and RAG implementation
using LangChain with FAISS vector store and Ollama for embeddings and retrieval
"""
import os
from typing import List, Union, Optional, Dict, Any
from urllib.parse import urlparse

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class RAGManager:
    """
    RAG Manager Class - Responsible for loading, processing and indexing documents
    for retrieval-augmented generation using LangChain.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        ollama_model_name: str = "llama3.1",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model_name: str = "mxbai-embed-large"
    ):
        """
        Initialize the RAG Manager.
        
        Args:
            chunk_size: Size of document chunks for splitting
            chunk_overlap: Overlap between document chunks
            ollama_model_name: Name of the Ollama model for generation
            ollama_base_url: Base URL for Ollama API
            embedding_model_name: Name of the embedding model to use
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ollama_model_name = ollama_model_name
        self.ollama_base_url = ollama_base_url
        self.embedding_model_name = embedding_model_name
        
        # Initialize text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.embedding_model_name,
            base_url=self.ollama_base_url
        )
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=self.ollama_model_name,
            base_url=self.ollama_base_url
        )
        
        self.vectorstore = None
        self.qa_chain = None

    def load_from_url(self, url: str) -> List:
        """
        Load content from a URL.
        
        Args:
            url: URL to load content from
            
        Returns:
            List of processed documents
        """
        loader = WebBaseLoader(url)
        documents = loader.load()
        return self._process_documents(documents)
    
    def load_from_file(self, file_path: str) -> List:
        """
        Load content from a file based on its extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of processed documents
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            loader = PyPDFLoader(file_path)
        else:
            # Default to text loader for txt and other formats
            loader = TextLoader(file_path)
            
        documents = loader.load()
        return self._process_documents(documents)
    
    def load_from_directory(self, directory_path: str, glob_pattern: str = "**/*.*") -> List:
        """
        Load documents from a directory.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Pattern for file matching
            
        Returns:
            List of processed documents
        """
        loader = DirectoryLoader(
            directory_path, 
            glob=glob_pattern,
            use_multithreading=True
        )
        documents = loader.load()
        return self._process_documents(documents)
    
    def load_documents(self, source: Union[str, List]) -> List:
        """
        Main method to load documents from various sources.
        
        Args:
            source: URL, file path, directory, or list of documents
            
        Returns:
            List of processed documents
        """
        if isinstance(source, list):
            # Assume these are already Document objects
            return self._process_documents(source)
        
        if isinstance(source, str):
            # Check if it's a URL
            parsed_url = urlparse(source)
            if parsed_url.scheme and parsed_url.netloc:
                return self.load_from_url(source)
            
            # Check if it's a directory
            elif os.path.isdir(source):
                return self.load_from_directory(source)
            
            # Otherwise assume it's a file
            elif os.path.isfile(source):
                return self.load_from_file(source)
        
        raise ValueError(f"Unsupported source type: {source}")
    
    def _process_documents(self, documents: List) -> List:
        """
        Process documents by splitting them into chunks.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed document chunks
        """
        return self.text_splitter.split_documents(documents)
    
    def create_vectorstore(self, documents: List) -> FAISS:
        """
        Create a vector store from processed documents.
        
        Args:
            documents: List of processed documents
            
        Returns:
            FAISS vector store
        """
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        return self.vectorstore
    
    def save_vectorstore(self, directory_path: str):
        """
        Save the vector store to disk.
        
        Args:
            directory_path: Directory to save the vector store in
        """
        if self.vectorstore:
            self.vectorstore.save_local(directory_path)
        else:
            raise ValueError("No vectorstore available to save")
    
    def load_vectorstore(self, directory_path: str):
        """
        Load a vector store from disk.
        
        Args:
            directory_path: Directory to load the vector store from
        """
        self.vectorstore = FAISS.load_local(directory_path, self.embeddings, allow_dangerous_deserialization=True)
        return self.vectorstore
    
    def setup_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Set up the retriever from the vector store.
        
        Args:
            search_kwargs: Search arguments for the retriever
        """
        if not self.vectorstore:
            raise ValueError("Vector store must be created or loaded before setting up retriever")
            
        search_kwargs = search_kwargs or {"k": 4}
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    def setup_qa_chain(self, template: Optional[str] = None):
        """
        Set up the QA chain for answering questions.
        
        Args:
            template: Optional custom prompt template
        """
        if not self.vectorstore:
            raise ValueError("Vector store must be created or loaded before setting up QA chain")
        
        retriever = self.setup_retriever()
        
        # Set up default RAG prompt if not provided
        if not template:
            template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Do not include sources
            
            {context}
            
            Question: {question}
            Answer: """
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return self.qa_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask the RAG system
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            self.setup_qa_chain()
            
        return self.qa_chain.invoke({"query": question})