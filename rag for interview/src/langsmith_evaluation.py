"""
Evaluating RAG system using LangSmith SDK

This script connects to a LangSmith dataset, runs inference using the RAGManager,
and evaluates the results using various evaluation metrics.
"""
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from langchain.smith import RunEvaluator
from langsmith import Client, RunEvalConfig
from langsmith.evaluation import StringEvaluator
from langsmith.schemas import Example, Run

from rag_manager import RAGManager

# Load environment variables
load_dotenv()

# Initialize LangSmith client
client = Client()

def setup_rag_system(vector_store_path: Optional[str] = None):
    """
    Set up the RAG system
    
    Args:
        vector_store_path: Path to an existing vector store, if available
        
    Returns:
        Initialized RAG Manager
    """
    # Initialize RAG Manager
    rag = RAGManager(
        ollama_model_name="llama2",
        embedding_model_name="nomic-embed-text"
    )
    
    # Load vector store if path is provided
    if vector_store_path and os.path.exists(vector_store_path):
        print(f"Loading vector store from {vector_store_path}")
        rag.load_vectorstore(vector_store_path)
        rag.setup_qa_chain()
    
    return rag

def evaluate_with_dataset(rag: RAGManager, dataset_name: str, num_examples: int = None):
    """
    Evaluate the RAG system using a LangSmith dataset
    
    Args:
        rag: Initialized RAG Manager
        dataset_name: Name of the LangSmith dataset to use
        num_examples: Optional limit on the number of examples to evaluate
    """
    print(f"Fetching dataset: {dataset_name}")
    
    # Get the dataset
    dataset = client.read_dataset(dataset_name=dataset_name)
    print(f"Found dataset: {dataset.name} (ID: {dataset.id})")
    
    # Get examples from the dataset
    examples = client.list_examples(dataset_id=dataset.id)
    example_list = list(examples)
    
    if num_examples:
        example_list = example_list[:num_examples]
    
    print(f"Running evaluation on {len(example_list)} examples")
    
    # Create a project for this evaluation run
    project_name = f"rag_evaluation_{dataset_name}_{os.urandom(4).hex()}"
    
    # Define evaluation metrics
    evaluators = [
        # Relevance evaluator
        StringEvaluator(
            evaluation_name="relevance",
            grading_function=relevance_grader,
        ),
        # Correctness evaluator
        StringEvaluator(
            evaluation_name="correctness",
            grading_function=correctness_grader,
        )
    ]
    
    # Run evaluation
    for i, example in enumerate(example_list):
        print(f"Processing example {i+1}/{len(example_list)}")
        
        # Extract question from the example
        question = example.inputs.get("question") or example.inputs.get("query")
        if not question:
            print("Skipping example: No question found in inputs")
            continue
        
        # Run the question through the RAG system
        try:
            result = rag.query(question)
            answer = result.get("result", "")
            
            # Log the run to LangSmith
            run = client.create_run(
                project_name=project_name,
                name=f"RAG Query {i+1}",
                inputs={"question": question},
                outputs={"answer": answer},
                reference_example_id=example.id
            )
            
            # Evaluate the run against each evaluator
            for evaluator in evaluators:
                client.evaluate_run(
                    run,
                    evaluator=evaluator,
                    reference_example=example
                )
                
        except Exception as e:
            print(f"Error processing example {i+1}: {str(e)}")
    
    print(f"Evaluation complete! View results in the LangSmith UI under project: {project_name}")
    print(f"Or use the results programmatically via the LangSmith client API")

def relevance_grader(run_input: Dict[str, Any], run_output: Dict[str, Any], 
                    reference_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evaluate the relevance of the RAG system's answer to the question
    
    Args:
        run_input: The input to the run (question)
        run_output: The output from the run (generated answer)
        reference_output: The reference output (ground truth)
    
    Returns:
        Dictionary with score and feedback
    """
    question = run_input.get("question", "")
    answer = run_output.get("answer", "")
    
    # Simple word overlap metric (this is a basic example - you might want to use more sophisticated metrics)
    question_words = set(question.lower().split())
    answer_words = set(answer.lower().split())
    
    # Calculate word overlap
    overlap = len(question_words.intersection(answer_words))
    total_words = len(question_words.union(answer_words))
    
    if total_words == 0:
        score = 0.0
    else:
        score = overlap / total_words
    
    # Assign category
    if score > 0.7:
        category = "HIGH_RELEVANCE"
    elif score > 0.4:
        category = "MEDIUM_RELEVANCE" 
    else:
        category = "LOW_RELEVANCE"
    
    return {
        "score": score,
        "value": category,
        "reasoning": f"The answer has a word overlap of {score:.2f} with the question."
    }

def correctness_grader(run_input: Dict[str, Any], run_output: Dict[str, Any], 
                      reference_output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evaluate the correctness of the RAG system's answer against a reference answer
    
    Args:
        run_input: The input to the run
        run_output: The output from the run
        reference_output: The reference output (ground truth)
    
    Returns:
        Dictionary with score and feedback
    """
    answer = run_output.get("answer", "")
    
    if not reference_output:
        return {
            "score": 0.5,
            "value": "AMBIGUOUS",
            "reasoning": "No reference answer provided for comparison."
        }
    
    # Get the reference answer
    reference_answer = reference_output.get("answer", "")
    
    if not reference_answer:
        return {
            "score": 0.5,
            "value": "AMBIGUOUS",
            "reasoning": "Reference answer is empty."
        }
    
    # Simple string comparison (this is a basic example - you might want more sophisticated metrics)
    # Calculate token overlap as a basic correctness measure
    reference_tokens = set(reference_answer.lower().split())
    answer_tokens = set(answer.lower().split())
    
    common_tokens = reference_tokens.intersection(answer_tokens)
    
    if len(reference_tokens) == 0:
        score = 0.0
    else:
        score = len(common_tokens) / len(reference_tokens)
    
    # Assign category
    if score > 0.8:
        category = "CORRECT"
    elif score > 0.5:
        category = "PARTIALLY_CORRECT"
    else:
        category = "INCORRECT"
    
    return {
        "score": score,
        "value": category,
        "reasoning": f"The answer matches {score:.2f} of the reference answer tokens."
    }

def main():
    # Parse command line arguments or use default values
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RAG system using LangSmith")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Name of the LangSmith dataset to use")
    parser.add_argument("--vectorstore", type=str, default=None,
                        help="Path to the vector store directory")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit the number of examples to evaluate")
    
    args = parser.parse_args()
    
    # Set up RAG system
    rag = setup_rag_system(args.vectorstore)
    
    # Run evaluation
    evaluate_with_dataset(rag, args.dataset, args.limit)

if __name__ == "__main__":
    main()