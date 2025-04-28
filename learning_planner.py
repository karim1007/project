import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_together import ChatTogether
from langchain_core.messages import AIMessage

# Load environment variables
load_dotenv()
prompt = """You are a helpful AI assistant that creates structured learning plans.
    Given a user message that may contain informal or natural language, do the following:

    Identify the topic the user wants to learn (e.g., from "I want to learn Python programming", extract "Python programming").

    Create a detailed learning roadmap for that topic, assuming the user is a beginner.

    The plan should include:

    A brief overview of the topic

    Prerequisites (if any)

    A structured weekly plan or logical learning stages

    Recommended resources (courses, books, tutorials)

    Practice projects or exercises

    Estimated time needed per stage/week

    Be clear, practical, and beginner-friendly.

    Format your response as:
    plan:
    [numbered steps with time estimates]

    Resources:
    [relevant links]

    Total Estimated Time: [sum of all steps]

    Remember to be realistic with time estimates and provide a clear progression from basics to advanced topics.
    ignore contacts of the user and any other information that is not related to the learning plan."""
# Define the input/output structure
def create_learning_planner_graph(llm):
# Create the Tavily search tool with API key from environment
    search = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

    @tool
    def search_topics(query: str) -> str:
        """Search for educational resources and information about a topic"""
        results = search.invoke(query + " tutorial learning resources guide")
        return str(results)

    # Define the tools
    tools = [search_topics]

    # Create the agent with specific prompt

    # Initialize the ChatTogether model
    model = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        api_key=os.getenv("TOGETHER_API_KEY")
    )

    # Create the REACT agent
    agent = create_react_agent(
        model,
        tools=tools,
        name="learning_planner",
    )

    return agent
    # Example usage:

def main(messages,llm):
    """
    Demonstrates the functionality of the Learning Planner agent.
    """
    try:
        # Create the learning planner graph
        agent = create_learning_planner_graph(llm)

        # Send a test message to the agent
        response = agent.invoke( {"messages": [{"role": "system", "content": prompt }]+messages})

        # Process the response and return it
        last_ai_message = None
        for message in reversed(response["messages"]):  # go backwards
            if isinstance(message, AIMessage):
                last_ai_message = message
                break

        messages.append({"role": "assistant", "content": last_ai_message.content})
        return messages

    except Exception as e:
        print(f"Error in main execution: {str(e)}")


if __name__ == "__main__":
    import asyncio
    try:
        # Example message to test the agent
        messages = [{"role": "user", "content": "generate a learning path for a beginner in python programming"}]
        main(messages)
    except Exception as e:
        print(f"Error: {str(e)}")