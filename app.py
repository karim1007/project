from langchain.prompts import ChatPromptTemplate
from langchain_together import ChatTogether
from dotenv import load_dotenv
import os
from langchain.chains import LLMChain
from whatsapp import main as whatsapp_main
from calendar_1 import main as calendar_main
import asyncio
from learning_planner import main as learning_planner_main
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langsmith import traceable
load_dotenv()
model = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    together_api_key=os.getenv("TOGETHER_API_KEY"),
)
class agent:
    def __init__(self, name):
        self.name = name
        self.llm = ChatGroq(temperature=0.1, groq_api_key=os.getenv("GROQ"), model_name="llama-3.3-70b-versatile")

    def summarize_messages(self, messages):
        if len(messages) > 25:
            # Create a summary prompt
            summary_prompt = "Summarize the following conversation while keeping important details and context:\n\n"
            for msg in messages[:-10]:  # Summarize all except last 10 messages
                summary_prompt += f"{msg['role']}: {msg['content']}\n"
            
            # Get summary from the model
            summary = self.llm.invoke(summary_prompt)
            
            # Keep the last 10 messages and add summary at the beginning
            summarized_messages = [{"role": "system", "content": f"Previous conversation summary: {summary.content}"}]
            summarized_messages.extend(messages[-10:])
            return summarized_messages
        return messages

    async def ainvoke(self, messages):
        # Summarize messages if needed
        messages = self.summarize_messages(messages)
        
        if self.name == "whatsapp":
            return await whatsapp_main(messages, self.llm)
        elif self.name == "calendar":
            return await calendar_main(messages, self.llm)
        return {"response": f"Agent {self.name} invoked with messages: {messages}"}

    def invoke(self, messages):
        # Summarize messages if needed
        messages = self.summarize_messages(messages)
        
        if self.name == "learning_planner":
            return learning_planner_main(messages, self.llm)
        return {"response": f"Agent {self.name} invoked with messages: {messages}"}

messages=[]
whatsapp_agent = agent("whatsapp")
calendar_agent = agent("calendar")
learning_planner_agent = agent("learning_planner")

def general_inquiry(user_message: str) -> str:
    """
    General inquiry function to handle user messages.
    
    Args:
        user_message: The message from the user
    
    Returns:
        A string indicating the response to the user's message
    """
    # Here you can implement any general inquiry logic if needed
    messages.append({"role": "user", "content": user_message})
    a=agent("general_inquiry")
    response= model.invoke("you are a friendly chatbot that assists user in whatsapp ,calendar , and a learning planner the user will send a message your job is to reply in a friendly way and ask if he needs assistance  user message: " +user_message)
    

           
    messages.append({"role": "assistant", "content": response.content})
    return messages



def detect_intent(user_message: str, first=False) -> str:
    """
    Detect the intent of the user's message using a language model.
    
    Args:
        user_message: The message from the user
        first: Whether this is the first message in the conversation
    
    Returns:
        A string indicating the detected intent (1, 2, 3, or 4)
    """
    # Step 1: Load environment variables
    load_dotenv()
    global model, messages
    
    # Convert message history to a readable format
    conversation_history = "\n".join([
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in messages[-5:] # Include last 5 messages for context
    ])
    
    # Step 2: Initialize the LLM with conversation context
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant that classifies user requests into one of three intents: (if the user gives contact information keep it in mind but do not put it into consideration when classifying the intent)

    1. Use WhatsApp
    2. Use Calendar
    3. Generate a Learning Path
    4. General Inquiry

    Recent conversation history:
    {conversation_history}

    Current user message: {user_message}

    Given the conversation context and current message, reply ONLY with the number (1, 2, 3 or 4) that matches the intent.
    """)

    chain = prompt | model
    
    # Step 4: Predict
    result = chain.invoke({
        "user_message": user_message,
        "conversation_history": conversation_history if not first else "No previous context"
    })
    
    print(result)
    response = result.content
    print(response)
    return response

@traceable
async def chat(user_message: str) -> str:
    """
    Main function to run the intent detection.
    """
    global messages
   
    messages.append({"role": "user", "content": user_message})
    intent = detect_intent(user_message)
    print(f"Detected Intent: {intent}")
    if intent == "1":
        print(messages)
        await whatsapp_agent.ainvoke(messages)
        response = messages[-1]["content"]
        return response
    elif intent == "2":
        await calendar_agent.ainvoke(messages)
        response = messages[-1]["content"]
        return response
    elif intent == "3":
        learning_planner_agent.invoke(messages)
        response = messages[-1]["content"]
        return response
    else:
        general_inquiry(user_message)
        response = messages[-1]["content"]
        return response


async def interactive_chat():
    """
    Interactive chat loop that continuously accepts user input
    """
    print("Welcome to the Interactive Chat System!")
    print("You can interact with WhatsApp, Calendar, and Learning Planner.")
    print("Type 'exit' or 'quit' to end the conversation.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        try:
            response = await chat(user_input)
            print(f"Assistant: {response}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(interactive_chat())
