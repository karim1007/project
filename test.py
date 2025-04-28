from contextlib import asynccontextmanager
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_together import ChatTogether
from dotenv import load_dotenv
from whatsapp import main as whatsapp_main
# Define the model
load_dotenv()
model=ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    api_key=os.getenv("TOGETHER_API_KEY")
)

@asynccontextmanager
async def make_graph():
    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Configure MCP clients for different services
    async with MultiServerMCPClient(
        {
            "whatsapp":  {
                    "command": "C:\\Users\\PC\\AppData\\Local\\Programs\\Python\\Python310\\Scripts\\uv.exe",
                    "args": ["--directory", "C:\\Users\\PC\\Desktop\\test\\whatsapp-mcp\\whatsapp-mcp-server", "run", "main.py"],
                    "transport": "stdio",
                },
            "google_calendar": {
                    "command": "node",
                    "args": [
                        "C:\\Users\\PC\\Desktop\\server\\google-calendar-mcp\\build\\index.js"
                    ],
                    "env": {
                        "GOOGLE_CLIENT_ID": os.getenv("GOOGLE_CLIENT_ID"),
                        "GOOGLE_CLIENT_SECRET": os.getenv("GOOGLE_CLIENT_SECRET"),
                        "GOOGLE_REDIRECT_URI": os.getenv("GOOGLE_REDIRECT_URI"),
                        "GOOGLE_REFRESH_TOKEN": os.getenv("GOOGLE_REFRESH_TOKEN")
                    },
                    "transport": "stdio",
                }
        }
    ) as mcp_client:
        # Get MCP tools
        mcp_tools = mcp_client.get_tools()
        
        # Create agents with both custom and MCP tools
        calendar_agent = create_react_agent(
            model=model,
            tools=mcp_tools,
            name="calendar_agent",
            prompt=f"You are a calendar agent responsible for managing calendar events and scheduling. Today's date is {today}. You have access to tools that can create, modify, and view calendar events. Always use one tool at a time and only when necessary. IMPORTANT: Report back to the supervisor with a short, concise status update about your task completion or findings. Do not address the user directly."
        )

        mail_agent = create_react_agent(
            model=model,
            tools= mcp_tools,
            name="whatsapp_agent",
            prompt=f"You are a WhatsApp assistant. Today's date is {today}. You have access to tools that can send or read whatsapp messages. Always use one tool at a time and only when necessary. IMPORTANT: Report back to the supervisor with a short, concise status update about your task completion or findings. Do not address the user directly."
        )

        # Create supervisor workflow
        workflow = create_supervisor(
            [calendar_agent, mail_agent],
            model=model,
            output_mode="full_history",  # or it can be "last_message" for last message only
            prompt=(
                "You are a personal assistant that helps manage whatsapp and calendar events. "
                "You are in charge of the team and responsible for chatting directly with the user. "
                "For scheduling and managing calendar events, delegate to calendar_agent. "
                "For composing and managing whatsapp, delegate to whatsapp_agent. "
                "The subagents only answer to you, and you are the one delivering the final message that the user sees. "
                "Ensure your responses are helpful, clear, and maintain a consistent voice for the user experience."
            )
        )
        
        # Compile and yield the workflow
        app = workflow.compile()
        
        yield app

async def main():
    # Example of how to use the supervised workflow
    async with make_graph() as app:
        # Send a message to the supervisor agent
        response = await app.ainvoke({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a personal assistant that helps manage whatsapp and calendar events."
                },
                {
                    "role": "user",
                    "content": "send a message to +201281140930 saying hello world , then send an book with karim.mo107@outlook.com tomorrow at 10 for 1 hour"
                }
            ]
        })
        print(response)
if __name__ == "__main__":
    # Run the graph
    import asyncio
    asyncio.run(main())