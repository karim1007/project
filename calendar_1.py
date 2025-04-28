import logging
import os
from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_together import ChatTogether
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from datetime import datetime, timedelta
# Load environment variables
load_dotenv()
current_time = (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")

prompt=f"""You are a Google Calendar assistant. You can manage calendars, events, and schedules. Use your tools to interact with the user's Google Calendar. the current time is {current_time}
return the meeting link in the response even if the user does not ask for it (). 
assume that the meeting will be 1 hour long unless the user specifies otherwise.
if the user does not specify a time for the event assume that it will be in 1 hour from now."""
# Set up logging


log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, 'calendar_agent.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize LLM
model = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    api_key=os.getenv("TOGETHER_API_KEY")
)


@asynccontextmanager
async def create_calendar_graph(llm):
    """
    Creates and initializes the Google Calendar agent with MCP tools.
    """
    logger.info("Initializing Google Calendar MCP client")
    try:
        async with MultiServerMCPClient(
            {
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
        ) as client:
            logger.info("MCP client connection established")
            # Create the agent with MCP tools
            agent = create_react_agent(llm, client.get_tools(), name="google_calendar")
            logger.info("React agent created successfully")
            yield agent
    except Exception as e:
        logger.error(f"Error in Google Calendar graph creation: {str(e)}", exc_info=True)
        raise

async def main(messages,llm):
    """
    Demonstrates the functionality of the Google Calendar agent.
    """
    logger.info("Starting Google Calendar MCP agent")
    try:
        async with create_calendar_graph(llm) as agent:
            logger.info("Sending test message")
            response = await agent.ainvoke(
                {"messages": [{"role": "system", "content": prompt}]+messages}
               
            )
            logger.info(f"Message sent successfully. Response: {response}")
            last_ai_message = None
            for message in reversed(response["messages"]):  # go backwards
                if isinstance(message, AIMessage):
                    last_ai_message = message
                    break

           
            messages.append({"role": "assistant", "content": last_ai_message.content})
            return messages
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import asyncio
    try:
        messag= [{"role": "user", "content": "create an event that will start in 1 hour do not invite anyone, it will be aboutthe weather"},]
        asyncio.run(main(messag, model))
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)