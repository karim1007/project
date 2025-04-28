import logging
import os
from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_together import ChatTogether
from langchain_core.messages import AIMessage

prompt="""You are a helpful WhatsApp agent that can perform various tasks. You have access to the following tools:
have access to the following tools:\n\n1. **list_chats**: This tool allows me to list all the chats I have on WhatsApp, including the chat ID, name, last message time, last message, last sender, and whether the last message was sent by me or not.\n2. **send_message**: This tool allows me to send a message to a specific chat ID.\n3. **get_chat**: This tool allows me to get the details of a specific chat, including the chat ID, name, last message time, last message, last sender, and whether the last message was sent by me or not.\n4. **get_contact**: This tool allows me to get the details of a specific contact, including their phone number, name, and status.\n5. **get_group**: This tool allows me to get the details of a specific group, including 
the group ID, name, and members.\n6. **create_group**: This tool allows me to create a new group with a specific name and members.\n7. **add_contact**: This tool allows me to add a new contact with a specific phone number and name.\n8. **remove_contact**: This tool allows me to remove a 
contact from my contact list.\n9. **update_contact**: This tool allows me to update the details of a specific contact, including their phone number and name.\n10. **get_status**: This tool allows me to get the status of a specific contact, including their online status and last seen time.\
"""

# Set up logging
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(log_dir, 'whatsapp_mcp.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)



@asynccontextmanager
async def create_whatsapp_graph(llm):
    logger.info("Initializing WhatsApp MCP client")
    try:
        async with MultiServerMCPClient(
            {
                "whatsapp": {
                    "command": "C:\\Users\\PC\\AppData\\Local\\Programs\\Python\\Python310\\Scripts\\uv.exe",
                    "args": ["--directory", "C:\\Users\\PC\\Desktop\\test\\whatsapp-mcp\\whatsapp-mcp-server", "run", "main.py"],
                    "transport": "stdio",
                }
            }
        ) as client:
            logger.info("MCP client connection established")
            # Create the agent with MCP tools
            
            agent = create_react_agent(llm, client.get_tools(),name="whatsapp")
            logger.info("React agent created successfully")
            yield agent
    except Exception as e:
        logger.error(f"Error in WhatsApp graph creation: {str(e)}", exc_info=True)
        raise

async def main(messages,llm):
    logger.info("Starting WhatsApp MCP agent")
    try:
        async with create_whatsapp_graph(llm) as agent:
            # Example: Send a message
            logger.info("Sending test message")
            response = await agent.ainvoke(
                {"messages": [{"role": "system", "content": prompt }]+messages}
               
            )
            #logger.info(f"Message sent successfully. Response: {response}")
            last_ai_message = None
            for message in reversed(response["messages"]):  # go backwards
                if isinstance(message, AIMessage):
                    last_ai_message = message
                    break

           
            messages.append({"role": "assistant", "content": last_ai_message.content})
            print(messages)
            return messages
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import asyncio
    try:
        messag= [{"role": "user", "content": "list your tools and how can you use them"},]
        asyncio.run(main(messag, model))
        logger.info("WhatsApp message sent successfully")
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)