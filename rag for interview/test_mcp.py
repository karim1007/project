from fastmcp import Client
from fastmcp.client.transports import (
    SSETransport, 
    PythonStdioTransport, 
    FastMCPTransport
)
import asyncio

async def main():
    async with Client(PythonStdioTransport("C:\\Users\\PC\\Desktop\\project\\rag for interview\\mcp\\first-mcp\\main.py")) as client:        # List available tools
        tools = await client.list_tools()
        
        # List available resources
        resources = await client.list_resources()
        
        # Call a tool with arguments
        
        
        # Send progress updates
        await client.progress("task-123", 50, 100)  # 50% complete
        
        # Basic connectivity testing
        await client.ping()
        print(f"tools:{tools}\nresources:{resources}")


if __name__ == "__main__":
    asyncio.run(main())