import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import SseMcpToolAdapter, SseServerParams ,StdioServerParams, mcp_server_tools ,StdioServerParams
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from dotenv import load_dotenv

load_dotenv()

async def main() -> None:

    # Create server params for the local MCP tool process
    mcp_server_prxy = StdioServerParams(command="python", args=[".././mcp_server/mcp_server.py"])
    tools = await mcp_server_tools(mcp_server_prxy)
    print("MCP Server tools loaded")
    # Create an agent that can use the fetch tool.
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    agent = AssistantAgent(
        name="nutritionist",
        system_message=(
            "You are an expert nutritionist assistant. Your primary function is to use the 'mcp_server_tool' to answer user questions. "
            "You MUST call the 'mcp_server_tool' whenever a user asks about food information, nutritional values, meal plans, or recipes. "
            "Do not answer from your own knowledge base. Rely exclusively on the tool's output for your response."
        ),
        model_client=model_client,
        tools=tools,  # Make sure 'mcp_server_tool' is correctly defined in this list
        reflect_on_tool_use=True
    )

    # Let the agent fetch the content of a URL and summarize it.
    result = await agent.run(task="tell me about food: samosa" , cancellation_token=CancellationToken())
    print(result.messages[-1].content)




if __name__ == "__main__":
    asyncio.run(main())
