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
    agent = AssistantAgent(name="nutritionist" , system_message="You are a helpful food Nutritionist assistant. Please call the tool 'search_foods' when user is asking about any food. Tool 'search_foods' then search the foods details in the USDA FoodData Central database" , model_client=model_client, tools=tools,
                           reflect_on_tool_use=True)  # type: ignore

    # Let the agent fetch the content of a URL and summarize it.
    result = await agent.run(task="tell me about food: samosa" , cancellation_token=CancellationToken())
    print(result.messages[-1])




if __name__ == "__main__":
    asyncio.run(main())
