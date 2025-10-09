import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import SseMcpToolAdapter, SseServerParams ,StdioServerParams, mcp_server_tools ,StdioServerParams
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from dotenv import load_dotenv

load_dotenv()

async def search_food_nutrition(food_name: str):
    """Search for nutrition information about a food item."""
    # Create server params for the local MCP tool process
    mcp_server_prxy = StdioServerParams(command="python", args=["../mcp_server/mcp_server.py"])
    tools = await mcp_server_tools(mcp_server_prxy)
    # Create an agent that can use the fetch tool.
    model_client = OpenAIChatCompletionClient(model="gpt-4o")

    agent = AssistantAgent(
        name="nutritionist",
        system_message=(
            "You are an expert nutritionist assistant. Use the 'search_foods' tool to search for food information. "
            "After calling the tool, extract ONLY the JSON response from the tool result and return it exactly as is. "
            "Do not add any explanation or text, just return the raw JSON object from the tool."
        ),
        model_client=model_client,
        tools=tools,
        reflect_on_tool_use=True
    )

    # Let the agent fetch the content of a URL and summarize it.
    result = await agent.run(task=f"tell me about food: {food_name}" , cancellation_token=CancellationToken())
    return result.messages[-1].content

async def main() -> None:
    result = await search_food_nutrition("samosa")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
