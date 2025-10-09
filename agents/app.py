import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import SseMcpToolAdapter, SseServerParams ,StdioMcpToolAdapter ,StdioServerParams
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from dotenv import load_dotenv

load_dotenv()

async def main() -> None:

    # Create server params for the local MCP tool process
    server_params = StdioServerParams(
        command=["python", "/mcp_server/mcp_server.py"],  # Path to your MCP server script
        timeout=30,
    )


    # Get the translation tool from the local process
    adapter = await StdioMcpToolAdapter.from_server_params(server_params, "search_foods")

    # Create an agent that can use the translation tool
    model_client = OpenAIChatCompletionClient(model="gpt-4")
    agent = AssistantAgent(
        name="chief",
        model_client=model_client,
        tools=[adapter],
        system_message="You are a helpful food assistant. Search query for food items",
    )

    # Let the agent translate some text
    await Console(
        agent.run_stream(task="can you tell me samosa details", cancellation_token=CancellationToken())
    )


if __name__ == "__main__":
    asyncio.run(main())
