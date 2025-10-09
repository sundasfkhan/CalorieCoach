import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioMcpToolAdapter,  mcp_server_tools ,StdioServerParams
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

    # Create an agent that can use the classify tool
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    agent = AssistantAgent(
        name="food_image_classifier",
        system_message=(
            "You are a food image classifier. Your primary function is to use the 'classify' tool to classify food images. "
            "When given an image path, call the classify tool and return the classification result as a string."
        ),
        model_client=model_client,
        tools=tools,  # Make sure 'classify' is correctly defined in this list
        reflect_on_tool_use=True
    )

    # Example task: classify a food image
    result = await agent.run(task="classify this food image: C:\Projects\CalorieCoach\data\Test\chicken_curry\chicken_curry-1016.jpg")  #C:\Projects\CalorieCoach\data\Test\donut\Donut-10005.jpeg", cancellation_token=CancellationToken())
    print(result.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())
