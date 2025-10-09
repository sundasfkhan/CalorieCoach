import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioMcpToolAdapter,  mcp_server_tools ,StdioServerParams
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from dotenv import load_dotenv

load_dotenv()

async def classify_food_image(image_path: str):
    """Classify a food image and return the result."""
    # Create server params for the local MCP tool process
    mcp_server_prxy = StdioServerParams(command="python", args=["../mcp_server/mcp_server.py"])
    tools = await mcp_server_tools(mcp_server_prxy)
    # Create an agent that can use the classify tool
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    agent = AssistantAgent(
        name="food_image_classifier",
        system_message=(
            "You are a food image classifier. Use the 'classify' tool to classify food images. "
            "After calling the tool, extract ONLY the JSON response from the tool result and return it exactly as is. "
            "Do not add any explanation or text, just return the raw JSON object from the tool."
        ),
        model_client=model_client,
        tools=tools,
        reflect_on_tool_use=True
    )

    # Classify the food image
    result = await agent.run(task=f"classify this food image: {image_path}", cancellation_token=CancellationToken())
    return result.messages[-1].content

async def main() -> None:
    result = await classify_food_image("C:\\Projects\\CalorieCoach\\data\\Test\\chicken_curry\\chicken_curry-1016.jpg")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
