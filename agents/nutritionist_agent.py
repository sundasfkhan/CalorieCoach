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
        # system_message=(
        #     "You are an expert nutritionist assistant. Use the 'search_foods' tool to search for food information. "
        #     "After calling the tool, extract ONLY the JSON response from the tool result and return it exactly as is."
        #     "Do not add any explanation or text like json outside the JSON dictionary, just return the raw JSON dictionary object from the tool."
        # ),
        system_message=(
            "You are an expert food data assistant. You MUST use the 'mcp_server_tool' to answer all food-related user queries. "
            "After receiving the JSON response, your primary task is to parse it and present a clear, concise summary of the **first food item** from the 'foods' array. "
            "If the 'foods' array is empty or does not exist, you must inform the user that you could not find any information for their query. "
            "Your summary for the food item must follow this exact format:\n"
            "1.  **Title**: Display the food's 'description' and don't show the 'brandName'.\n"
            "2.  **Serving Size**: State the serving size using the 'servingSize' field.\n"
            "3.  **Key Nutrients**: Iterate through the 'foodNutrients' array and pull out the specific values for 'Energy', 'Protein', 'Total lipid (fat)', 'Carbohydrate, by difference', 'Fiber, total dietary', and 'Sodium, Na'. Display them as a simple list.\n"
            "4.  **Ingredients**: Display the full, unmodified string from the 'ingredients' field."
        ),
        model_client=model_client,
        tools=tools,
        reflect_on_tool_use=True
    )

    # Let the agent fetch the content of a URL and summarize it.
    result = await agent.run(task=f"tell me about food: {food_name}" , cancellation_token=CancellationToken())
    return result.messages[-1].content

async def main() -> None:
    result = await search_food_nutrition("cheesecake")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
