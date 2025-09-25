import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast, Any

import anyio
from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER_PATH = ROOT / "mcp_server//mcp_server.py"


async def search_foods_tool(query: str) -> str:
    """Search for foods in the USDA FoodData Central database"""
    params = StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER_PATH)], env=None)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            arguments = {"query": query}

            try:
                result = await session.call_tool(name="search_foods", arguments=arguments)
                if result.isError or not result.content:
                    return json.dumps({"error": "Search failed"})
                return result.content[0].text
            except Exception as e:
                return json.dumps({"error": str(e)})


async def get_food_details_tool(fdc_id: int, format: str = "full") -> str:
    """Get detailed information about a specific food item"""
    params = StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER_PATH)], env=None)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            try:
                result = await session.call_tool(
                    name="get_food_details",
                    arguments={"fdc_id": fdc_id, "format": format}
                )
                if result.isError or not result.content:
                    return json.dumps({"error": "Food details fetch failed"})
                return result.content[0].text
            except Exception as e:
                return json.dumps({"error": str(e)})


async def get_multiple_foods_tool(fdc_ids: list[int], format: str = "full") -> str:
    """Get information about multiple food items"""
    params = StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER_PATH)], env=None)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            try:
                result = await session.call_tool(
                    name="get_multiple_foods",
                    arguments={"fdc_ids": fdc_ids, "format": format}
                )
                if result.isError or not result.content:
                    return json.dumps({"error": "Multiple foods fetch failed"})
                return result.content[0].text
            except Exception as e:
                return json.dumps({"error": str(e)})


async def classify_food_tool(image_path: str) -> str:
    """Classify a food image using the MCP server's classify tool."""
    params = StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER_PATH)], env=None)
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            try:
                result = await session.call_tool(
                    name="classify",
                    arguments={"image_path": image_path}
                )
                if result.isError or not result.content:
                    return json.dumps({"error": "Classification failed"})
                return result.content[0].text
            except Exception as e:
                return json.dumps({"error": str(e)})


def get_openai_tools():
    """Define OpenAI function tools that map to MCP server tools"""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_foods",
                "description": "Search for foods in the USDA FoodData Central database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for food items"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_food_details",
                "description": "Get detailed information about a specific food item",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fdc_id": {
                            "type": "integer",
                            "description": "Food Data Central ID"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["full", "abridged"],
                            "default": "full",
                            "description": "Response format"
                        }
                    },
                    "required": ["fdc_id"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_multiple_foods",
                "description": "Get information about multiple food items",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fdc_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "List of Food Data Central IDs"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["full", "abridged"],
                            "default": "full",
                            "description": "Response format"
                        }
                    },
                    "required": ["fdc_ids"]
                }
            }
        }
    ]


async def handle_tool_call(tool_call):
    """Handle OpenAI tool calls by routing to MCP server tools"""
    function_name = tool_call.function.name
    arguments = json.loads(tool_call.function.arguments)

    if function_name == "search_foods":
        return await search_foods_tool(**arguments)
    elif function_name == "get_food_details":
        return await get_food_details_tool(**arguments)
    elif function_name == "get_multiple_foods":
        return await get_multiple_foods_tool(**arguments)
    elif function_name == "classify":
        return await classify_food_tool(**arguments)
    else:
        return json.dumps({"error": f"Unknown function: {function_name}"})


async def chat_with_openai(food_items: list[str]) -> str:
    """Chat with OpenAI using MCP tools for nutrition analysis"""
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessageParam

    client = OpenAI()
    model = os.getenv("OPENAI_TEST_MODEL", "gpt-4o-mini")

    # Initial system message
    messages = cast(list[ChatCompletionMessageParam], [
        {
            "role": "system",
            "content": (
                "You are a senior nutrition expert. Use the available tools to search for and analyze "
                "food items. For each food item provided, extract and summarize clearly with sections: "
                "Nutrients, Portions, Other Information, Ingredients. Be concise and practical. "
                "If a section is unavailable, state 'Not available'."
            )
        },
        {
            "role": "user",
            "content": f"Please analyze the nutrition information for these food items: {', '.join(food_items)}"
        }
    ])

    tools = get_openai_tools()
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1000,
        )

        message = response.choices[0].message
        messages.append(cast(ChatCompletionMessageParam, {
            "role": "assistant",
            "content": message.content,
            "tool_calls": message.tool_calls
        }))

        if not message.tool_calls:
            # No more tool calls, return the final response
            return message.content or "No response generated"

        # Handle tool calls
        for tool_call in message.tool_calls:
            tool_result = await handle_tool_call(tool_call)
            messages.append(cast(ChatCompletionMessageParam, {
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call.id
            }))

    return "Maximum iterations reached. Analysis may be incomplete."


def food_summary(argv: list[str]) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze nutrition info for food items using OpenAI + MCP tools")
    parser.add_argument("items", nargs="*", help="Food item names. Example: apple banana 'greek yogurt'")
    parser.add_argument("-i", "--input", help="Comma- or semicolon-separated list of items (alternative to positional)")
    args = parser.parse_args(argv)

    usda_key = os.getenv("USDA_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not usda_key:
        print("Error: USDA_API_KEY is not set (in env or .env)", file=sys.stderr)
        return 2
    if not openai_key:
        print("Error: OPENAI_API_KEY is not set (in env or .env)", file=sys.stderr)
        return 2

    # Parse items
    terms: list[str] = []
    if args.input:
        sep = ";" if ";" in args.input else ","
        terms = [t.strip() for t in args.input.split(sep) if t.strip()]
    if args.items:
        terms.extend(args.items)
    # Deduplicate while preserving order
    seen = set()
    terms = [t for t in terms if not (t in seen or seen.add(t))]

    print(terms)

    if not terms:
        print("Provide food item names via positional args or --input.")
        print("Example: python tools/food_summary_cli.py apple banana")
        return 1

    try:
        # Let OpenAI use MCP tools directly
        summary = anyio.run(chat_with_openai, terms)
        print("\n=== Nutrition Analysis ===\n")
        print(summary)
        print("\n=========================\n")
        return 0
    except Exception as e:
        print(f"Error during OpenAI + MCP flow: {e}", file=sys.stderr)
        print("Hint: Ensure app.py is running on port 8004 before using this tool.", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(food_summary(sys.argv[1:]))
