"""
Food Helper - Complete wrapper for MCP server food tools

This module provides a comprehensive set of functions that wrap MCP server calls
for food search, classification, and nutritional analysis. It includes OpenAI
integration for intelligent food summary analysis.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, cast

from dotenv import load_dotenv
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Load environment variables
load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER_PATH = ROOT / "mcp_server" / "mcp_server.py"


class FoodHelper:
    """Complete helper class for food-related operations using MCP server."""

    def __init__(self):
        self.mcp_server_path = str(MCP_SERVER_PATH)

    async def _get_mcp_session(self):
        """Create and return an MCP client session."""
        params = StdioServerParameters(
            command=sys.executable,
            args=[self.mcp_server_path],
            env=None
        )
        return stdio_client(params)

    async def search_foods(self, query: str) -> Dict[str, Any]:
        """
        Search for foods in the USDA FoodData Central database.

        Args:
            query: Search query for food items

        Returns:
            Dictionary containing search results or error information
        """
        try:
            async with await self._get_mcp_session() as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        name="search_foods",
                        arguments={"query": query}
                    )

                    if result.isError or not result.content:
                        return {"error": "Search failed", "query": query}

                    return json.loads(result.content[0].text)
        except Exception as e:
            return {"error": str(e), "query": query}

    # async def get_food_details(self, fdc_id: int, format: str = "full") -> Dict[str, Any]:
    #     """
    #     Get detailed information about a specific food item.
    #
    #     Args:
    #         fdc_id: Food Data Central ID
    #         format: Response format ('full' or 'abridged')
    #
    #     Returns:
    #         Dictionary containing food details or error information
    #     """
    #     try:
    #         async with await self._get_mcp_session() as (read_stream, write_stream):
    #             async with ClientSession(read_stream, write_stream) as session:
    #                 await session.initialize()
    #
    #                 result = await session.call_tool(
    #                     name="get_food_details",
    #                     arguments={"fdc_id": fdc_id, "format": format}
    #                 )
    #
    #                 if result.isError or not result.content:
    #                     return {"error": "Food details fetch failed", "fdc_id": fdc_id}
    #
    #                 return json.loads(result.content[0].text)
    #     except Exception as e:
    #         return {"error": str(e), "fdc_id": fdc_id}
    #
    # async def get_multiple_foods(self, fdc_ids: List[int], format: str = "full") -> Dict[str, Any]:
    #     """
    #     Get information about multiple food items.
    #
    #     Args:
    #         fdc_ids: List of Food Data Central IDs
    #         format: Response format ('full' or 'abridged')
    #
    #     Returns:
    #         Dictionary containing multiple food details or error information
    #     """
    #     try:
    #         async with await self._get_mcp_session() as (read_stream, write_stream):
    #             async with ClientSession(read_stream, write_stream) as session:
    #                 await session.initialize()
    #
    #                 result = await session.call_tool(
    #                     name="get_multiple_foods",
    #                     arguments={"fdc_ids": fdc_ids, "format": format}
    #                 )
    #
    #                 if result.isError or not result.content:
    #                     return {"error": "Multiple foods fetch failed", "fdc_ids": fdc_ids}
    #
    #                 return json.loads(result.content[0].text)
    #     except Exception as e:
    #         return {"error": str(e), "fdc_ids": fdc_ids}

    async def classify_food(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a food image using the MCP server's classify tool.

        Args:
            image_path: Path to the image file to classify

        Returns:
            Dictionary containing classification results or error information
        """
        try:
            async with await self._get_mcp_session() as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    result = await session.call_tool(
                        name="classify",
                        arguments={"image_path": image_path}
                    )

                    if result.isError or not result.content:
                        return {"error": "Classification failed", "image_path": image_path}

                    return json.loads(result.content[0].text)
        except Exception as e:
            return {"error": str(e), "image_path": image_path}

    def _get_openai_tools(self) -> List[Dict[str, Any]]:
        """Define OpenAI function tools that map to MCP server tools."""
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
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "get_food_details",
            #         "description": "Get detailed information about a specific food item",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "fdc_id": {
            #                     "type": "integer",
            #                     "description": "Food Data Central ID"
            #                 },
            #                 "format": {
            #                     "type": "string",
            #                     "enum": ["full", "abridged"],
            #                     "default": "full",
            #                     "description": "Response format"
            #                 }
            #             },
            #             "required": ["fdc_id"]
            #         }
            #     }
            # },
            # {
            #     "type": "function",
            #     "function": {
            #         "name": "get_multiple_foods",
            #         "description": "Get information about multiple food items",
            #         "parameters": {
            #             "type": "object",
            #             "properties": {
            #                 "fdc_ids": {
            #                     "type": "array",
            #                     "items": {"type": "integer"},
            #                     "description": "List of Food Data Central IDs"
            #                 },
            #                 "format": {
            #                     "type": "string",
            #                     "enum": ["full", "abridged"],
            #                     "default": "full",
            #                     "description": "Response format"
            #                 }
            #             },
            #             "required": ["fdc_ids"]
            #         }
            #     }
            # },
            {
                "type": "function",
                "function": {
                    "name": "classify_food",
                    "description": "Classify a food image and return the predicted class and confidence score",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Path to the image file to classify"
                            }
                        },
                        "required": ["image_path"]
                    }
                }
            }
        ]

    async def _handle_openai_tool_call(self, tool_call) -> str:
        """Handle OpenAI tool calls by routing to MCP server tools."""
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        if function_name == "search_foods":
            result = await self.search_foods(**arguments)
        # elif function_name == "get_food_details":
        #     result = await self.get_food_details(**arguments)
        # elif function_name == "get_multiple_foods":
        #     result = await self.get_multiple_foods(**arguments)
        elif function_name == "classify_food":
            result = await self.classify_food(**arguments)
        else:
            result = {"error": f"Unknown function: {function_name}"}

        return json.dumps(result)

    async def food_summary_with_openai(self, food_items: List[str]) -> str:
        """
        Get comprehensive food analysis using OpenAI with MCP tools.

        Args:
            food_items: List of food item names to analyze

        Returns:
            Comprehensive nutritional analysis as a string
        """
        try:
            from openai import OpenAI
            from openai.types.chat import ChatCompletionMessageParam
        except ImportError:
            return "Error: OpenAI package not installed. Please install with 'pip install openai'"

        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            return "Error: OPENAI_API_KEY not set in environment variables"

        client = OpenAI()
        model = os.getenv("OPENAI_TEST_MODEL", "gpt-4o-mini")

        # Initial system message
        messages = cast(List[ChatCompletionMessageParam], [
            {
                "role": "system",
                "content": (
                    "You are a senior nutrition expert. Use the available tools to search for and analyze "
                    "food items. For each food item provided, extract and summarize clearly with sections: "
                    "Nutrients, Portions, Other Information, Ingredients. Be concise and practical. "
                    "If a section is unavailable, state 'Not available'. Focus on the most important "
                    "nutritional information like calories, protein, carbohydrates, fats, vitamins, and minerals."
                )
            },
            {
                "role": "user",
                "content": f"Please analyze the nutrition information for these food items: {', '.join(food_items)}"
            }
        ])

        tools = self._get_openai_tools()
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=2000,
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
                    tool_result = await self._handle_openai_tool_call(tool_call)
                    messages.append(cast(ChatCompletionMessageParam, {
                        "role": "tool",
                        "content": tool_result,
                        "tool_call_id": tool_call.id
                    }))

            except Exception as e:
                return f"Error during OpenAI analysis: {str(e)}"

        return "Maximum iterations reached. Analysis may be incomplete."

    async def food_summary_basic(self, label: str) -> Dict[str, Any]:
        """
        Get basic nutritional summary for a food item (without OpenAI).

        Args:
            label: The food label/name to search for

        Returns:
            Dictionary containing nutritional information and summary
        """
        try:
            # Search for the food in the database
            search_result = await self.search_foods(label)

            if "error" in search_result:
                return {
                    "label": label,
                    "error": "Food search failed",
                    "summary": f"Unable to find nutritional information for {label}",
                    "source": "food_helper"
                }

            if not search_result.get("foods") or len(search_result["foods"]) == 0:
                return {
                    "label": label,
                    "error": "No foods found",
                    "summary": f"No nutritional data available for {label}",
                    "source": "food_helper"
                }

            # Get the first food item's details
            first_food = search_result["foods"][0]
            fdc_id = first_food.get("fdcId")

            if not fdc_id:
                return {
                    "label": label,
                    "food_name": first_food.get("description", label),
                    "summary": f"Found {first_food.get('description', label)} but detailed nutrition data unavailable",
                    "source": "food_helper"
                }

            # Get detailed nutritional information
            details_result = await self.get_food_details(fdc_id, format="full")

            if "error" in details_result:
                return {
                    "label": label,
                    "food_name": first_food.get("description", label),
                    "summary": f"Found {first_food.get('description', label)} but failed to retrieve detailed nutrition information",
                    "source": "food_helper"
                }

            # Extract key nutritional information
            nutrients = {}
            if "foodNutrients" in details_result:
                for nutrient in details_result["foodNutrients"]:
                    nutrient_name = nutrient.get("nutrient", {}).get("name", "")
                    nutrient_value = nutrient.get("amount", 0)
                    nutrient_unit = nutrient.get("nutrient", {}).get("unitName", "")

                    if nutrient_name and nutrient_value:
                        nutrients[nutrient_name] = {
                            "amount": nutrient_value,
                            "unit": nutrient_unit
                        }

            # Create summary
            food_name = details_result.get("description", first_food.get("description", label))
            calories = nutrients.get("Energy", {}).get("amount", "Unknown")
            protein = nutrients.get("Protein", {}).get("amount", "Unknown")
            carbs = nutrients.get("Carbohydrate, by difference", {}).get("amount", "Unknown")
            fat = nutrients.get("Total lipid (fat)", {}).get("amount", "Unknown")
            fiber = nutrients.get("Fiber, total dietary", {}).get("amount", "Unknown")
            sodium = nutrients.get("Sodium, Na", {}).get("amount", "Unknown")

            summary_text = f"Nutritional information for {food_name}:\n"
            summary_text += f"• Calories: {calories} kcal per 100g\n" if calories != "Unknown" else ""
            summary_text += f"• Protein: {protein}g per 100g\n" if protein != "Unknown" else ""
            summary_text += f"• Carbohydrates: {carbs}g per 100g\n" if carbs != "Unknown" else ""
            summary_text += f"• Fat: {fat}g per 100g\n" if fat != "Unknown" else ""
            summary_text += f"• Fiber: {fiber}g per 100g\n" if fiber != "Unknown" else ""
            summary_text += f"• Sodium: {sodium}mg per 100g" if sodium != "Unknown" else ""

            return {
                "label": label,
                "food_name": food_name,
                "fdc_id": fdc_id,
                "calories_per_100g": calories,
                "protein_per_100g": protein,
                "carbs_per_100g": carbs,
                "fat_per_100g": fat,
                "fiber_per_100g": fiber,
                "sodium_per_100g": sodium,
                "nutrients": nutrients,
                "summary": summary_text.strip(),
                "source": "USDA FoodData Central"
            }

        except Exception as e:
            return {
                "label": label,
                "error": str(e),
                "summary": f"Error retrieving nutritional information for {label}: {str(e)}",
                "source": "food_helper"
            }

    async def batch_food_analysis(self, food_items: List[str], use_openai: bool = True) -> Dict[str, Any]:
        """
        Analyze multiple food items at once.

        Args:
            food_items: List of food item names
            use_openai: Whether to use OpenAI for analysis (default: True)

        Returns:
            Dictionary containing analysis results
        """
        if use_openai:
            # Use OpenAI for comprehensive analysis
            analysis = await self.food_summary_with_openai(food_items)
            return {
                "method": "openai",
                "food_items": food_items,
                "analysis": analysis,
                "source": "OpenAI + USDA FoodData Central"
            }
        else:
            # Use basic analysis for each item
            results = {}
            for item in food_items:
                results[item] = await self.food_summary_basic(item)

            return {
                "method": "basic",
                "food_items": food_items,
                "individual_results": results,
                "source": "USDA FoodData Central"
            }


# Convenience functions for direct use
async def search_foods(query: str) -> Dict[str, Any]:
    """Search for foods in the USDA FoodData Central database."""
    helper = FoodHelper()
    return await helper.search_foods(query)


# async def get_food_details(fdc_id: int, format: str = "full") -> Dict[str, Any]:
#     """Get detailed information about a specific food item."""
#     helper = FoodHelper()
#     return await helper.get_food_details(fdc_id, format)
#
#
# async def get_multiple_foods(fdc_ids: List[int], format: str = "full") -> Dict[str, Any]:
#     """Get information about multiple food items."""
#     helper = FoodHelper()
#     return await helper.get_multiple_foods(fdc_ids, format)


async def classify_food(image_path: str) -> Dict[str, Any]:
    """Classify a food image."""
    helper = FoodHelper()
    return await helper.classify_food(image_path)


async def food_summary(label: str, use_openai: bool = True) -> Any:
    """
    Get comprehensive nutritional summary for a food item.

    Args:
        label: The food label/name to search for
        use_openai: Whether to use OpenAI for analysis (default: True)

    Returns:
        String (if use_openai=True) or Dictionary (if use_openai=False) containing analysis
    """
    helper = FoodHelper()

    if use_openai:
        return await helper.food_summary_with_openai([label])
    else:
        return await helper.food_summary_basic(label)


async def batch_food_analysis(food_items: List[str], use_openai: bool = True) -> Dict[str, Any]:
    """Analyze multiple food items at once."""
    helper = FoodHelper()
    return await helper.batch_food_analysis(food_items, use_openai)


# Example usage
if __name__ == "__main__":
    async def example_usage():
        # Initialize helper
        helper = FoodHelper()

        #Example 1: Search for foods
        # print("=== Searching for 'chai' ===")
        # search_result = await helper.search_foods("chai")
        # print(json.dumps(search_result, indent=2))

        # Example 2: OpenAI-powered analysis (if API key is available)
        if os.getenv("OPENAI_API_KEY"):
            print("\n=== OpenAI Food Analysis ===")
            ai_analysis = await helper.food_summary_with_openai(["dhokla", "cheesecake", "chai"])
            print(ai_analysis)


        # food_name = await helper.classify_food(".\\..\\data\Test\\cheesecake\\cheesecake-1314.jpg")
        # print(food_name)         # Run the example

    asyncio.run(example_usage())
