import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

import anyio
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER_PATH = ROOT / "mcp_server/mcp_server.py"


async def food_summary(label: str) -> Dict[str, Any]:
    """
    Get nutritional summary for a food item based on its label.

    Args:
        label: The food label/name to search for

    Returns:
        Dictionary containing nutritional information and summary
    """
    try:
        # Search for the food in the database
        params = StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER_PATH)], env=None)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Search for foods matching the label
                search_result = await session.call_tool(
                    name="search_foods",
                    arguments={"query": label, "pageSize": 5, "pageNumber": 1}
                )

                if search_result.isError or not search_result.content:
                    return {
                        "label": label,
                        "error": "Food search failed",
                        "summary": f"Unable to find nutritional information for {label}",
                        "source": "food_summary"
                    }

                search_data = json.loads(search_result.content[0].text)

                if not search_data.get("foods") or len(search_data["foods"]) == 0:
                    return {
                        "label": label,
                        "error": "No foods found",
                        "summary": f"No nutritional data available for {label}",
                        "source": "food_summary"
                    }

                # Get the first food item's details
                first_food = search_data["foods"][0]
                fdc_id = first_food.get("fdcId")

                if not fdc_id:
                    return {
                        "label": label,
                        "food_name": first_food.get("description", label),
                        "summary": f"Found {first_food.get('description', label)} but detailed nutrition data unavailable",
                        "source": "food_summary"
                    }

                # Get detailed nutritional information
                details_result = await session.call_tool(
                    name="get_food_details",
                    arguments={"fdc_id": fdc_id, "format": "full"}
                )

                if details_result.isError or not details_result.content:
                    return {
                        "label": label,
                        "food_name": first_food.get("description", label),
                        "summary": f"Found {first_food.get('description', label)} but failed to retrieve detailed nutrition information",
                        "source": "food_summary"
                    }

                details_data = json.loads(details_result.content[0].text)

                # Extract key nutritional information
                nutrients = {}
                if "foodNutrients" in details_data:
                    for nutrient in details_data["foodNutrients"]:
                        nutrient_name = nutrient.get("nutrient", {}).get("name", "")
                        nutrient_value = nutrient.get("amount", 0)
                        nutrient_unit = nutrient.get("nutrient", {}).get("unitName", "")

                        if nutrient_name and nutrient_value:
                            nutrients[nutrient_name] = {
                                "amount": nutrient_value,
                                "unit": nutrient_unit
                            }

                # Create summary
                food_name = details_data.get("description", first_food.get("description", label))
                calories = nutrients.get("Energy", {}).get("amount", "Unknown")
                protein = nutrients.get("Protein", {}).get("amount", "Unknown")
                carbs = nutrients.get("Carbohydrate, by difference", {}).get("amount", "Unknown")
                fat = nutrients.get("Total lipid (fat)", {}).get("amount", "Unknown")

                summary_text = f"Nutritional information for {food_name}:\n"
                summary_text += f"• Calories: {calories} kcal per 100g\n" if calories != "Unknown" else ""
                summary_text += f"• Protein: {protein}g per 100g\n" if protein != "Unknown" else ""
                summary_text += f"• Carbohydrates: {carbs}g per 100g\n" if carbs != "Unknown" else ""
                summary_text += f"• Fat: {fat}g per 100g" if fat != "Unknown" else ""

                return {
                    "label": label,
                    "food_name": food_name,
                    "fdc_id": fdc_id,
                    "calories_per_100g": calories,
                    "protein_per_100g": protein,
                    "carbs_per_100g": carbs,
                    "fat_per_100g": fat,
                    "nutrients": nutrients,
                    "summary": summary_text.strip(),
                    "source": "USDA FoodData Central"
                }

    except Exception as e:
        return {
            "label": label,
            "error": str(e),
            "summary": f"Error retrieving nutritional information for {label}: {str(e)}",
            "source": "food_summary"
        }
