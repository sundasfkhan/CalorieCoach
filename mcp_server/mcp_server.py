import asyncio
import json
import httpx
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import Resource, Tool, TextContent
from mpmath.libmp import BACKEND
from pydantic import AnyUrl
import mcp.types as types

"""MCP Server for Food Data Central tools.

This module exposes a set of tools (via MCP protocol) that an LLM can call to
interact with a local Flask API (see app.py) running on http://localhost:8004.

Exposed tools (LLM-callable):
- search_foods(args): Find foods using USDA FDC search
  Input schema:
    {
      query: string (required)
    }
  Output: list with one TextContent item containing the JSON string result
  Errors: returns a single TextContent with an error message on failure

- get_food_details(args): Fetch details for a single food by FDC ID
  Input schema:
    {
      fdc_id: integer (required),
      format?: 'full'|'abridged' (default 'full')
    }
  Output: list with one TextContent item containing the JSON string result
  Errors: returns a single TextContent with an error message on failure

- get_multiple_foods(args): Fetch details for multiple foods
  Input schema:
    {
      fdc_ids: integer[] (required),
      format?: 'full'|'abridged' (default 'full')
    }
  Output: list with one TextContent item containing the JSON string result
  Errors: returns a single TextContent with an error message on failure

- classify(args): Classify a food image and return the predicted class and confidence score.
  Input schema:
    {
      image_path: string (required): Path to the image file to classify. Must be accessible to the server.
    }
  Output: list with one TextContent item containing the JSON string result
  Errors: returns a single TextContent with an error message on failure

General behavior:
- This server proxies requests to the Flask API; it does not talk to USDA directly.
- All outputs are returned as an array of content blocks (here, TextContent only).
- Network issues or non-2xx responses will surface as error TextContent.
"""

# Configuration
BACKEND_URL = "http://localhost:8004"


class FoodDataMCPServer:
    """Model Context Protocol server exposing Food Data tools for LLMs.

    Responsibilities:
    - Register resources (descriptive entries) and tools (callable functions)
    - Validate and forward tool calls to the Flask API
    - Normalize outputs into MCP content blocks for the LLM

    Lifecycle:
    - Instantiate, then call run() to start stdio server for MCP transport.
    """

    def __init__(self):
        self.server = Server("food-data-central")
        self.client = httpx.AsyncClient()

    async def setup_handlers(self):
        """Setup MCP server handlers for resources and tools.

        Handlers registered:
        - list_resources: Announces resource catalog entries
        - read_resource: Reads simple string descriptions for known URIs
        - list_tools: Publishes available tool metadata and input schemas
        - call_tool: Entry point for LLM tool invocations
        """

        @self.server.list_resources()
        async def handle_list_resources() -> list[Resource]:
            """List available resources for discovery.

            Outputs: a list of Resource with uri/name/description/mimeType.
            Errors: raises ValueError only if misconfigured (not expected).
            """
            return [
                Resource(
                    uri=AnyUrl("food://search"),
                    name="Food Search",
                    description="Search for foods in USDA database",
                    mimeType="application/json"
                ),
                Resource(
                    uri=AnyUrl("food://details"),
                    name="Food Details",
                    description="Get detailed food information",
                    mimeType="application/json"
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: AnyUrl) -> str:
            """Read resource content for a known URI.

            Inputs:
            - uri: AnyUrl, e.g., food://search or food://details
            Outputs:
            - Human-readable instruction string for how to use the tools
            Errors:
            - ValueError if the URI is not recognized
            """
            if str(uri) == "food://search":
                return "Use the search_foods tool to search for foods"
            elif str(uri) == "food://details":
                return "Use the get_food_details tool to get food information"
            else:
                raise ValueError(f"Unknown resource: {uri}")

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Publish available tools and their input schemas for the LLM.

            Outputs: array of Tool with JSON schema for inputs.
            Notes: All tools return a list[TextContent] on success/failure.
            """
            return [
                Tool(
                    name="search_foods",
                    description="Search for foods in the USDA FoodData Central database",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for food items"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                # Tool(
                #     name="get_food_details",
                #     description="Get detailed information about a specific food item",
                #     inputSchema={
                #         "type": "object",
                #         "properties": {
                #             "fdc_id": {
                #                 "type": "integer",
                #                 "description": "Food Data Central ID"
                #             },
                #             "format": {
                #                 "type": "string",
                #                 "enum": ["full", "abridged"],
                #                 "default": "full",
                #                 "description": "Response format"
                #             }
                #         },
                #         "required": ["fdc_id"]
                #     }
                # ),
                # Tool(
                #     name="get_multiple_foods",
                #     description="Get information about multiple food items",
                #     inputSchema={
                #         "type": "object",
                #         "properties": {
                #             "fdc_ids": {
                #                 "type": "array",
                #                 "items": {"type": "integer"},
                #                 "description": "List of Food Data Central IDs"
                #             },
                #             "format": {
                #                 "type": "string",
                #                 "enum": ["full", "abridged"],
                #                 "default": "full",
                #                 "description": "Response format"
                #             }
                #         },
                #         "required": ["fdc_ids"]
                #     }
                # ),
                Tool(
                    name="classify",
                    description="Classify a food image and return the predicted class and confidence score.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Path to the image file to classify. Must be accessible to the server."
                            }
                        },
                        "required": ["image_path"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> list[
            types.TextContent | types.ImageContent | types.EmbeddedResource]:
            """Dispatch a tool call requested by the LLM.

            Inputs:
            - name: one of 'search_foods' | 'get_food_details' | 'get_multiple_foods'
            - arguments: JSON object matching the tool's inputSchema

            Outputs:
            - list with a single TextContent in all cases
              - On success: text contains the JSON response as a string
              - On error:  text contains an 'Error: ...' message

            Error modes:
            - ValueError for unknown tool names
            - Any exception during HTTP calls is caught and returned as TextContent
            """
            try:
                if name == "search_foods":
                    return await self._search_foods(arguments)
                elif name == "get_food_details":
                    return await self._get_food_details(arguments)
                elif name == "get_multiple_foods":
                    return await self._get_multiple_foods(arguments)
                elif name == "classify":
                    return await self._classify(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def _search_foods(self, args: dict) -> list[TextContent]:
        """Search for foods via the Flask API.

        Inputs (args):
        - query (str, required)
        Behavior: GET {FLASK_API_BASE_URL}/api/search?food_name=<query>
        Success output: [TextContent(text='<json>')]
        Errors: raises httpx.HTTPError (caught by caller and returned as TextContent)
        Example call:
        - {"name":"search_foods","arguments":{"query":"apple"}}
        """
        params = {
            "food_name": args["query"]
        }

        response = await self.client.get(f"{BACKEND_URL}/api/search", params=params)
        response.raise_for_status()

        result = response.json()
        return [TextContent(type="text", text=json.dumps(result))]

    async def _get_food_details(self, args: dict) -> list[TextContent]:
        """Get single food details via the Flask API.

        Inputs (args):
        - fdc_id (int, required)
        - format (str, optional)
        Behavior: GET {FLASK_API_BASE_URL}/api/food/{fdc_id}
        Success output: [TextContent(text='<json>')]
        Errors: raises httpx.HTTPError (caught by caller and returned as TextContent)
        Example call:
        - {"name":"get_food_details","arguments":{"fdc_id":2344719,"format":"abridged"}}
        """
        fdc_id = args["fdc_id"]
        params = {"format": args.get("format", "full")}

        response = await self.client.get(f"{BACKEND_URL}/api/food/{fdc_id}", params=params)
        response.raise_for_status()

        result = response.json()
        return [TextContent(type="text", text=json.dumps(result))]

    async def _get_multiple_foods(self, args: dict) -> list[TextContent]:
        """Get multiple food details via the Flask API.

        Inputs (args):
        - fdc_ids (int[], required)
        - format (str, optional)
        Behavior: GET {FLASK_API_BASE_URL}/api/foods?fdcIds=...&format=...
        Success output: [TextContent(text='<json>')]
        Errors: raises httpx.HTTPError (caught by caller and returned as TextContent)
        Example call:
        - {"name":"get_multiple_foods","arguments":{"fdc_ids":[2344719,2344720]}}
        """
        fdc_ids = ",".join(map(str, args["fdc_ids"]))
        params = {
            "fdcIds": fdc_ids,
            "format": args.get("format", "full")
        }

        response = await self.client.get(f"{BACKEND_URL}/api/foods", params=params)
        response.raise_for_status()

        result = response.json()
        return [TextContent(type="text", text=json.dumps(result))]

    async def _classify(self, args: dict) -> list[TextContent]:
        """Classify a food image via the Flask API.
        Inputs (args):
        - image_path (str, required): Path to the image file accessible to the server.
        Behavior: POST {FLASK_API_BASE_URL}/api/classify with image file
        Success output: [TextContent(text='<json>')]
        Errors: raises httpx.HTTPError (caught by caller and returned as TextContent)
        """
        image_path = args["image_path"]
        # Open the image file and send as multipart/form-data
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            response = await self.client.post(f"{BACKEND_URL}/api/classify", files=files)
            response.raise_for_status()
            result = response.json()
            return [TextContent(type="text", text=json.dumps(result))]

    async def run(self):
        """Run the MCP server with stdio transport.

        Behavior:
        - Registers handlers
        - Opens stdio transport and runs the MCP server loop
        """
        await self.setup_handlers()

        # Run server with stdio transport
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="food-data-central",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )


async def main():
    """Entry point for running the MCP server.

    Creates FoodDataMCPServer and runs it on stdio transport.
    """
    server = FoodDataMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
