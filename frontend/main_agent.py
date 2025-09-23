import os
import base64
from openai import OpenAI


def encode_image_to_base64(image_path):
    """Encodes an image to a base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: The file at {image_path} was not found.")
        return None

def analyze_food_image(image_path):
    """Analyzes a food image using OpenAI's GPT-4o model to find ingredients."""
    # Initialize the OpenAI client with the API key from environment variables
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Encode the image
    base64_image = encode_image_to_base64(image_path)
    if not base64_image:
        return "Image processing failed."

    print("Analyzing image and requesting ingredients from OpenAI...")

    try:
        # Create a chat completion request with an image and a text prompt
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What are the ingredients in this dish? List them concisely."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=300,
        )

        # Extract and return the ingredients from the response
        ingredients = response.choices[0].message.content.strip()
        return ingredients
    except Exception as e:
        return f"An error occurred: {e}"

# --- Main part of the script ---
if __name__ == "__main__":
    # Specify the path to your image file
    # Make sure to replace 'food_image.jpg' with your image file name.
    # Place the image in the same directory as this script.
    image_file_path = "C:\Projects\StreamlitTest\pizza-1003.jpg"

    if os.path.exists(image_file_path):
        ingredients_list = analyze_food_image(image_file_path)
        print("\n--- Identified Ingredients ---")
        print(ingredients_list)
    else:
        print(f"Image file '{image_file_path}' not found. Please place a food image in the script's directory and rename it accordingly.")