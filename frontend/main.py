import os
import base64
from openai import OpenAI
import torch
from torchvision import models, transforms
from PIL import Image


# --- Function to use your pre-trained PyTorch model ---
def get_food_class_from_model(image_path):
    """
    Loads a pre-trained PyTorch model to classify the food in the image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The predicted class name of the food.
    """
    try:
        # Load a pre-trained model (e.g., ResNet18) and set it to evaluation mode
        model = models.efficientnet_v2_m(pretrained=True)
        model.eval()

        # Define the image transformations required by the pre-trained model
        # Pre-trained models on ImageNet expect input images of a specific size and normalization
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load the image using PIL and apply the transformations
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)  # Add a batch dimension

        # Make a prediction with the model
        with torch.no_grad():
            outputs = model(img_tensor)

        # Get the predicted class index
        _, predicted_idx = torch.max(outputs, 1)

        # Load the class names to map the index to a human-readable name
        # You would need to load your custom class names here.
        # This example uses ImageNet's class labels as a placeholder.
        # You can find these labels in a text file online, for example.
        with open("imagenet_classes.txt") as f:
            class_names = [line.strip() for line in f.readlines()]

        predicted_food_name = class_names[predicted_idx.item()]

        return predicted_food_name

    except FileNotFoundError:
        print(f"Error: The image file at {image_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred during model prediction: {e}")
        return None


# --- Function to query OpenAI based on the food name ---
def get_nutritional_info_from_openai(food_name):
    """
    Queries OpenAI for ingredients and nutritional information based on a food name.
    """
    if not food_name:
        return "Food name could not be identified by the model."

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = f"Provide the common ingredients and a brief summary of the nutritional information (calories, protein, carbs, fats) for a typical serving of {food_name}. Please be concise and use a clear format."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error from OpenAI API: {e}"


# --- Main part of the script ---
if __name__ == "__main__":
    image_file_path = r"C:\Projects\StreamlitTest\pizza-1003.jpg"

    if os.path.exists(image_file_path):
        # Step 1: Use your PyTorch model to get the food name
        food_name = get_food_class_from_model(image_file_path)

        if food_name:
            print(f"Your ML model identified the food as: **{food_name}**")

            # Step 2: Use the identified name to query OpenAI
            info = get_nutritional_info_from_openai(food_name)

            print("\n--- Ingredients and Nutritional Information ---")
            print(info)
        else:
            print("Failed to classify the food. Check your model and image.")
    else:
        print(f"Image file '{image_file_path}' not found. Please verify the path.")