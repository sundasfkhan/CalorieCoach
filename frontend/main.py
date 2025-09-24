import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import io

if __name__ == '__main__':

    # Define the class labels based on your training
    class_names = ['apple_pie',
                   'baked_potato',
                   'burger',
                   'butter_naan',
                   'chai',
                   'chapati',
                   'cheesecake',
                   'chicken_curry',
                   'chole_bhature',
                   'crispy_chicken',
                   'dal_makhani',
                   'dhokla',
                   'donut',
                   'fried_rice',
                   'fries',
                   'hot_dog',
                   'ice_cream',
                   'idli',
                   'jalebi',
                   'kaathi_rolls',
                   'kadai_paneer',
                   'kulfi',
                   'masala_dosa',
                   'momos',
                   'omelette',
                   'paani_puri',
                   'pakode',
                   'pav_bhaji',
                   'pizza',
                   'samosa',
                   'sandwich',
                   'sushi',
                   'taco',
                   'taquito']

    # --- 1. Load the Model ---
    # This function caches the model so it only loads once
    @st.cache_resource
    def load_model():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = "./../models/model_efficientnet_v2_m_1.pth"
        model = torchvision.models.efficientnet_v2_m(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
        # You MUST change the last number to match your number of classes
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model


    model = load_model()

    # --- 2. Set Up the UI ---
    st.title("UFA Calorie Coach - Food Image Classifier")
    st.write("Upload an image to get a class prediction.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])



    # --- 3. Prediction Logic ---
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Pre-process the image with the same transformations as validation data
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Process the image and unsqueeze to add a batch dimension
        image_tensor = preprocess(image).unsqueeze(0)

        # Make a prediction
        with torch.no_grad():
            output = model(image_tensor)

        # Get the predicted class
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = class_names[predicted_class_index]
        confidence = probabilities[0][predicted_class_index].item() * 100

        # Display the result to the user
        st.success(f"Prediction: **{predicted_class_name}**")
        st.info(f"Confidence: {confidence:.2f}%")

