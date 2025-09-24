import streamlit as st
from PIL import Image
import requests
import io

if __name__ == '__main__':

    # Backend API configuration
    BACKEND_URL = "http://localhost:8004"
    CLASSIFY_ENDPOINT = f"{BACKEND_URL}/api/classify"

    def classify_image_via_api(uploaded_file):
        """Send image to backend API for classification."""
        try:
            # Prepare the file for upload
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

            # Make POST request to backend
            response = requests.post(CLASSIFY_ENDPOINT, files=files, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Backend API error: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend: {e}")
            return None

    # --- Set Up the UI ---
    st.title("UFA Calorie Coach - Food Image Classifier")
    st.write("Upload an image to get a class prediction from our backend service.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # --- Prediction Logic ---
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Call backend API for classification
        result = classify_image_via_api(uploaded_file)

        if result and result.get('success'):
            predicted_class = result.get('predicted_class')
            confidence = result.get('confidence')

            # Display the result to the user
            st.success(f"Prediction: **{predicted_class}**")
            st.info(f"Confidence: {confidence:.2f}%")
        else:
            st.error("Failed to classify the image. Please try again or check if the backend service is running.")
            if result:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
