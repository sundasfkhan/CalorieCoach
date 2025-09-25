import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd

if __name__ == '__main__':

    # Backend API configuration
    BACKEND_URL = "http://localhost:8004"
    CLASSIFY_ENDPOINT = f"{BACKEND_URL}/api/classify"
    SEARCH_ENDPOINT = f"{BACKEND_URL}/api/search"

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

    def search_food_data(food_name):
        """Search for food data using the USDA API via backend."""
        try:
            params = {'food_name': food_name}
            response = requests.get(SEARCH_ENDPOINT, params=params, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Food search API error: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to search food data: {e}")
            return None

    def display_nutrition_table(food_data):
        """Display nutrition data in a formatted table."""
        if not food_data or not food_data.get('foods'):
            st.warning("No nutrition data available for this food item.")
            return

        food_item = food_data['foods'][0]  # Get the first result

        # Display basic food information
        st.subheader("📊 Nutrition Information")
        st.write(f"**Food:** {food_item.get('description', 'N/A')}")
        st.write(f"**Brand:** {food_item.get('brandName', 'N/A')}")
        st.write(f"**Serving Size:** {food_item.get('servingSize', 'N/A')} {food_item.get('servingSizeUnit', '').lower()}")

        # Extract and format nutrition data
        nutrients = food_item.get('foodNutrients', [])

        if nutrients:
            # Create a list to store nutrition data
            nutrition_data = []

            for nutrient in nutrients:
                nutrition_data.append({
                    'Nutrient': nutrient.get('nutrientName', 'N/A'),
                    'Amount': f"{nutrient.get('value', 0)} {nutrient.get('unitName', '').lower()}",
                    'Daily Value (%)': f"{nutrient.get('percentDailyValue', 'N/A')}%" if nutrient.get('percentDailyValue') is not None else 'N/A'
                })

            # Create DataFrame and display as table
            df = pd.DataFrame(nutrition_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Display key nutrients in a more prominent way
            key_nutrients = {}
            for nutrient in nutrients:
                name = nutrient.get('nutrientName', '')
                if 'Energy' in name:
                    key_nutrients['Calories'] = f"{nutrient.get('value', 0)} {nutrient.get('unitName', '').lower()}"
                elif 'Protein' in name:
                    key_nutrients['Protein'] = f"{nutrient.get('value', 0)} {nutrient.get('unitName', '').lower()}"
                elif 'Total lipid (fat)' in name:
                    key_nutrients['Fat'] = f"{nutrient.get('value', 0)} {nutrient.get('unitName', '').lower()}"
                elif 'Carbohydrate, by difference' in name:
                    key_nutrients['Carbs'] = f"{nutrient.get('value', 0)} {nutrient.get('unitName', '').lower()}"

            if key_nutrients:
                st.subheader("🥗 Key Nutrients")
                cols = st.columns(len(key_nutrients))
                for i, (nutrient, value) in enumerate(key_nutrients.items()):
                    with cols[i]:
                        st.metric(nutrient, value)
        else:
            st.warning("No detailed nutrition information available for this food item.")

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

            # Search for nutrition data based on predicted class
            st.write("---")
            st.write("🔍 Searching for nutrition information...")

            # Convert predicted class to a more searchable format
            search_term = predicted_class.replace('_', ' ')
            food_data = search_food_data(search_term)

            if food_data:
                display_nutrition_table(food_data)
            else:
                st.warning(f"Could not find nutrition information for '{search_term}'. Try searching for a similar food item.")

        else:
            st.error("Failed to classify the image. Please try again or check if the backend service is running.")
            if result:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
