import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd

# Set page config for mobile-responsive layout
st.set_page_config(
    page_title="UFA Calorie Coach",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper Classes
class APIClient:
    """Handles all API communications with the backend."""

    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.classify_endpoint = f"{backend_url}/api/classify"
        self.search_endpoint = f"{backend_url}/api/search"

    def classify_image(self, uploaded_file):
        """Send image to backend API for classification."""
        try:
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(self.classify_endpoint, files=files, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Backend API error: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend: {e}")
            return None

    def search_food_data(self, food_name: str):
        """Search for food data using the USDA API via backend."""
        try:
            params = {'food_name': food_name}
            response = requests.get(self.search_endpoint, params=params, timeout=30)

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Food search API error: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to search food data: {e}")
            return None


class NutritionDisplay:
    """Handles display of nutrition data and ingredients."""

    @staticmethod
    def display_food_info(food_item: dict):
        """Display basic food information in a card."""
        with st.container():
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="category-header"><span>📊</span> Food Information</div>', unsafe_allow_html=True)

            description = food_item.get('description', 'N/A')
            brand = food_item.get('brandName', 'Generic')
            serving = f"{food_item.get('servingSize', 'N/A')} {food_item.get('servingSizeUnit', '').lower()}"
            category = food_item.get('foodCategory', 'N/A')

            st.markdown(f"""
            <div class="info-card">
                <div class="food-title">{description}</div>
                <div class="food-info"><strong>Brand:</strong> {brand}</div>
                <div class="food-info"><strong>Serving:</strong> {serving}</div>
                <div class="food-info"><strong>Category:</strong> {category}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def display_ingredients(food_item: dict):
        """Display ingredients information in a card."""
        with st.container():
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="category-header"><span>🥄</span> Ingredients</div>', unsafe_allow_html=True)

            ingredients = food_item.get('ingredients', '')

            if ingredients:
                ingredients_list = [ingredient.strip() for ingredient in ingredients.split(',')]
                formatted_ingredients = '\n'.join([f"• {ingredient}" for ingredient in ingredients_list if ingredient])

                st.markdown(f"""
                <div class="ingredient-card">
                    <div class="ingredient-list">
                        {formatted_ingredients.replace(chr(10), '<br>')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="ingredient-card">
                    <p style="color: #888; margin: 0; font-size: 0.9rem;">No ingredients information available for this food item.</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def display_key_nutrients(nutrients: list):
        """Display key nutrients in badge format."""
        with st.container():
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="category-header"><span>🥗</span> Key Nutrients</div>', unsafe_allow_html=True)

            key_nutrients = {}
            nutrient_classes = {}

            for nutrient in nutrients:
                name = nutrient.get('nutrientName', '')
                value = nutrient.get('value', 0)
                unit = nutrient.get('unitName', '').lower()

                if 'Energy' in name:
                    key_nutrients['Calories'] = f"{value} {unit}"
                    nutrient_classes['Calories'] = "calories"
                elif 'Protein' in name:
                    key_nutrients['Protein'] = f"{value} {unit}"
                    nutrient_classes['Protein'] = "protein"
                elif 'Total lipid (fat)' in name:
                    key_nutrients['Fat'] = f"{value} {unit}"
                    nutrient_classes['Fat'] = "fat"
                elif 'Carbohydrate, by difference' in name:
                    key_nutrients['Carbs'] = f"{value} {unit}"
                    nutrient_classes['Carbs'] = "carbs"
                elif 'Fiber, total dietary' in name:
                    key_nutrients['Fiber'] = f"{value} {unit}"
                    nutrient_classes['Fiber'] = "fiber"
                elif 'Total Sugars' in name:
                    key_nutrients['Sugar'] = f"{value} {unit}"
                    nutrient_classes['Sugar'] = "sugar"

            if key_nutrients:
                cols = st.columns(min(len(key_nutrients), 3))
                for i, (nutrient, value) in enumerate(key_nutrients.items()):
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                        <div class="nutrient-badge {nutrient_classes.get(nutrient, '')}">
                            <h4>{nutrient}</h4>
                            <p>{value}</p>
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def display_complete_nutrition_table(nutrients: list):
        """Display complete nutrition facts in expandable table."""
        with st.expander("📋 Complete Nutrition Facts", expanded=False):
            nutrition_data = []

            for nutrient in nutrients:
                nutrition_data.append({
                    'Nutrient': nutrient.get('nutrientName', 'N/A'),
                    'Amount': f"{nutrient.get('value', 0)} {nutrient.get('unitName', '').lower()}",
                    'Daily Value (%)': f"{nutrient.get('percentDailyValue', 'N/A')}%" if nutrient.get('percentDailyValue') is not None else '-'
                })

            df = pd.DataFrame(nutrition_data)
            st.dataframe(df, use_container_width=True, hide_index=True, height=350)

    def display_nutrition_analysis(self, food_data: dict):
        """Display complete nutrition analysis including food info, ingredients, and nutrients."""
        if not food_data or not food_data.get('foods'):
            st.warning("⚠️ No nutrition data available for this food item.")
            return

        food_item = food_data['foods'][0]

        # Display food information
        self.display_food_info(food_item)

        # Display ingredients
        self.display_ingredients(food_item)

        # Display nutrients
        nutrients = food_item.get('foodNutrients', [])
        if nutrients:
            self.display_key_nutrients(nutrients)
            self.display_complete_nutrition_table(nutrients)
        else:
            st.warning("No detailed nutrition information available for this food item.")


class UIComponents:
    """Handles UI component rendering."""

    @staticmethod
    def render_app_header():
        """Render the main app header."""
        st.markdown("""
        <div class="app-header">
            <h1 class="app-title">🍽️ UFA Calorie Coach</h1>
            <p class="app-subtitle">AI-Powered Food Classification & Nutrition Analysis</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_upload_section():
        """Render the file upload section."""
        with st.container():
            st.markdown("""
            <div class="upload-area">
                <div class="category-header" style="justify-content: center;"><span>📸</span> Upload Your Food Image</div>
            </div>
            """, unsafe_allow_html=True)

            return st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of your food for best results"
            )

    @staticmethod
    def render_prediction_result(predicted_class: str, confidence: float):
        """Render the AI prediction result."""
        st.markdown(f"""
        <div class="result-card">
            <h3>Prediction Result</h3>
            <h2>{predicted_class.title().replace('_', ' ')}</h2>
            <p>Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence / 100)

    @staticmethod
    def render_welcome_screen():
        """Render the welcome screen with instructions."""
        with st.container():
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="category-header"><span>💡</span> How to Use</div>', unsafe_allow_html=True)

            cols = st.columns(3)

            instructions = [
                {"step": "1️⃣ Upload Image", "desc": "Take a clear photo of your food item"},
                {"step": "2️⃣ Get AI Prediction", "desc": "Our AI will identify your food"},
                {"step": "3️⃣ View Nutrition", "desc": "Get detailed nutritional information"}
            ]

            for i, col in enumerate(cols):
                with col:
                    st.markdown(f"""
                    <div class="instruction-item">
                        <h4 class="instruction-title">{instructions[i]['step']}</h4>
                        <p class="instruction-desc">{instructions[i]['desc']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # Sample foods section
        with st.container():
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="category-header"><span>🍎</span> Supported Foods</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <ul class="food-list">
                    <li><strong>Fruits & Desserts:</strong> Apple pie, ice cream</li>
                    <li><strong>Main Dishes:</strong> Pizza, burger, sushi, tacos</li>
                    <li><strong>Snacks:</strong> Fries, donuts, momos</li>
                </ul>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <ul class="food-list">
                    <li><strong>Indian Cuisine:</strong> Samosa, naan, curry</li>
                    <li><strong>Breakfast:</strong> Omelette, sandwich</li>
                    <li><strong>And many more!</strong></li>
                </ul>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)


# Custom CSS for modern green and blue UI
st.markdown("""
<style>
    /* Modern green and blue theme */
    .stApp {
        background: linear-gradient(135deg, #e8f5e8 0%, #e3f2fd 100%);
    }
    
    .main > div {
        padding: 0.5rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header styles */
    .app-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #2e7d32, #1976d2);
        color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
    }
    
    .app-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .app-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Card styles with green/blue theme */
    .modern-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid rgba(46, 125, 50, 0.1);
        transition: all 0.3s ease;
    }
    
    .modern-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(25, 118, 210, 0.15);
        border-color: rgba(25, 118, 210, 0.2);
    }
    
    /* Category headers */
    .category-header {
        color: #2e7d32;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        border-bottom: 2px solid #e8f5e8;
        padding-bottom: 0.5rem;
    }
    
    .category-header span {
        margin-right: 0.5rem;
        font-size: 1.2rem;
    }
    
    /* Prediction result card */
    .result-card {
        background: linear-gradient(135deg, #4caf50, #2196f3);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
    }
    
    .result-card h3 {
        margin: 0;
        font-size: 1rem;
        font-weight: 500;
        color: rgba(255,255,255,0.9);
    }
    
    .result-card h2 {
        margin: 0.5rem 0;
        font-size: 1.6rem;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .result-card p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.95;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #f1f8e9, #e8f5e8);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    
    /* Ingredient card */
    .ingredient-card {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #2196f3;
    }
    
    /* Nutrient badges with green/blue theme */
    .nutrient-badge {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.4rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .nutrient-badge:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .nutrient-badge h4 {
        margin: 0;
        font-size: 0.85rem;
        font-weight: 600;
        color: #666;
    }
    
    .nutrient-badge p {
        margin: 0.3rem 0 0 0;
        font-size: 1.2rem;
        font-weight: 700;
        color: #333;
    }
    
    .nutrient-badge.calories {
        border-color: #ff7043;
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
    }
    
    .nutrient-badge.protein {
        border-color: #42a5f5;
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
    }
    
    .nutrient-badge.fat {
        border-color: #ffa726;
        background: linear-gradient(135deg, #fff8e1, #ffecb3);
    }
    
    .nutrient-badge.carbs {
        border-color: #66bb6a;
        background: linear-gradient(135deg, #e8f5e8, #c8e6c9);
    }
    
    .nutrient-badge.fiber {
        border-color: #ab47bc;
        background: linear-gradient(135deg, #f3e5f5, #e1bee7);
    }
    
    .nutrient-badge.sugar {
        border-color: #ec407a;
        background: linear-gradient(135deg, #fce4ec, #f8bbd9);
    }
    
    /* Upload area */
    .upload-area {
        background: linear-gradient(135deg, #e8f5e8, #e3f2fd);
        border: 2px dashed #4caf50;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        border-color: #2196f3;
        background: linear-gradient(135deg, #e3f2fd, #e8f5e8);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4caf50, #2196f3);
        border-radius: 6px;
    }
    
    /* Image styling */
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 2px solid rgba(76, 175, 80, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4caf50, #2196f3);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.3);
        transform: translateY(-2px);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Data tables */
    .stDataFrame {
        font-size: 0.9rem;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Divider */
    .modern-divider {
        height: 2px;
        background: linear-gradient(90deg, #4caf50, #2196f3);
        margin: 2rem 0;
        width: 100%;
        border-radius: 1px;
    }
    
    /* Text styling */
    .food-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #2e7d32;
        margin: 0 0 0.8rem 0;
    }
    
    .food-info {
        font-size: 0.95rem;
        color: #555;
        margin: 0.4rem 0;
        display: flex;
        align-items: center;
    }
    
    .food-info strong {
        font-weight: 600;
        margin-right: 0.5rem;
        color: #1976d2;
    }
    
    .ingredient-list {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #444;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1rem;
        font-weight: 600;
        color: #2e7d32;
        background: rgba(232, 245, 232, 0.5);
        border-radius: 8px;
    }
    
    /* Instructions styling */
    .instruction-item {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, rgba(232, 245, 232, 0.3), rgba(227, 242, 253, 0.3));
        border-radius: 8px;
        margin: 0.5rem;
    }
    
    .instruction-title {
        color: #2e7d32;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .instruction-desc {
        color: #666;
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Supported foods styling */
    .food-list {
        color: #555;
        font-size: 0.95rem;
        margin: 0;
        padding-left: 1.5rem;
        line-height: 1.6;
    }
    
    .food-list li {
        margin-bottom: 0.5rem;
    }
    
    .food-list strong {
        color: #2e7d32;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .app-title {
            font-size: 1.6rem;
        }
        
        .modern-card {
            padding: 1rem;
            margin: 0.7rem 0;
        }
        
        .result-card h2 {
            font-size: 1.4rem;
        }
        
        .nutrient-badge {
            padding: 0.8rem;
            margin: 0.2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

if __name__ == '__main__':
    # Initialize helper classes
    api_client = APIClient("http://localhost:8004")
    nutrition_display = NutritionDisplay()
    ui_components = UIComponents()

    # Render app header
    ui_components.render_app_header()

    # Render upload section
    uploaded_file = ui_components.render_upload_section()

    # Main application logic
    if uploaded_file is not None:
        # Layout with two columns
        col1, col2 = st.columns([1, 1])

        # Display uploaded image
        with col1:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="category-header"><span>🖼️</span> Your Image</div>', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Food Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Display AI analysis
        with col2:
            st.markdown('<div class="modern-card">', unsafe_allow_html=True)
            st.markdown('<div class="category-header"><span>🤖</span> AI Analysis</div>', unsafe_allow_html=True)

            with st.spinner('🔍 Analyzing your food image...'):
                result = api_client.classify_image(uploaded_file)

            if result and result.get('success'):
                predicted_class = result.get('predicted_class')
                confidence = result.get('confidence')

                ui_components.render_prediction_result(predicted_class, confidence)
            else:
                st.error("❌ Failed to classify the image. Please try again.")
                if result:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
                st.stop()

            st.markdown('</div>', unsafe_allow_html=True)

        # Divider
        st.markdown('<div class="modern-divider"></div>', unsafe_allow_html=True)

        # Search for and display nutrition data
        with st.spinner('🔍 Fetching nutrition information...'):
            search_term = predicted_class.replace('_', ' ')
            food_data = api_client.search_food_data(search_term)

        if food_data:
            st.markdown('<div class="category-header" style="font-size: 1.3rem; justify-content: center;"><span>📊</span> Nutrition Analysis</div>', unsafe_allow_html=True)
            nutrition_display.display_nutrition_analysis(food_data)
        else:
            st.warning(f"⚠️ Could not find nutrition information for '{search_term}'.")

            # Manual search section
            with st.container():
                st.markdown('<div class="modern-card">', unsafe_allow_html=True)
                st.markdown('<div class="category-header"><span>🔍</span> Manual Search</div>', unsafe_allow_html=True)

                manual_search = st.text_input(
                    "Search for nutrition data:",
                    placeholder="e.g., apple, chicken, rice",
                    help="Enter a food name to search in the USDA database"
                )

                if manual_search:
                    with st.spinner('🔍 Searching...'):
                        manual_food_data = api_client.search_food_data(manual_search)
                    if manual_food_data:
                        nutrition_display.display_nutrition_analysis(manual_food_data)

                st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display welcome screen
        ui_components.render_welcome_screen()
