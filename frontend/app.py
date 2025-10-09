import streamlit as st
from PIL import Image
import requests
import pandas as pd
import asyncio
import json
import tempfile
import os
import sys

# Add parent directory to path to import agents
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.nutritionist_agent import search_food_nutrition
from agents.foodImageClassifier_agent import classify_food_image
from dotenv import load_dotenv

load_dotenv()

# Set page config for mobile-responsive layout
st.set_page_config(
    page_title="UFA Calorie Coach",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern glassmorphism UI with dark mode
st.markdown("""
<style>
    /* Modern Dark Theme with Glassmorphism */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    .main > div {
        padding: 1rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Animated gradient background */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px 0 rgba(31, 38, 135, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Header with gradient text */
    .app-header {
        text-align: center;
        padding: 3rem 1rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 300%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent,
            rgba(255, 255, 255, 0.05),
            transparent
        );
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(0); }
        100% { transform: translateX(50%); }
    }
    
    .app-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
        animation: titleGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.3)); }
        to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.5)); }
    }
    
    .app-subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2rem;
        margin-top: 0.8rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    .app-emoji {
        font-size: 4rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Category headers with icons */
    .category-header {
        color: #fff;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0 0 1.5rem 0;
        display: flex;
        align-items: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        position: relative;
    }
    
    .category-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 80px;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    .category-header span {
        margin-right: 0.8rem;
        font-size: 1.6rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    /* Prediction result with animated gradient */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        animation: gradientShift 3s ease infinite;
        color: white;
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .result-card h3 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 400;
        color: rgba(255,255,255,0.9);
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
    }
    
    .result-card h2 {
        margin: 1rem 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .result-card p {
        margin: 0;
        font-size: 1.2rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    /* Info cards with glass effect */
    .info-card {
        background: rgba(103, 126, 234, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid rgba(103, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        background: rgba(103, 126, 234, 0.15);
        border-color: rgba(103, 126, 234, 0.4);
        transform: translateX(5px);
    }
    
    /* Ingredient card */
    .ingredient-card {
        background: rgba(118, 75, 162, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid rgba(118, 75, 162, 0.2);
        transition: all 0.3s ease;
    }
    
    .ingredient-card:hover {
        background: rgba(118, 75, 162, 0.15);
        border-color: rgba(118, 75, 162, 0.4);
        transform: translateX(5px);
    }
    
    /* Nutrient badges with neon glow */
    .nutrient-badge {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        margin: 0.5rem;
        border: 2px solid;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .nutrient-badge::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.1);
        transform: translate(-50%, -50%);
        transition: width 0.4s, height 0.4s;
    }
    
    .nutrient-badge:hover::before {
        width: 200%;
        height: 200%;
    }
    
    .nutrient-badge:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .nutrient-badge h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    .nutrient-badge p {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .nutrient-badge.calories {
        border-color: #ff6b6b;
        color: #ff6b6b;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .nutrient-badge.protein {
        border-color: #4ecdc4;
        color: #4ecdc4;
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
    }
    
    .nutrient-badge.fat {
        border-color: #ffe66d;
        color: #ffe66d;
        box-shadow: 0 4px 15px rgba(255, 230, 109, 0.3);
    }
    
    .nutrient-badge.carbs {
        border-color: #95e1d3;
        color: #95e1d3;
        box-shadow: 0 4px 15px rgba(149, 225, 211, 0.3);
    }
    
    .nutrient-badge.fiber {
        border-color: #c7b3f5;
        color: #c7b3f5;
        box-shadow: 0 4px 15px rgba(199, 179, 245, 0.3);
    }
    
    .nutrient-badge.sugar {
        border-color: #ff9ff3;
        color: #ff9ff3;
        box-shadow: 0 4px 15px rgba(255, 159, 243, 0.3);
    }
    
    /* Upload area with animated border */
    .upload-area {
        background: rgba(255, 255, 255, 0.03);
        border: 3px dashed rgba(102, 126, 234, 0.4);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-area::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 300% 300%;
        border-radius: 20px;
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: -1;
        animation: gradientRotate 4s ease infinite;
    }
    
    @keyframes gradientRotate {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .upload-area:hover::before {
        opacity: 1;
    }
    
    .upload-area:hover {
        border-color: transparent;
        background: rgba(255, 255, 255, 0.05);
        transform: scale(1.02);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Image styling with frame effect */
    .stImage > img {
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        border: 3px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stImage > img:hover {
        transform: scale(1.02);
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
    }
    
    /* Button styling with glow effect */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
        transform: translateY(-3px);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Data tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Text styling */
    .food-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
        margin: 0 0 1rem 0;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .food-info {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        margin: 0.6rem 0;
        display: flex;
        align-items: center;
    }
    
    .food-info strong {
        font-weight: 600;
        margin-right: 0.8rem;
        color: #667eea;
        min-width: 120px;
    }
    
    .ingredient-list {
        font-size: 1rem;
        line-height: 1.8;
        color: rgba(255, 255, 255, 0.85);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    /* Instructions cards with hover effects */
    .instruction-item {
        text-align: center;
        padding: 2rem 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        margin: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .instruction-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .instruction-item:hover::before {
        left: 100%;
    }
    
    .instruction-item:hover {
        transform: translateY(-10px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
    }
    
    .instruction-title {
        color: #fff;
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .instruction-desc {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1rem;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Supported foods styling */
    .food-list {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1rem;
        margin: 0;
        padding-left: 1.5rem;
        line-height: 2;
    }
    
    .food-list li {
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    
    .food-list li:hover {
        color: #667eea;
        transform: translateX(5px);
    }
    
    .food-list strong {
        color: #764ba2;
        font-weight: 600;
    }
    
    /* Warning/Info messages */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .app-title {
            font-size: 2.5rem;
        }
        
        .app-emoji {
            font-size: 3rem;
        }
        
        .glass-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .result-card h2 {
            font-size: 2rem;
        }
        
        .nutrient-badge {
            padding: 1.2rem 0.8rem;
            margin: 0.3rem;
        }
        
        .instruction-item {
            padding: 1.5rem 1rem;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #f093fb);
    }
</style>
""", unsafe_allow_html=True)

# Helper Classes
class NutritionDisplay:
    """Handles display of nutrition data and ingredients."""

    @staticmethod
    def parse_text_nutrition(text_data: str) -> dict:
        """Parse text-based nutrition data into structured format."""
        result = {
            'title': '',
            'serving_size': '',
            'nutrients': {},
            'ingredients': ''
        }

        lines = text_data.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse title
            if line.startswith('**Title**:'):
                result['title'] = line.replace('**Title**:', '').strip()
                current_section = None

            # Parse serving size
            elif line.startswith('**Serving Size**:'):
                result['serving_size'] = line.replace('**Serving Size**:', '').strip()
                current_section = None

            # Parse key nutrients section
            elif line.startswith('**Key Nutrients**:'):
                current_section = 'nutrients'

            # Parse ingredients section header
            elif line.startswith('**Ingredients**:'):
                current_section = 'ingredients'
                # Check if ingredients are on the same line
                ingredients_on_same_line = line.replace('**Ingredients**:', '').strip()
                if ingredients_on_same_line:
                    result['ingredients'] = ingredients_on_same_line

            # Parse nutrient values
            elif current_section == 'nutrients' and line.startswith('-'):
                # Format: "- Energy: 163 kcal"
                nutrient_line = line[1:].strip()  # Remove the dash
                if ':' in nutrient_line:
                    name, value = nutrient_line.split(':', 1)
                    result['nutrients'][name.strip()] = value.strip()

            # Parse ingredients text (if on separate lines)
            elif current_section == 'ingredients' and not line.startswith('**'):
                if result['ingredients']:
                    result['ingredients'] += ' ' + line
                else:
                    result['ingredients'] = line

        return result

    @staticmethod
    def display_food_info_from_text(parsed_data: dict):
        """Display basic food information from parsed text data."""
        with st.container():

            st.markdown('<div class="category-header"><span>📊</span> Food Information</div>', unsafe_allow_html=True)

            title = parsed_data.get('title', 'N/A')
            serving = parsed_data.get('serving_size', 'N/A')

            st.markdown(f"""
            <div class="info-card">
                <div class="food-title">{title}</div>
                <div class="food-info"><strong>Serving Size:</strong> {serving}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def display_ingredients_from_text(parsed_data: dict):
        """Display ingredients information from parsed text data."""
        with st.container():

            st.markdown('<div class="category-header"><span>🥄</span> Ingredients</div>', unsafe_allow_html=True)

            ingredients = parsed_data.get('ingredients', '').strip()

            if ingredients:
                # Split by commas for comma-separated ingredients
                if ',' in ingredients:
                    ingredients_parts = [part.strip() for part in ingredients.split(',') if part.strip()]
                # Fallback to periods if no commas
                elif '.' in ingredients:
                    ingredients_parts = [part.strip() for part in ingredients.split('.') if part.strip()]
                else:
                    ingredients_parts = [ingredients]

                formatted_ingredients = '<br>'.join([f"• {part}" for part in ingredients_parts])

                st.markdown(f"""
                <div class="ingredient-card">
                    <div class="ingredient-list">
                        {formatted_ingredients}
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
    def display_key_nutrients_from_text(parsed_data: dict):
        """Display key nutrients from parsed text data in badge format."""
        with st.container():

            st.markdown('<div class="category-header"><span>🥗</span> Key Nutrients</div>', unsafe_allow_html=True)

            nutrients = parsed_data.get('nutrients', {})

            if nutrients:
                # Map nutrient names to display format
                nutrient_display = {}
                nutrient_classes = {}

                for name, value in nutrients.items():
                    if 'Energy' in name or 'energy' in name.lower():
                        nutrient_display['Calories'] = value
                        nutrient_classes['Calories'] = "calories"
                    elif 'Protein' in name:
                        nutrient_display['Protein'] = value
                        nutrient_classes['Protein'] = "protein"
                    elif 'lipid' in name or 'fat' in name.lower():
                        nutrient_display['Fat'] = value
                        nutrient_classes['Fat'] = "fat"
                    elif 'Carbohydrate' in name:
                        nutrient_display['Carbs'] = value
                        nutrient_classes['Carbs'] = "carbs"
                    elif 'Fiber' in name:
                        nutrient_display['Fiber'] = value
                        nutrient_classes['Fiber'] = "fiber"
                    elif 'Sodium' in name:
                        nutrient_display['Sodium'] = value
                        nutrient_classes['Sodium'] = "sugar"  # Reuse sugar styling

                # Display in columns
                num_nutrients = len(nutrient_display)
                cols = st.columns(min(num_nutrients, 3))

                for i, (nutrient, value) in enumerate(nutrient_display.items()):
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                        <div class="nutrient-badge {nutrient_classes.get(nutrient, '')}">
                            <h4>{nutrient}</h4>
                            <p>{value}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No nutrient information available")

            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def display_complete_nutrition_from_text(parsed_data: dict):
        """Display complete nutrition facts in expandable table from parsed text."""
        nutrients = parsed_data.get('nutrients', {})

        if nutrients:
            with st.expander("📋 Complete Nutrition Facts", expanded=False):
                nutrition_data = []

                for name, value in nutrients.items():
                    nutrition_data.append({
                        'Nutrient': name,
                        'Amount': value
                    })

                df = pd.DataFrame(nutrition_data)
                st.dataframe(df, use_container_width=True, hide_index=True, height=350)

    # Keep original methods for backwards compatibility
    @staticmethod
    def display_food_info(food_item: dict):
        """Display basic food information in a card."""
        with st.container():

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

    def display_nutrition_analysis(self, food_data):
        """Display complete nutrition analysis - handles both text and JSON formats."""
        # Check if it's text-based output
        if isinstance(food_data, str):
            # Parse the text format
            parsed_data = self.parse_text_nutrition(food_data)

            # Display parsed information
            self.display_food_info_from_text(parsed_data)
            self.display_ingredients_from_text(parsed_data)
            self.display_key_nutrients_from_text(parsed_data)
            self.display_complete_nutrition_from_text(parsed_data)

        # Handle JSON format (legacy support)
        elif isinstance(food_data, dict):
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
        else:
            st.warning("⚠️ Invalid nutrition data format.")


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


# Helper Classes
class NutritionDisplay:
    """Handles display of nutrition data and ingredients."""

    @staticmethod
    def parse_text_nutrition(text_data: str) -> dict:
        """Parse text-based nutrition data into structured format."""
        result = {
            'title': '',
            'serving_size': '',
            'nutrients': {},
            'ingredients': ''
        }

        lines = text_data.strip().split('\n')
        current_section = None

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse title
            if line.startswith('**Title**:'):
                result['title'] = line.replace('**Title**:', '').strip()
                current_section = None

            # Parse serving size
            elif line.startswith('**Serving Size**:'):
                result['serving_size'] = line.replace('**Serving Size**:', '').strip()
                current_section = None

            # Parse key nutrients section
            elif line.startswith('**Key Nutrients**:'):
                current_section = 'nutrients'

            # Parse ingredients section header
            elif line.startswith('**Ingredients**:'):
                current_section = 'ingredients'
                # Check if ingredients are on the same line
                ingredients_on_same_line = line.replace('**Ingredients**:', '').strip()
                if ingredients_on_same_line:
                    result['ingredients'] = ingredients_on_same_line

            # Parse nutrient values
            elif current_section == 'nutrients' and line.startswith('-'):
                # Format: "- Energy: 163 kcal"
                nutrient_line = line[1:].strip()  # Remove the dash
                if ':' in nutrient_line:
                    name, value = nutrient_line.split(':', 1)
                    result['nutrients'][name.strip()] = value.strip()

            # Parse ingredients text (if on separate lines)
            elif current_section == 'ingredients' and not line.startswith('**'):
                if result['ingredients']:
                    result['ingredients'] += ' ' + line
                else:
                    result['ingredients'] = line

        return result

    @staticmethod
    def display_food_info_from_text(parsed_data: dict):
        """Display basic food information from parsed text data."""
        with st.container():
            
            st.markdown('<div class="category-header"><span>📊</span> Food Information</div>', unsafe_allow_html=True)

            title = parsed_data.get('title', 'N/A')
            serving = parsed_data.get('serving_size', 'N/A')

            st.markdown(f"""
            <div class="info-card">
                <div class="food-title">{title}</div>
                <div class="food-info"><strong>Serving Size:</strong> {serving}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def display_ingredients_from_text(parsed_data: dict):
        """Display ingredients information from parsed text data."""
        with st.container():
            
            st.markdown('<div class="category-header"><span>🥄</span> Ingredients</div>', unsafe_allow_html=True)

            ingredients = parsed_data.get('ingredients', '').strip()

            if ingredients:
                # Split by commas for comma-separated ingredients
                if ',' in ingredients:
                    ingredients_parts = [part.strip() for part in ingredients.split(',') if part.strip()]
                # Fallback to periods if no commas
                elif '.' in ingredients:
                    ingredients_parts = [part.strip() for part in ingredients.split('.') if part.strip()]
                else:
                    ingredients_parts = [ingredients]

                formatted_ingredients = '<br>'.join([f"• {part}" for part in ingredients_parts])

                st.markdown(f"""
                <div class="ingredient-card">
                    <div class="ingredient-list">
                        {formatted_ingredients}
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
    def display_key_nutrients_from_text(parsed_data: dict):
        """Display key nutrients from parsed text data in badge format."""
        with st.container():
            
            st.markdown('<div class="category-header"><span>🥗</span> Key Nutrients</div>', unsafe_allow_html=True)

            nutrients = parsed_data.get('nutrients', {})

            if nutrients:
                # Map nutrient names to display format
                nutrient_display = {}
                nutrient_classes = {}

                for name, value in nutrients.items():
                    if 'Energy' in name or 'energy' in name.lower():
                        nutrient_display['Calories'] = value
                        nutrient_classes['Calories'] = "calories"
                    elif 'Protein' in name:
                        nutrient_display['Protein'] = value
                        nutrient_classes['Protein'] = "protein"
                    elif 'lipid' in name or 'fat' in name.lower():
                        nutrient_display['Fat'] = value
                        nutrient_classes['Fat'] = "fat"
                    elif 'Carbohydrate' in name:
                        nutrient_display['Carbs'] = value
                        nutrient_classes['Carbs'] = "carbs"
                    elif 'Fiber' in name:
                        nutrient_display['Fiber'] = value
                        nutrient_classes['Fiber'] = "fiber"
                    elif 'Sodium' in name:
                        nutrient_display['Sodium'] = value
                        nutrient_classes['Sodium'] = "sugar"  # Reuse sugar styling

                # Display in columns
                num_nutrients = len(nutrient_display)
                cols = st.columns(min(num_nutrients, 3))

                for i, (nutrient, value) in enumerate(nutrient_display.items()):
                    with cols[i % len(cols)]:
                        st.markdown(f"""
                        <div class="nutrient-badge {nutrient_classes.get(nutrient, '')}">
                            <h4>{nutrient}</h4>
                            <p>{value}</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No nutrient information available")

            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def display_complete_nutrition_from_text(parsed_data: dict):
        """Display complete nutrition facts in expandable table from parsed text."""
        nutrients = parsed_data.get('nutrients', {})

        if nutrients:
            with st.expander("📋 Complete Nutrition Facts", expanded=False):
                nutrition_data = []

                for name, value in nutrients.items():
                    nutrition_data.append({
                        'Nutrient': name,
                        'Amount': value
                    })

                df = pd.DataFrame(nutrition_data)
                st.dataframe(df, use_container_width=True, hide_index=True, height=350)

    # Keep original methods for backwards compatibility
    @staticmethod
    def display_food_info(food_item: dict):
        """Display basic food information in a card."""
        with st.container():
            
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

    def display_nutrition_analysis(self, food_data):
        """Display complete nutrition analysis - handles both text and JSON formats."""
        # Check if it's text-based output
        if isinstance(food_data, str):
            # Parse the text format
            parsed_data = self.parse_text_nutrition(food_data)

            # Display parsed information
            self.display_food_info_from_text(parsed_data)
            self.display_ingredients_from_text(parsed_data)
            self.display_key_nutrients_from_text(parsed_data)
            self.display_complete_nutrition_from_text(parsed_data)

        # Handle JSON format (legacy support)
        elif isinstance(food_data, dict):
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
        else:
            st.warning("⚠️ Invalid nutrition data format.")


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


# Custom CSS for modern glassmorphism UI with dark mode
st.markdown("""
<style>
    /* Modern Dark Theme with Glassmorphism */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    .main > div {
        padding: 1rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Animated gradient background */
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glass Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 45px 0 rgba(31, 38, 135, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    /* Header with gradient text */
    .app-header {
        text-align: center;
        padding: 3rem 1rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 300%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent,
            rgba(255, 255, 255, 0.05),
            transparent
        );
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { transform: translateX(0); }
        100% { transform: translateX(50%); }
    }
    
    .app-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(102, 126, 234, 0.3);
        animation: titleGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 10px rgba(102, 126, 234, 0.3)); }
        to { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 0.5)); }
    }
    
    .app-subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2rem;
        margin-top: 0.8rem;
        font-weight: 300;
        letter-spacing: 1px;
    }
    
    .app-emoji {
        font-size: 4rem;
        display: inline-block;
        animation: float 3s ease-in-out infinite;
        margin-bottom: 1rem;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    /* Category headers with icons */
    .category-header {
        color: #fff;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0 0 1.5rem 0;
        display: flex;
        align-items: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid rgba(102, 126, 234, 0.3);
        position: relative;
    }
    
    .category-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 80px;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 2px;
    }
    
    .category-header span {
        margin-right: 0.8rem;
        font-size: 1.6rem;
        animation: bounce 2s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    /* Prediction result with animated gradient */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-size: 200% 200%;
        animation: gradientShift 3s ease infinite;
        color: white;
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: rotate 10s linear infinite;
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    
    .result-card h3 {
        margin: 0;
        font-size: 1.1rem;
        font-weight: 400;
        color: rgba(255,255,255,0.9);
        text-transform: uppercase;
        letter-spacing: 2px;
        position: relative;
        z-index: 1;
    }
    
    .result-card h2 {
        margin: 1rem 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 4px 8px rgba(0,0,0,0.2);
        position: relative;
        z-index: 1;
    }
    
    .result-card p {
        margin: 0;
        font-size: 1.2rem;
        opacity: 0.95;
        position: relative;
        z-index: 1;
    }
    
    /* Info cards with glass effect */
    .info-card {
        background: rgba(103, 126, 234, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid rgba(103, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        background: rgba(103, 126, 234, 0.15);
        border-color: rgba(103, 126, 234, 0.4);
        transform: translateX(5px);
    }
    
    /* Ingredient card */
    .ingredient-card {
        background: rgba(118, 75, 162, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.8rem;
        margin: 1rem 0;
        border: 1px solid rgba(118, 75, 162, 0.2);
        transition: all 0.3s ease;
    }
    
    .ingredient-card:hover {
        background: rgba(118, 75, 162, 0.15);
        border-color: rgba(118, 75, 162, 0.4);
        transform: translateX(5px);
    }
    
    /* Nutrient badges with neon glow */
    .nutrient-badge {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem 1rem;
        text-align: center;
        margin: 0.5rem;
        border: 2px solid;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .nutrient-badge::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.1);
        transform: translate(-50%, -50%);
        transition: width 0.4s, height 0.4s;
    }
    
    .nutrient-badge:hover::before {
        width: 200%;
        height: 200%;
    }
    
    .nutrient-badge:hover {
        transform: translateY(-8px) scale(1.05);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .nutrient-badge h4 {
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 500;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }
    
    .nutrient-badge p {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        position: relative;
        z-index: 1;
    }
    
    .nutrient-badge.calories {
        border-color: #ff6b6b;
        color: #ff6b6b;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    .nutrient-badge.protein {
        border-color: #4ecdc4;
        color: #4ecdc4;
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.3);
    }
    
    .nutrient-badge.fat {
        border-color: #ffe66d;
        color: #ffe66d;
        box-shadow: 0 4px 15px rgba(255, 230, 109, 0.3);
    }
    
    .nutrient-badge.carbs {
        border-color: #95e1d3;
        color: #95e1d3;
        box-shadow: 0 4px 15px rgba(149, 225, 211, 0.3);
    }
    
    .nutrient-badge.fiber {
        border-color: #c7b3f5;
        color: #c7b3f5;
        box-shadow: 0 4px 15px rgba(199, 179, 245, 0.3);
    }
    
    .nutrient-badge.sugar {
        border-color: #ff9ff3;
        color: #ff9ff3;
        box-shadow: 0 4px 15px rgba(255, 159, 243, 0.3);
    }
    
    /* Upload area with animated border */
    .upload-area {
        background: rgba(255, 255, 255, 0.03);
        border: 3px dashed rgba(102, 126, 234, 0.4);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-area::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #667eea, #764ba2, #f093fb, #667eea);
        background-size: 300% 300%;
        border-radius: 20px;
        opacity: 0;
        transition: opacity 0.3s ease;
        z-index: -1;
        animation: gradientRotate 4s ease infinite;
    }
    
    @keyframes gradientRotate {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .upload-area:hover::before {
        opacity: 1;
    }
    
    .upload-area:hover {
        border-color: transparent;
        background: rgba(255, 255, 255, 0.05);
        transform: scale(1.02);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Image styling with frame effect */
    .stImage > img {
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        border: 3px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stImage > img:hover {
        transform: scale(1.02);
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
    }
    
    /* Button styling with glow effect */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
        font-weight: 600;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.2);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
        transform: translateY(-3px);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        padding: 1rem;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Data tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Text styling */
    .food-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
        margin: 0 0 1rem 0;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .food-info {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.8);
        margin: 0.6rem 0;
        display: flex;
        align-items: center;
    }
    
    .food-info strong {
        font-weight: 600;
        margin-right: 0.8rem;
        color: #667eea;
        min-width: 120px;
    }
    
    .ingredient-list {
        font-size: 1rem;
        line-height: 1.8;
        color: rgba(255, 255, 255, 0.85);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem;
        font-weight: 600;
        color: #fff;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.4);
    }
    
    /* Instructions cards with hover effects */
    .instruction-item {
        text-align: center;
        padding: 2rem 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        margin: 0.8rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .instruction-item::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .instruction-item:hover::before {
        left: 100%;
    }
    
    .instruction-item:hover {
        transform: translateY(-10px);
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(102, 126, 234, 0.4);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.2);
    }
    
    .instruction-title {
        color: #fff;
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
        font-weight: 600;
    }
    
    .instruction-desc {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1rem;
        margin: 0;
        line-height: 1.6;
    }
    
    /* Supported foods styling */
    .food-list {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1rem;
        margin: 0;
        padding-left: 1.5rem;
        line-height: 2;
    }
    
    .food-list li {
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    
    .food-list li:hover {
        color: #667eea;
        transform: translateX(5px);
    }
    
    .food-list strong {
        color: #764ba2;
        font-weight: 600;
    }
    
    /* Warning/Info messages */
    .stAlert {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .app-title {
            font-size: 2.5rem;
        }
        
        .app-emoji {
            font-size: 3rem;
        }
        
        .glass-card {
            padding: 1.5rem;
            margin: 1rem 0;
        }
        
        .result-card h2 {
            font-size: 2rem;
        }
        
        .nutrient-badge {
            padding: 1.2rem 0.8rem;
            margin: 0.3rem;
        }
        
        .instruction-item {
            padding: 1.5rem 1rem;
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2, #f093fb);
    }
</style>
""", unsafe_allow_html=True)

if __name__ == '__main__':
    # Initialize helper classes
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
            
            st.markdown('<div class="category-header"><span>🖼️</span> Your Image</div>', unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Food Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Display AI analysis
        with col2:
            
            st.markdown('<div class="category-header"><span>🤖</span> AI Analysis</div>', unsafe_allow_html=True)

            # Save uploaded file to temp path for agent
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                image_path = tmp_file.name

            try:
                with st.spinner('🔍 Analyzing your food image...'):
                    result_str = asyncio.run(classify_food_image(image_path))
                    try:
                        if isinstance(result_str, str):
                            result = json.loads(result_str)
                        else:
                            result = None
                    except (json.JSONDecodeError, TypeError):
                        result = None
            finally:
                os.unlink(image_path)

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

            food_data_str = asyncio.run(search_food_nutrition(search_term))
            print(food_data_str)
            # The agent returns formatted text, not JSON - pass it directly
            if food_data_str and isinstance(food_data_str, str):
                # Check if it's a text response (starts with **Title** or similar)
                if '**Title**' in food_data_str or '**Serving Size**' in food_data_str:
                    food_data = food_data_str  # Use text directly
                else:
                    # Try to parse as JSON (legacy format)
                    try:
                        food_data = json.loads(food_data_str)
                    except (json.JSONDecodeError, TypeError):
                        food_data = food_data_str  # Fall back to text
            else:
                food_data = food_data_str

        if food_data:
            st.markdown('<div class="category-header" style="font-size: 1.3rem; justify-content: center;"><span>📊</span> Nutrition Analysis</div>', unsafe_allow_html=True)

            nutrition_display.display_nutrition_analysis(food_data)
        else:
            st.warning(f"⚠️ Could not find nutrition information for '{search_term}'.")

            # Manual search section
            with st.container():
                
                st.markdown('<div class="category-header"><span>🔍</span> Manual Search</div>', unsafe_allow_html=True)

                manual_search = st.text_input(
                    "Search for nutrition data:",
                    placeholder="e.g., apple, chicken, rice",
                    help="Enter a food name to search in the USDA database"
                )

                if manual_search:
                    with st.spinner('🔍 Searching...'):
                        manual_food_data_str = asyncio.run(search_food_nutrition(manual_search))

                        # Handle text response
                        if manual_food_data_str and isinstance(manual_food_data_str, str):
                            if '**Title**' in manual_food_data_str or '**Serving Size**' in manual_food_data_str:
                                manual_food_data = manual_food_data_str
                            else:
                                try:
                                    manual_food_data = json.loads(manual_food_data_str)
                                except (json.JSONDecodeError, TypeError):
                                    manual_food_data = manual_food_data_str
                        else:
                            manual_food_data = manual_food_data_str

                    if manual_food_data:
                        nutrition_display.display_nutrition_analysis(manual_food_data)

                st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Display welcome screen
        ui_components.render_welcome_screen()
