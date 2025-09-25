#!/usr/bin/env python3
"""
Calorie Coach - FastAPI Implementation
A comprehensive API for food classification and USDA food data access
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Depends
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add tools directory to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.food_summary import food_summary

# Load environment variables
load_dotenv()

# Configuration
USDA_API_KEY = os.getenv("USDA_API_KEY")
BASE_URL = "https://api.nal.usda.gov/fdc/v1"
PORT = 8004

# Food classification labels
CLASS_NAMES = [
    'apple_pie', 'baked_potato', 'burger', 'butter_naan', 'chai', 'chapati',
    'cheesecake', 'chicken_curry', 'chole_bhature', 'crispy_chicken', 'dal_makhani',
    'dhokla', 'donut', 'fried_rice', 'fries', 'hot_dog', 'ice_cream', 'idli',
    'jalebi', 'kaathi_rolls', 'kadai_paneer', 'kulfi', 'masala_dosa', 'momos',
    'omelette', 'paani_puri', 'pakode', 'pav_bhaji', 'pizza', 'samosa',
    'sandwich', 'sushi', 'taco', 'taquito'
]

# Global model cache
_model = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    service: str

class ClassificationResponse(BaseModel):
    predicted_class: str
    confidence: float
    success: bool

class ErrorResponse(BaseModel):
    error: str

class SearchResponse(BaseModel):
    totalHits: int
    currentPage: int
    totalPages: int
    foods: list

# FastAPI app initialization
app = FastAPI(
    title="Calorie Coach API",
    description="A comprehensive API for food classification and USDA food data access",
    version="1.0.0",
    contact={
        "name": "API Support",
        "email": "support@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model loading and preprocessing functions
def load_classification_model():
    """Load the EfficientNet model for food classification."""
    global _model
    if _model is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = project_root / "models" / "model_efficientnet_v2_m_1.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        model = torchvision.models.efficientnet_v2_m(pretrained=False)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(CLASS_NAMES))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        _model = model
        logger.info("Food classification model loaded successfully")
    return _model

def preprocess_image(image_file):
    """Preprocess uploaded image for model prediction."""
    image = Image.open(image_file).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return preprocess(image).unsqueeze(0)

async def make_usda_request(endpoint: str, params: Dict[str, Any]) -> dict:
    """Make request to USDA API with error handling."""
    if not USDA_API_KEY:
        raise HTTPException(status_code=500, detail="USDA_API_KEY not configured")

    params['api_key'] = USDA_API_KEY

    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"USDA API error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch data from USDA API")

# Routes

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        service="Calorie Coach FastAPI"
    )

@app.get("/api/search", response_model=SearchResponse, tags=["Food Search"])
async def search_foods(
    food_name: str = Query(..., description="Name of food to search for")
):
    """
    Search foods in the USDA FoodData Central database.

    Returns the first result matching the food name query.
    """
    try:
        params = {
            'query': food_name,
            'pageSize': 1,
            'pageNumber': 1,
            'sortBy': 'dataType.keyword',
            'sortOrder': 'asc'
        }

        result = await make_usda_request('foods/search', params)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.post("/api/classify", response_model=ClassificationResponse, tags=["Food Classification"])
async def classify_food_image(
    file: UploadFile = File(..., description="Image file to classify (JPG, JPEG, PNG)")
):
    """
    Classify a food image using machine learning.

    Accepts an image file and returns the predicted food class with confidence score.

    **Supported Food Categories:**
    apple_pie, baked_potato, burger, butter_naan, chai, chapati, cheesecake,
    chicken_curry, chole_bhature, crispy_chicken, dal_makhani, dhokla, donut,
    fried_rice, fries, hot_dog, ice_cream, idli, jalebi, kaathi_rolls, kadai_paneer,
    kulfi, masala_dosa, momos, omelette, paani_puri, pakode, pav_bhaji, pizza,
    samosa, sandwich, sushi, taco, taquito
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Load model
        model = load_classification_model()

        # Preprocess image
        image_tensor = preprocess_image(file.file)

        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)

        # Get predicted class
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class_index = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = probabilities[0][predicted_class_index].item() * 100

        return ClassificationResponse(
            predicted_class=predicted_class_name,
            confidence=round(confidence, 2),
            success=True
        )

    except FileNotFoundError:
        logger.error("Model file not found")
        raise HTTPException(status_code=500, detail="Classification model not available")
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    print('Calorie Coach FastAPI server starting...')
    print('Available endpoints:')
    print('  GET / - Redirects to API documentation')
    print('  GET /health - Health check')
    print('  GET /api/search - Search foods')
    print('  POST /api/classify - Classify food image')
    print('  GET /docs - Interactive API documentation')
    print('  GET /openapi.json - OpenAPI specification')

    uvicorn.run("app:app", host="127.0.0.1", port=PORT, reload=True)
