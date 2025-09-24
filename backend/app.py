#!/usr/bin/env python3
"""
Calorie Coach - Python Flask Implementation
A comprehensive API for accessing USDA's FoodData Central database

Contracts overview:
- Base URL: http://<host>:8004
- OpenAPI docs: GET /docs/
- Schema JSON: GET /openapi.json

Common error modes:
- 400 when required query params are missing or invalid
- 404 when resource not found (single food endpoint)
- 500 on upstream/network errors or unexpected failures

Notes:
- All endpoints are read-only (GET) and proxy/shape data from USDA FDC API.
- Requests time out after 30s when talking to USDA API.
"""

import os
import logging
from typing import Dict, Any
from flask import Flask, jsonify, request, redirect
from flask_restx import Api, Resource, fields, Namespace
import requests
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import io
import sys
from pathlib import Path

# Add tools directory to path for food_summary import
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tools.food_summary import food_summary

# Load environment variables
load_dotenv()

# Configuration
USDA_API_KEY = os.getenv("USDA_API_KEY")
BASE_URL = "https://api.nal.usda.gov/fdc/v1"
PORT = 8004

# Define the class labels for food classification
CLASS_NAMES = ['apple_pie',
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

# Global variable to store the loaded model
_model = None

def load_classification_model():
    """Load the EfficientNet model for food classification."""
    global _model
    if _model is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = os.path.join(project_root, "models", "model_efficientnet_v2_m_1.pth")

        if not os.path.exists(model_path):
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
    """Preprocess the uploaded image for model prediction."""
    image = Image.open(image_file).convert('RGB')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return preprocess(image).unsqueeze(0)

# Initialize Flask app
app = Flask(__name__)
app.config['RESTX_MASK_SWAGGER'] = False

# Initialize Flask-RESTX for OpenAPI documentation
api = Api(
    app,
    version='1.0.0',
    title='Calorie Coach API',
    description='A comprehensive API for accessing USDA\'s FoodData Central database',
    contact_email='support@example.com',
    license='MIT',
    license_url='https://opensource.org/licenses/MIT',
    doc='/docs/',
)

# Define namespaces
health_ns = Namespace('health', description='Health check operations')
search_ns = Namespace('search', description='Food search operations')
classify_ns = Namespace('classify', description='Food classification operations')

api.add_namespace(health_ns)
api.add_namespace(search_ns, path='/api')
api.add_namespace(classify_ns, path='/api')

# Data models for OpenAPI documentation
nutrient_model = api.model('Nutrient', {
    'nutrientId': fields.Integer(description='Nutrient ID', example=203),
    'nutrientName': fields.String(description='Nutrient name', example='Protein'),
    'value': fields.Float(description='Nutrient value', example=0.26),
    'unitName': fields.String(description='Unit of measurement', example='G')
})

food_model = api.model('Food', {
    'fdcId': fields.Integer(description='Food Data Central ID', example=2344719),
    'description': fields.String(description='Food description', example='Apple, raw'),
    'dataType': fields.String(description='Type of food data', example='Foundation'),
    'brandOwner': fields.String(description='Brand owner (for branded foods)', example='Generic'),
    'foodNutrients': fields.List(fields.Nested(nutrient_model), description='Nutritional information')
})

search_result_model = api.model('SearchResult', {
    'totalHits': fields.Integer(description='Total number of search results', example=1250),
    'currentPage': fields.Integer(description='Current page number', example=1),
    'totalPages': fields.Integer(description='Total number of pages', example=25),
    'foods': fields.List(fields.Nested(food_model), description='Array of food items')
})

health_status_model = api.model('HealthStatus', {
    'status': fields.String(description='Health status', example='ok'),
    'service': fields.String(description='Service name', example='Food Data Central Python API')
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message', example='Query parameter is required')
})

classification_result_model = api.model('ClassificationResult', {
    'predicted_class': fields.String(description='Predicted food class', example='pizza'),
    'confidence': fields.Float(description='Confidence percentage', example=95.67),
    'success': fields.Boolean(description='Classification success status', example=True)
})

classification_with_nutrition_model = api.model('ClassificationWithNutrition', {
    'predicted_class': fields.String(description='Predicted food class', example='pizza'),
    'confidence': fields.Float(description='Confidence percentage', example=95.67),
    'nutrition_data': fields.Raw(description='Nutritional information from food_summary'),
    'success': fields.Boolean(description='Classification success status', example=True)
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add redirect route for root URL
@app.route('/')
def index():
    """Redirect root URL to API documentation."""
    return redirect('/docs')

# Utility functions
def make_usda_request(endpoint: str, params: Dict[str, Any]) -> requests.Response:
    """Make a request to the USDA API with proper error handling.

    Inputs:
    - endpoint: relative USDA path, e.g. 'foods/search', 'food/{fdc_id}', 'foods'
    - params: dict of query params to forward; 'api_key' is injected here

    Behavior:
    - Issues a GET with ?api_key and provided params to BASE_URL/endpoint
    - Uses a 30s timeout
    - Raises for non-2xx responses (requests.exceptions.RequestException)

    Outputs:
    - requests.Response on success (caller should .json())

    Error modes:
    - Network/timeout: logs and re-raises RequestException
    - 4xx/5xx from USDA: response.raise_for_status() triggers HTTPError

    Example:
    - make_usda_request('foods/search', {'query': 'apple', 'pageSize': 10})
    """
    if not USDA_API_KEY:
        logger.error("USDA_API_KEY is not set. Please set the environment variable USDA_API_KEY or add it to a .env file.")
        raise requests.exceptions.HTTPError("USDA_API_KEY is not set")
    params['api_key'] = USDA_API_KEY
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Error making USDA API request: {e}")
        raise

# Health check endpoint
@health_ns.route('')
class HealthCheck(Resource):
    @health_ns.doc('health_check')
    @health_ns.marshal_with(health_status_model)
    def get(self):
        """Health check endpoint.

        Inputs: none
        Outputs: JSON object with fields:
        - status: 'ok'
        - service: static service label
        Errors: none (always 200)
        """
        return {
            'status': 'ok',
            'service': 'Food Data Central Python API'
        }

# Search endpoint
# Arguments contract (query string):
# - query (str, required): search query
# - dataType (str, optional): one of ['Foundation','SR Legacy','Survey','Branded']
# - pageSize (int, optional, default 50, 1..200)
# - pageNumber (int, optional, default 1)
# - sortBy (str, optional): one of ['dataType.keyword','lowercaseDescription.keyword','fdcId','publishedDate']
# - sortOrder (str, optional): ['asc','desc']
# - brandOwner (str, optional)
search_parser = api.parser()
search_parser.add_argument('query', type=str, required=True, help='Search query for food items', location='args')
search_parser.add_argument('dataType', type=str, help='Filter by data type', location='args',
                          choices=['Foundation', 'SR Legacy', 'Survey', 'Branded'])
search_parser.add_argument('pageSize', type=int, help='Number of results per page (1-200)',
                          location='args', default=50)
search_parser.add_argument('pageNumber', type=int, help='Page number (starts from 1)',
                          location='args', default=1)
search_parser.add_argument('sortBy', type=str, help='Field to sort results by', location='args',
                          choices=['dataType.keyword', 'lowercaseDescription.keyword', 'fdcId', 'publishedDate'])
search_parser.add_argument('sortOrder', type=str, help='Sort order for results', location='args',
                          choices=['asc', 'desc'])
search_parser.add_argument('brandOwner', type=str, help='Filter by brand owner (for branded foods)', location='args')


@search_ns.route('/search')
class FoodSearch(Resource):
    @search_ns.doc('search_foods')
    @search_ns.expect(search_parser)
    @search_ns.marshal_with(search_result_model)
    @search_ns.response(400, 'Bad Request', error_model)
    @search_ns.response(500, 'Internal Server Error', error_model)
    def get(self):
        """Search foods in the USDA FoodData Central database.

        Inputs (query params): see search_parser above.
        Behavior:
        - Validates presence of 'query'
        - Forwards request to USDA 'foods/search' with mapped params
        Outputs:
        - JSON including pagination metadata and an array of foods
        Errors:
        - 400: missing/invalid 'query'
        - 500: upstream/network errors
        Example:
        - GET /api/search?query=apple&pageSize=10
        """
        args = search_parser.parse_args()

        if not args['query']:
            return {'error': 'Query parameter is required'}, 400

        try:
            # Prepare parameters for USDA API
            params = {
                'query': args['query'],
                'pageSize': args['pageSize'],
                'pageNumber': args['pageNumber']
            }

            # Add optional parameters
            if args['dataType']:
                params['dataType'] = args['dataType']
            if args['sortBy']:
                params['sortBy'] = args['sortBy']
            if args['sortOrder']:
                params['sortOrder'] = args['sortOrder']
            if args['brandOwner']:
                params['brandOwner'] = args['brandOwner']

            response = make_usda_request('foods/search', params)
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching foods: {e}")
            return {'error': 'Failed to search foods'}, 500

# Image Classification endpoints
classify_parser = api.parser()
classify_parser.add_argument('image_path', type=str, required=True, help='Path to the image file to classify', location='args')

@classify_ns.route('/classify')
class ImageClassify(Resource):
    @classify_ns.doc('classify_food_image')
    @classify_ns.expect(classify_parser)
    @classify_ns.marshal_with(classification_result_model)
    @classify_ns.response(400, 'Bad Request', error_model)
    @classify_ns.response(500, 'Internal Server Error', error_model)
    def get(self):
        """Classify a food image and return the predicted food class.

        Inputs:
        - query: image_path (str) - path to the image file
        Behavior:
        - Loads the classification model if not already loaded
        - Preprocesses the image
        - Makes prediction using EfficientNet model
        Outputs:
        - JSON with predicted_class, confidence, and success status
        Errors:
        - 400: missing image_path or file not found
        - 500: model loading or prediction errors
        Example:
        - GET /api/classify?image_path=/path/to/pizza.jpg
        """
        args = classify_parser.parse_args()

        if not args['image_path']:
            return {'error': 'image_path parameter is required'}, 400

        image_path = args['image_path']

        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return {'error': f'Image file not found: {image_path}'}, 400

            # Load the classification model
            model = load_classification_model()

            # Preprocess the image
            image_tensor = preprocess_image(image_path)

            # Make prediction
            with torch.no_grad():
                output = model(image_tensor)

            # Get the predicted class
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_class_index = torch.argmax(probabilities, dim=1).item()
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = probabilities[0][predicted_class_index].item() * 100

            return {
                'predicted_class': predicted_class_name,
                'confidence': round(confidence, 2),
                'success': True
            }

        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            return {'error': 'Classification model not found'}, 500
        except Exception as e:
            logger.error(f"Error during image classification: {e}")
            return {'error': f'Classification failed: {str(e)}'}, 500

# Image Classification with nutrition data
classify_nutrition_parser = api.parser()
classify_nutrition_parser.add_argument('image_path', type=str, required=True, help='Path to the image file to classify', location='args')

# OpenAPI JSON endpoint
@app.route('/openapi.json')
def get_openapi_json():
    """Serve the OpenAPI JSON specification for this API.

    Inputs: none
    Outputs: OpenAPI v3 schema JSON as generated by Flask-RESTX
    Errors: none (200)
    """
    return jsonify(api.__schema__)


if __name__ == '__main__':
    print('Calorie Coach Python API server running on port 8004')
    print('Available endpoints:')
    print('  GET /health - Health check')
    print('  GET /api/search?query=... - Search foods')
    print('  GET /api/classify?image_path=... - Classify food image')
    print('  GET /docs/ - Interactive OpenAPI documentation')
    print('  GET /openapi.json - OpenAPI JSON specification')

    app.run(host='0.0.0.0', port=PORT, debug=True)
