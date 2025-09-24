# CalorieCoach 🍎📊

A comprehensive AI-powered food classification and nutrition analysis application that combines computer vision, machine learning, and nutritional data to provide detailed food insights.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Information](#model-information)
- [Contributing](#contributing)

## 🎯 Overview

CalorieCoach is an intelligent food analysis system that:
- **Classifies food images** using a trained EfficientNet V2 model
- **Provides detailed nutritional information** via USDA FoodData Central API
- **Offers multiple interfaces** including REST API, MCP server, and frontend applications
- **Supports 35 food categories** including both international and Indian cuisine

## 🏗️ Architecture

![Application Architecture](Request_flow.jpg)

The application follows a modular architecture with the following components:

### Core Components

1. **Backend API Server** (`backend/app.py`)
   - Flask-based REST API with OpenAPI documentation
   - Food image classification using PyTorch EfficientNet V2
   - USDA FoodData Central API integration
   - Nutritional data retrieval and processing

2. **MCP Server** (`mcp_server/mcp_server.py`)
   - Model Context Protocol server for LLM integration
   - Provides tools for food search and nutritional analysis
   - Enables AI agents to access food data programmatically

3. **Frontend Applications** (`frontend/`)
   - `main.py`: Core food classification interface
   - `main_agent.py`: OpenAI GPT-4 Vision integration for ingredient analysis
   - `main_backup.py`: Backup implementation

4. **Machine Learning Pipeline** (`ml/`)
   - Jupyter notebook for model training and testing
   - EfficientNet V2 model for food classification
   - Support for 35 food categories

5. **Tools & Utilities** (`tools/`)
   - Food summary generation
   - CLI interface for nutritional analysis
   - Helper functions for data processing

## ✨ Features

### 🔍 Food Classification
- **35 Food Categories**: apple_pie, baked_potato, burger, butter_naan, chai, chapati, cheesecake, chicken_curry, chole_bhature, crispy_chicken, dal_makhani, dhokla, donut, fried_rice, fries, hot_dog, ice_cream, idli, jalebi, kaathi_rolls, kadai_paneer, kulfi, masala_dosa, momos, omelette, paani_puri, pakode, pav_bhaji, pizza, samosa, sandwich, sushi, taco, taquito
- **High Accuracy**: EfficientNet V2 pre-trained model
- **Real-time Processing**: Fast image classification

### 📊 Nutritional Analysis
- **Comprehensive Data**: Calories, proteins, fats, carbohydrates, vitamins, minerals
- **USDA Integration**: Access to extensive food database
- **Detailed Breakdowns**: Macro and micronutrient information
- **Portion Analysis**: Multiple serving size options

### 🤖 AI Integration
- **OpenAI GPT-4 Vision**: Ingredient identification from images
- **MCP Protocol**: LLM-friendly API for AI agents
- **Automated Analysis**: End-to-end food analysis pipeline

### 🌐 API Features
- **REST API**: Complete food search and analysis endpoints
- **OpenAPI Documentation**: Interactive API documentation at `/docs/`
- **Health Monitoring**: System health check endpoints
- **Error Handling**: Comprehensive error responses

## 📁 Project Structure

```
CalorieCoach/
├── backend/
│   └── app.py                    # Flask API server with food classification
├── frontend/
│   ├── main.py                   # Core classification interface
│   ├── main_agent.py             # OpenAI integration for ingredient analysis
│   └── main_backup.py            # Backup implementation
├── mcp_server/
│   ├── mcp_server.py             # Model Context Protocol server
│   └── mcp_config.json           # MCP configuration
├── ml/
│   └── UFA_Calorie_Coach_Testing.ipynb  # Model training/testing notebook
├── models/
│   └── model_efficientnet_v2_m_1.pth    # Trained classification model
├── tools/
│   ├── food_summary.py           # Nutritional summary generator
│   ├── food_summary_cli.py       # CLI interface
│   └── __pycache__/
├── data/
│   ├── Train/                    # Training dataset (35 food categories)
│   ├── Valid/                    # Validation dataset
│   └── Test/                     # Test dataset
├── zipfile/
│   └── food_classification_dataset_V21.zip  # Complete dataset archive
├── requirements.txt              # Python dependencies
├── test.md                       # Sample nutrition output
└── Request_flow.jpg              # Architecture diagram
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (recommended)
- USDA FoodData Central API key
- OpenAI API key (for GPT-4 Vision features)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CalorieCoach
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   USDA_API_KEY=your_usda_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   ```

4. **Download Model**
   Ensure the trained model is placed at:
   ```
   models/model_efficientnet_v2_m_1.pth
   ```

5. **Verify Installation**
   ```bash
   python backend/app.py
   ```

## 💻 Usage

### Backend API Server

Start the Flask API server:
```bash
cd backend
python app.py
```

The API will be available at `http://localhost:8004`

**Available Endpoints:**
- `GET /docs/` - Interactive API documentation
- `GET /health` - Health check
- `POST /api/classify` - Food image classification
- `GET /api/search` - Search foods in USDA database
- `GET /api/food/{fdc_id}` - Get detailed food information

### MCP Server

Start the MCP server for LLM integration:
```bash
cd mcp_server
python mcp_server.py
```

### Frontend Applications

**Food Classification:**
```bash
cd frontend
python main.py 
```
 

### CLI Tools

**Nutritional Summary:**
```bash
cd tools
python food_summary_cli.py "samosa"
```

## 📖 API Documentation

### Food Classification

**POST** `/api/classify`

Upload an image for food classification:

```bash
curl -X POST -F "image=@path/to/food_image.jpg" http://localhost:8004/api/classify
```

**Response:**
```json
{
  "predicted_class": "samosa",
  "confidence": 0.95,
  "nutritional_summary": {
    "calories": 310,
    "protein": 5.14,
    "total_fat": 17.47,
    "carbohydrates": 33.16
  }
}
```

### Food Search

**GET** `/api/search?query=apple&pageSize=10`

Search for foods in the USDA database:

**Response:**
```json
{
  "foods": [
    {
      "fdcId": 171688,
      "description": "Apples, raw, with skin",
      "brandOwner": "",
      "ingredients": ""
    }
  ],
  "totalHits": 150,
  "currentPage": 1
}
```

### Food Details

**GET** `/api/food/171688`

Get detailed nutritional information:

**Response:**
```json
{
  "fdcId": 171688,
  "description": "Apples, raw, with skin",
  "foodNutrients": [
    {
      "nutrient": {
        "name": "Energy",
        "unitName": "kcal"
      },
      "amount": 52
    }
  ]
}
```

## 🤖 Model Information

### EfficientNet V2 Medium
- **Architecture**: EfficientNet V2 Medium
- **Training Data**: 35 food categories with thousands of images per class
- **Input Size**: 224x224 RGB images
- **Output**: 35-class classification
- **Preprocessing**: Standard ImageNet normalization

### Supported Food Categories

**International Cuisine:**
- apple_pie, baked_potato, burger, cheesecake, donut, fries, hot_dog, ice_cream, pizza, sandwich, sushi, taco, taquito

**Indian Cuisine:**
- butter_naan, chai, chapati, chicken_curry, chole_bhature, crispy_chicken, dal_makhani, dhokla, idli, jalebi, kaathi_rolls, kadai_paneer, kulfi, masala_dosa, momos, paani_puri, pakode, pav_bhaji, samosa

**Others:**
- omelette, fried_rice

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `USDA_API_KEY` | USDA FoodData Central API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 Vision | Optional |

### Model Configuration

- **Model Path**: `models/model_efficientnet_v2_m_1.pth`
- **Device**: Automatically detects CUDA/CPU
- **Batch Size**: Configurable (default: 1 for inference)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the requirements.txt file for third-party package licenses.

## 🙏 Acknowledgments

- **USDA FoodData Central** for comprehensive nutritional data
- **PyTorch** and **torchvision** for deep learning capabilities
- **OpenAI** for GPT-4 Vision API
- **EfficientNet** architecture by Google Research
- **Flask** and **Flask-RESTX** for API development

## 📞 Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the API documentation at `/docs/`
- Review the sample outputs in `test.md`

---

**CalorieCoach** - Making nutrition analysis intelligent and accessible! 🍎✨
