# 🌾 Climate-Smart Agriculture: AI-Powered Crop Recommendation

A production-grade machine learning system for crop recommendation, disease detection, and agricultural risk analysis. Features satellite data integration, geospatial mapping, and MLOps tracking.

## 🌟 Key Features

- **🎯 99.09% Accuracy** - High-precision crop recommendations for 22 crops
- **🛰️ Satellite Integration** - Real-time soil moisture and weather data from NASA POWER API
- **🗺️ Karnataka Risk Mapping** - Interactive mapping of 30 Karnataka districts
- **🔬 MLOps Tracking** - Experiment tracking and model versioning with MLflow
- **🍃 Disease Detection** - AI-powered CNN models for 9 crops (fully functional)
- **📊 Analytics Dashboard** - Comprehensive data visualization and insights
- **🎨 Premium UI** - Beautiful dark/light theme with smooth animations
- **📚 Complete Documentation** - Comprehensive technical documentation for all AI/ML models

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Application

```bash
# Start the web app
streamlit run src/app_enhanced.py --server.port 8501
```

Access at: **http://localhost:8501**

### 3. Documentation

- **📖 [Model Documentation](MODEL_DOCUMENTATION.md)** - Complete AI/ML technical guide
- **🗺️ [Karnataka Map Update](KARNATAKA_MAP_UPDATE.md)** - Geographic focus information
- **🖼️ [Disease Detection Testing](test_images/HOW_TO_USE.md)** - How to test disease detection with sample images

## 📂 Project Structure

```
agri-ml/
├── data/                           # Datasets and cache
│   ├── crop_recommendation.csv    # Training data (2,200 samples)
│   ├── satellite_cache/           # Cached satellite data
│   └── maps/                      # Generated risk maps
├── models/                         # Trained models (99.09% accuracy)
│   ├── crop_classifier.pkl        # Main model (RandomForest)
│   ├── scaler.pkl                 # Feature scaler (StandardScaler)
│   ├── disease/                   # Disease detection CNN models
│   │   ├── apple_model.h5         # Apple disease CNN
│   │   ├── apple_classes.json     # Class mappings
│   │   └── ... (9 crops total)    # All 9 disease models
│   └── *.png                      # Performance visualizations
├── src/                            # Source code
│   ├── app_enhanced.py            # 🎯 Main Streamlit app
│   ├── train_high_accuracy.py     # High-accuracy training
│   ├── train_enhanced_mlops.py    # 🔬 Enhanced training with MLOps
│   ├── satellite_data.py          # 🛰️ Satellite data integration
│   ├── geospatial_mapping.py      # 🗺️ Risk mapping
│   ├── mlops_tracking.py          # MLflow tracking
│   └── theme_styles.py            # UI themes
├── test_images/                    # Sample disease images (24 files)
│   ├── HOW_TO_USE.md              # Testing guide
│   └── *.jpg                      # Test images for all 9 crops
├── mlruns/                         # MLflow experiment tracking
├── MODEL_DOCUMENTATION.md          # 📖 Complete AI/ML technical documentation
├── KARNATAKA_MAP_UPDATE.md         # 🗺️ Karnataka map focus documentation
└── requirements.txt                # Dependencies
```

## 🎯 Advanced Features

### 1. 🛰️ Satellite Data Integration

Fetch real-time soil moisture and weather data from NASA POWER API:

```python
from src.satellite_data import SatelliteDataProvider

provider = SatelliteDataProvider()

# Get soil moisture data
data = provider.get_soil_moisture(
    latitude=12.9716, 
    longitude=77.5946,
    start_date='2025-01-01',
    end_date='2025-01-31'
)

# Get enhanced features
features = provider.get_enhanced_features(
    latitude=12.9716,
    longitude=77.5946,
    district="Bengaluru Urban",
    state="Karnataka",
    lookback_days=90
)
```

**Features Retrieved:**
- Soil moisture (root zone & profile)
- Precipitation
- Temperature
- Relative humidity
- Station-level weather data

### 2. 🗺️ Karnataka Geospatial Risk Mapping

Create interactive district-level risk maps for Karnataka:

```python
from src.geospatial_mapping import GeospatialRiskMapper

mapper = GeospatialRiskMapper()

# Create Karnataka risk map (all 30 districts)
risk_map = mapper.generate_karnataka_comprehensive_map()

# Or create custom district risk map
risk_map = mapper.create_district_risk_map(
    district_data,  # DataFrame with risk scores
    save_path='risk_map.html'
)
```

**Karnataka Districts (30 total):**
Bengaluru Urban, Mysuru, Dharwad, Belagavi, Mangaluru, Kalaburagi, Ballari, Tumakuru, Shivamogga, Hubballi, Raichur, Vijayapura, Hassan, Chitradurga, Davanagere, Mandya, Chikkamagaluru, Udupi, Bidar, Chamarajanagara, Yadgir, Gadag, Haveri, Bagalkot, Koppal, Ramanagara, Chikkaballapura, Kodagu, Uttara Kannada, Dakshina Kannada

**Map Configuration:**
- Center: Karnataka (15.3173°N, 75.7139°E)
- Zoom Level: 7 (state-level view)
- Focus: Karnataka agricultural regions

**Risk Factors:**
- Soil moisture levels
- Rainfall patterns
- Temperature extremes
- Disease probability

### 3. 🔬 MLOps Experiment Tracking

Track experiments with MLflow:

```python
from src.mlops_tracking import MLOpsTracker

tracker = MLOpsTracker(experiment_name="crop-recommendation")

# Start tracking
tracker.start_run(run_name="experiment_1")

# Log parameters
tracker.log_params({
    'n_estimators': 300,
    'max_depth': 20
})

# Log metrics
tracker.log_metrics({
    'accuracy': 0.9977,
    'f1_macro': 0.9977
})

# Log model
tracker.log_model(model, model_type='sklearn')

# End run
tracker.end_run()
```

**View MLflow UI:**
```bash
mlflow ui --backend-store-uri ./mlruns
```

Access at: **http://localhost:5000**

## 🔧 Training Options

### Basic Training (99.77% Accuracy)

```bash
python src/train_high_accuracy.py
```

### Enhanced Training with MLOps

```bash
# Without satellite data
python src/train_enhanced_mlops.py

# With satellite data (requires internet)
python src/train_enhanced_mlops.py --satellite
```

**Features:**
- ✅ Experiment tracking with MLflow
- ✅ Geospatial risk map generation
- ✅ Optional satellite data enhancement
- ✅ Comprehensive logging and metrics
- ✅ Model versioning and registry

## 📊 Model Performance

### Crop Prediction Model
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| RandomForest (Current) | **99.09%** | 99.10% | 99.09% | 99.09% |

**Model Details:**
- Algorithm: Random Forest Classifier
- Estimators: 500 trees
- Max Depth: 20
- Features: 7 (N, P, K, temperature, humidity, pH, rainfall)
- Training Samples: 2,200

**Supported Crops (22):**
Apple, Banana, Blackgram, Chickpea, Coconut, Coffee, Cotton, Grapes, Jute, Kidney beans, Lentil, Maize, Mango, Mothbeans, Mungbean, Muskmelon, Orange, Papaya, Pigeonpeas, Pomegranate, Rice, Watermelon

### Disease Detection Models
All 9 crop disease detection models loaded successfully using CNN architecture.

See **[MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md)** for complete technical details.

## 🍃 Disease Detection

Upload plant leaf images to detect diseases for 9 crops using trained CNN models:

**Supported Crops:**
- 🍎 Apple (4 classes: Healthy, Apple Scab, Black Rot, Cedar Apple Rust)
- 🍅 Tomato (10 classes: Healthy, Early Blight, Late Blight, Leaf Mold, etc.)
- 🌽 Corn (4 classes: Healthy, Common Rust, Gray Leaf Spot, Northern Leaf Blight)
- 🍇 Grape (4 classes: Healthy, Black Rot, Esca, Leaf Blight)
- 🥔 Potato (3 classes: Healthy, Early Blight, Late Blight)
- 🌶️ Pepper (2 classes: Healthy, Bacterial Spot)
- 🍒 Cherry (2 classes: Healthy, Powdery Mildew)
- 🍑 Peach (2 classes: Healthy, Bacterial Spot)
- 🍓 Strawberry (2 classes: Healthy, Leaf Scorch)

**Model Status:** ✅ All 9 disease detection models loaded successfully

**Test Images:** 24 sample images available in `test_images/` directory
- See [test_images/HOW_TO_USE.md](test_images/HOW_TO_USE.md) for testing guide

**Model Architecture:**
- CNN with Conv2D layers (32→64→128 filters)
- MaxPooling, Dropout (0.3), BatchNormalization
- Input: 224x224x3 RGB images
- Data augmentation: flip, rotate, zoom

For complete technical details, see **[MODEL_DOCUMENTATION.md](MODEL_DOCUMENTATION.md)**

## 🎨 Web Application Features

```bash
streamlit run src/app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎯 Features

### Preprocessing & Feature Engineering

- **Data cleaning:** Median imputation, outlier removal
- **Encoding:** One-hot encoding for categorical features
- **Feature engineering:**
  - Rainfall intensity (mm/rainy day)
  - Growing Degree Days (GDD)
  - Moisture stress indicator
  - NPK nutrient ratios
- **Temporal split:** Train on older years, test on recent years (prevents data leakage)

### Model Architecture

**Yield Prediction (Regression):**
- XGBoost Regressor (primary)
- Random Forest Regressor
- Ridge Regression (baseline)

**Metrics:** RMSE, MAE, R²

**Optional Enhancements:**
- Crop recommendation classifier
- Time-series rainfall forecasting
- SHAP feature importance analysis

### Web Application

Interactive Streamlit app with:
- **Input controls:** Soil, weather, and nutrient parameters
- **Yield prediction:** Real-time ML inference
- **Smart recommendations:**
  - Soil pH correction (lime/gypsum)
  - Organic matter enhancement
  - Irrigation scheduling
  - Nutrient management
  - Temperature adaptation

## 📈 Model Performance

Expected performance (varies with data quality):
- **RMSE:** 0.3-0.8 t/ha
- **R²:** 0.75-0.90
- **MAE:** 0.2-0.6 t/ha

## 🔬 Advanced Usage

### Hyperparameter Tuning with Optuna

```python
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_squared_error(y_test, preds, squared=False)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
```

### Feature Importance Analysis

```python
import shap

# Load model
model = joblib.load("models/yield_xgboost.pkl")

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot
shap.summary_plot(shap_values, X_test)
```

## 🌍 Localization for Karnataka

For Udupi/Karnataka region:

**Kharif Season (Jun-Oct):**
- Monsoon: 2000-3500mm rainfall
- Crops: Rice, Maize, Groundnut
- Base temperature: 10-12°C

**Rabi Season (Nov-Feb):**
- Post-monsoon: 200-800mm rainfall
- Crops: Ragi, Pulses
- Irrigation critical

## 🛠️ Troubleshooting

**Models not found error:**
```bash
# Train model first
python src/train.py
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

**Data file not found:**
- Ensure `data/merged_agri.csv` exists with required columns
- Check column names match exactly (case-sensitive)

## 📚 Next Steps

1. **Data quality:** Add satellite-derived soil moisture, station-level weather
2. **Geospatial:** Integrate district shapefiles for risk mapping
3. **MLOps:** Track experiments with MLflow, version models
4. **Deployment:** Containerize with Docker, deploy to cloud (AWS/Azure)
5. **Mobile app:** Build farmer-facing mobile interface

## 📄 License

MIT License - feel free to use and modify for your agricultural projects!

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📧 Contact

For questions or collaboration: [Your contact info]

---

**Built with ❤️ for sustainable agriculture and climate adaptation**
