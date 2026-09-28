# GeoForecast

GeoForecast is an advanced machine learning platform for spatiotemporal groundwater level forecasting in Maharashtra, India. It combines NASA GRACE satellite gravity data, ERA5-Land climate reanalysis, and historical observation well records to predict groundwater levels at both state and district resolutions.

## 1. Project Overview
GeoForecast addresses the critical need for proactive water resource management. By predicting groundwater levels up to 6 months in advance, it allows policymakers and agricultural stakeholders to anticipate water stress and plan interventions effectively.

## 2. Problem Statement
Maharashtra faces significant groundwater depletion due to erratic monsoons and over-extraction for agriculture. Traditional monitoring is reactive; stakeholders only know water levels after they drop. GeoForecast shifts the paradigm from reactive observation to proactive forecasting.

## 3. Objectives
- Forecast groundwater levels (m bgl - meters below ground level) at +1, +3, and +6 month horizons.
- Provide state-wide aggregated forecasts and high-resolution district-level forecasts.
- Discover the optimal ML architecture (tree-based vs deep learning) for this specific spatiotemporal problem.
- Deliver an interactive, fast, and accessible dashboard for exploring predictions.

## 4. Dataset Sources
- **NASA/JPL GRACE/GRACE-FO Mascon TWS**: Terrestrial Water Storage anomalies tracking deep aquifer changes.
- **NOAA ONI**: Oceanic Niño Index tracking El Niño/La Niña events.
- **NOAA MEI**: Multivariate ENSO Index.
- **ERA5-Land**: High-resolution precipitation (Rainfall) and 2m temperature.
- **Maharashtra Groundwater Well Records**: Historical monthly observations representing true groundwater depth.

## 5. Data Preprocessing
- Geographic alignment of satellite grid points to Maharashtra district boundaries.
- Temporal interpolation and chronological alignment to a unified monthly frequency.
- Handling missing satellite data (e.g., GRACE gaps) via robust median imputation coupled with missing-indicator features.

## 6. ML Methodology
The project evaluated several architectures:
- Tree-based models (Random Forest, XGBoost)
- Sequential models (LSTMs)
- Spatiotemporal ensembles

The strict chronological split (Train 70%, Validation 15%, Test 15%) ensured no future data leakage. Hyperparameter tuning was performed exclusively using the training and validation sets.

## 7. Experiments
Eight rigorous experiments drove model selection:
1. Base XGBoost/RF implementations
2. Imputation Strategies for GRACE gaps
3. Feature Selection & Regularization limits
4. Target Formulation (Absolute vs Change prediction)
5. Deep Learning (LSTM) evaluation
6. Spatiotemporal Cross-Validation Strategies
7. Ensemble Evaluation (RF + XGBoost combinations)
8. District Model Optimization (Temporal feature engineering: lags, rolling means)

## 8. Final Model Selection
The final selection prioritized performance on the untouched test set and generalization capability.
- **Maharashtra State**: XGBoost proved best for capturing broad state-wide climatic signals.
- **District Level**: Engineered temporal features combined with XGBoost provided massive gains at short horizons, while Random Forest remained most stable at the 6-month horizon.

## 9. Final Results

### MAHARASHTRA
- **+1 Month**: XGBoost (Test R² = 0.610, RMSE = 1.140 m)
- **+3 Months**: XGBoost (Test R² = 0.550, RMSE = 1.310 m)
- **+6 Months**: Regularized XGBoost (Test R² = 0.510, RMSE = 1.510 m)

### DISTRICT
- **+1 Month**: Optimized XGBoost (Test R² = 0.5735)
- **+3 Months**: Optimized XGBoost (Test R² = 0.5071)
- **+6 Months**: Random Forest (Test R² = 0.5979)

*(Note: State and district results come from different evaluation scopes and variance pools; they are not directly comparable.)*

## 10. System Architecture
The application uses a modern, decoupled three-tier architecture:
- Data Processing & ML Pipeline (Offline)
- FastAPI Backend (Online API & Model Serving)
- React Frontend (Dashboard UI)

## 11. Backend Architecture
The backend is built with FastAPI for high-performance async requests. It utilizes a `PredictionService` that dynamically generates temporal features (lags, rolling averages) on demand from historical database records before passing them to the loaded ML models.

## 12. REST API
- `GET /api/health`: System status
- `GET /api/districts`: Supported districts list
- `GET /api/forecast/maharashtra?horizon={h}`: State-wide forecast
- `GET /api/forecast/{district}?horizon={h}`: District-specific forecast

## 13. PostgreSQL Schema
- `districts`: Stores geospatial metadata and bounds.
- `monthly_observations`: Stores chronological features for the state and each district.
- `model_metadata`: Tracks selected features, hyperparameters, and test metrics.
- `forecasts`: Stores generated predictions for historical tracking.

## 14. Prediction Pipeline
1. Client requests a forecast.
2. Backend queries the database for the latest historical observation.
3. Feature builder constructs the exact chronological temporal vector (e.g., `Rainfall_Rolling_3M`, `Groundwater_Lag_1`) required by the specific model.
4. Model executes and returns the predicted m bgl.

## 15. Frontend Architecture
Built with React and styled with Tailwind CSS, the frontend features:
- A responsive, full-width Overview Dashboard
- Interactive District Explorer with dynamic chart generation via Recharts
- Interactive Leaflet/OpenStreetMap integration
- A Model Performance page driven dynamically by the API's model metadata.

## 16. Project Structure
```
geoforecast/
├── backend/          # FastAPI application & PostgreSQL integration
├── data/             # Raw and processed datasets
├── frontend/         # React application
├── models/           # Production joblib artifacts and metadata
├── notebooks/        # Exploratory data analysis
├── reports/          # Experiment results and data quality logs
├── src/              # ML pipelines, feature engineering, and model training
└── README.md
```

## 17. Installation
1. Clone the repository: `git clone https://github.com/moksh2209/GeoForecast.git`
2. Create a Python virtual environment: `python -m venv venv`
3. Install backend dependencies: `pip install -r backend/requirements.txt`
4. Install frontend dependencies: `cd frontend && npm install`

## 18. Running Locally
1. Start PostgreSQL server and configure `.env` (copy from `.env.example`).
2. Run database setup: `python create_db.py`
3. Start backend: `python -m uvicorn backend.app.main:app --reload`
4. Start frontend: `cd frontend && npm run dev`

## 19. Scientific Assumptions
- The target variable is measured in meters below ground level (m bgl). A higher value indicates deeper (lower) groundwater.
- GRACE TWS anomalies are assumed to have a linear or monotonic relationship with shallow aquifer changes in this region.
- Historical well distributions are assumed to be a representative sample of district-wide water tables.

## 20. Limitations
- Unrecorded localized extraction (e.g., illegal borewells) introduces noise that cannot be captured by meteorological or satellite inputs.
- Long-horizon (+6 months) predictions are heavily constrained by the unpredictability of future monsoon onset.

## 21. Future Scope
- Integration of high-resolution crop-sowing patterns.
- Real-time IoT sensor ingestion for daily corrections.
- Extension to neighboring states.
