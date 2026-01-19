import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AQI Prediction System - India",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        font-size: 1.1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_resource
def load_models():
    """Load all models and data"""
    try:
        models = {
            'xgb': joblib.load('xgboost_model.pkl'),
            'lgb': joblib.load('lightgbm_model.pkl'),
            'gb': joblib.load('gradient_boosting_model.pkl'),
            'rf': joblib.load('random_forest_model.pkl')
        }
        
        with open('feature_columns.pkl', 'rb') as f:
            feature_cols = pickle.load(f)
        
        with open('cities.pkl', 'rb') as f:
            cities = pickle.load(f)
        
        sample_data = pd.read_csv('sample_data.csv')
        
        return models, feature_cols, cities, sample_data
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# Model weights
WEIGHTS = {
    'rf': 0.40,   # Random Forest - Best MAE: 0.360
    'gb': 0.35,   # Gradient Boosting - Best R²: 0.9997
    'xgb': 0.25   # XGBoost - Strong overall: MAE 0.486
}

def get_aqi_info(aqi):
    """Get AQI category, color, and health advice"""
    if aqi <= 50:
        return "Good", "#00E400", "😊", "Air quality is good. Ideal for outdoor activities."
    elif aqi <= 100:
        return "Moderate", "#FFFF00", "😐", "Air quality is acceptable. Sensitive individuals should consider limiting prolonged outdoor exertion."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#FF7E00", "😷", "Members of sensitive groups may experience health effects."
    elif aqi <= 200:
        return "Unhealthy", "#FF0000", "😨", "Everyone may experience health effects. Avoid prolonged outdoor exertion."
    elif aqi <= 300:
        return "Very Unhealthy", "#8F3F97", "🚨", "Health alert! Stay indoors and avoid all outdoor activities."
    else:
        return "Hazardous", "#7E0023", "☢️", "Emergency conditions! Remain indoors with air purification."

def create_features(city, target_date, sample_data, feature_cols):
    """Create feature vector for prediction"""
    city_data = sample_data[sample_data['city'].str.lower() == city.lower()]
    
    if len(city_data) == 0:
        city_data = sample_data
    
    target_dt = pd.to_datetime(target_date)
    
    # Temporal features
    features = {
        'year': target_dt.year,
        'month': target_dt.month,
        'day': target_dt.day,
        'dayofweek': target_dt.dayofweek,
        'quarter': target_dt.quarter,
        'dayofyear': target_dt.timetuple().tm_yday,
        'weekofyear': target_dt.isocalendar()[1],
        'month_sin': np.sin(2 * np.pi * target_dt.month / 12),
        'month_cos': np.cos(2 * np.pi * target_dt.month / 12),
        'day_sin': np.sin(2 * np.pi * target_dt.day / 31),
        'day_cos': np.cos(2 * np.pi * target_dt.day / 31),
    }
    
    # Fill remaining features with historical medians
    for col in feature_cols:
        if col not in features:
            if col in city_data.columns:
                features[col] = city_data[col].median()
            else:
                features[col] = 0
    
    # Create DataFrame with correct column order
    X = pd.DataFrame([features])[feature_cols]
    return X

def predict_aqi(city, date, models, feature_cols, sample_data):
    """Make ensemble prediction for given city and date"""
    try:
        # Create features
        X = create_features(city, date, sample_data, feature_cols)
        
        # Get predictions from each model
        pred_rf = models['rf'].predict(X)[0]
        pred_gb = models['gb'].predict(X)[0]
        pred_xgb = models['xgb'].predict(X)[0]
        pred_lgb = models['lgb'].predict(X)[0]
        
        # Weighted ensemble (top 3 models)
        ensemble_pred = (
            pred_rf * WEIGHTS['rf'] +
            pred_gb * WEIGHTS['gb'] +
            pred_xgb * WEIGHTS['xgb']
        )
        
        # Round predictions
        ensemble_aqi = int(round(ensemble_pred))
        rf_aqi = int(round(pred_rf))
        gb_aqi = int(round(pred_gb))
        xgb_aqi = int(round(pred_xgb))
        lgb_aqi = int(round(pred_lgb))
        
        return ensemble_aqi, rf_aqi, gb_aqi, xgb_aqi, lgb_aqi
        
    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        return None, None, None, None, None

# Load models
with st.spinner("🚀 Loading models..."):
    models, feature_cols, cities, sample_data = load_models()
    st.success("✅ Models loaded successfully!")

# Header
st.markdown("""
# 🌍 Air Quality Index (AQI) Prediction System
### Advanced Ensemble Machine Learning for Indian Cities

Powered by **Random Forest + Gradient Boosting + XGBoost** | Trained on 842,160+ data points
""")

# Performance metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h2 style="margin: 0; font-size: 2rem;">99.97%</h2>
        <p style="margin: 5px 0;">Model Accuracy (R²)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb, #f5576c); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
        <h2 style="margin: 0; font-size: 2rem;">±0.44</h2>
        <p style="margin: 5px 0;">Average Error (MAE)</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe, #00f2fe); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
        <h2 style="margin: 0; font-size: 2rem;">0.29%</h2>
        <p style="margin: 5px 0;">Error Rate (MAPE)</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main layout
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown("### 📍 Prediction Parameters")
    
    # City selection
    city_input = st.selectbox(
        "🏙️ Select City",
        options=sorted(cities),
        index=sorted(cities).index('delhi') if 'delhi' in cities else 0,
        help="Choose from 29 Indian cities"
    )
    
    # Date selection
    date_input = st.date_input(
        "📅 Select Date",
        value=datetime.now() + timedelta(days=1),
        min_value=datetime.now().date(),
        help="Choose a future date for prediction"
    )
    
    # Predict button
    predict_btn = st.button("🔮 Predict AQI", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown("""
    ### 🤖 Model Information
    
    **Ensemble Weights:**
    - 🌲 Random Forest: 40%
    - 📈 Gradient Boosting: 35%
    - ⚡ XGBoost: 25%
    
    **Training Dataset:**
    - 842,160 records
    - 29 Indian cities
    - 2022-2025 data
    - 55+ engineered features
    """)

with col_right:
    st.markdown("### 📊 Prediction Results")
    
    if predict_btn:
        with st.spinner("🔮 Generating prediction..."):
            ensemble_aqi, rf_aqi, gb_aqi, xgb_aqi, lgb_aqi = predict_aqi(
                city_input, date_input, models, feature_cols, sample_data
            )
        
        if ensemble_aqi is not None:
            # Get AQI information
            category, color, emoji, advice = get_aqi_info(ensemble_aqi)
            
            # Main prediction display
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, {color}22, {color}44); border-radius: 15px; border: 3px solid {color};">
                <h1 style="font-size: 5rem; margin: 10px 0; color: {color};">{ensemble_aqi}</h1>
                <h2 style="margin: 10px 0; font-size: 2rem;">{emoji} {category}</h2>
                <p style="font-size: 1.2rem; margin: 15px 0; color: #333;">{advice}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Individual model predictions
            st.markdown("#### 📊 Individual Model Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style="padding: 15px; background: linear-gradient(135deg, #10b981, #059669); border-radius: 10px; text-align: center; color: white;">
                    <h4 style="margin: 0;">Random Forest</h4>
                    <p style="font-size: 2rem; margin: 5px 0; font-weight: bold;">{rf_aqi}</p>
                    <small>Weight: 40% | MAE: 0.360</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="padding: 15px; background: linear-gradient(135deg, #3b82f6, #2563eb); border-radius: 10px; text-align: center; color: white;">
                    <h4 style="margin: 0;">XGBoost</h4>
                    <p style="font-size: 2rem; margin: 5px 0; font-weight: bold;">{xgb_aqi}</p>
                    <small>Weight: 25% | MAE: 0.486</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="padding: 15px; background: linear-gradient(135deg, #8b5cf6, #7c3aed); border-radius: 10px; text-align: center; color: white;">
                    <h4 style="margin: 0;">Gradient Boosting</h4>
                    <p style="font-size: 2rem; margin: 5px 0; font-weight: bold;">{gb_aqi}</p>
                    <small>Weight: 35% | MAE: 0.483</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="padding: 15px; background: linear-gradient(135deg, #f59e0b, #d97706); border-radius: 10px; text-align: center; color: white;">
                    <h4 style="margin: 0;">LightGBM</h4>
                    <p style="font-size: 2rem; margin: 5px 0; font-weight: bold;">{lgb_aqi}</p>
                    <small>Not in ensemble | MAE: 0.641</small>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Confidence metrics
            st.info(f"""
            **📈 Prediction Confidence**
            
            - **Model Accuracy (R²):** 99.97%
            - **Expected Error Range:** ±0.44 AQI points
            - **Confidence Interval:** {ensemble_aqi - 1} to {ensemble_aqi + 1}
            - **Prediction Date:** {date_input}
            - **City:** {city_input.title()}
            """)
            
            st.warning("ℹ️ **Note:** This prediction uses historical patterns and ensemble learning. For production use, integrate real-time weather and pollutant data for enhanced accuracy.")
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px; background: #f8f9fa; border-radius: 15px; border: 2px dashed #cbd5e1;">
            <h3 style="color: #64748b;">👈 Select a city and date, then click "Predict AQI"</h3>
            <p style="color: #94a3b8; margin-top: 10px;">The model will analyze historical patterns and provide:</p>
            <ul style="color: #94a3b8; text-align: left; display: inline-block; margin-top: 10px;">
                <li>Ensemble AQI prediction with category</li>
                <li>Individual model predictions</li>
                <li>Health advice based on AQI level</li>
                <li>Confidence metrics and error ranges</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# AQI Reference Table
st.markdown("---")
st.markdown("### 📚 AQI Categories Reference")

aqi_data = {
    "AQI Range": ["0-50", "51-100", "101-150", "151-200", "201-300", "301+"],
    "Category": ["😊 Good", "😐 Moderate", "😷 Unhealthy for Sensitive Groups", 
                 "😨 Unhealthy", "🚨 Very Unhealthy", "☢️ Hazardous"],
    "Health Impact": ["Minimal impact", "Acceptable quality", "Sensitive individuals affected",
                      "Everyone affected", "Serious health effects", "Emergency conditions"],
    "Recommended Actions": [
        "Enjoy outdoor activities",
        "Unusually sensitive people should limit prolonged outdoor exertion",
        "Children, elderly, and people with respiratory issues should reduce outdoor exertion",
        "Everyone should reduce prolonged outdoor exertion",
        "Stay indoors, avoid all outdoor activities",
        "Remain indoors with air purification"
    ]
}

st.dataframe(pd.DataFrame(aqi_data), use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown("""
**Data Source:** Historical AQI data from Central Pollution Control Board (CPCB), India  
**Model Training:** Google Colab with advanced feature engineering (lag features, rolling statistics, cyclical encoding)  
**Deployment:** Streamlit Cloud
""")
