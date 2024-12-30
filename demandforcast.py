import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# Train and Save Models (Only if not already saved)
@st.cache_resource
def train_and_save_models():
    # Check if models are already saved
    if os.path.exists("random_forest_model.pkl") and os.path.exists("gradient_boosting_model.pkl"):
        rf = joblib.load("random_forest_model.pkl")
        gb = joblib.load("gradient_boosting_model.pkl")
        return rf, gb

    # Load and preprocess the training data
    data = pd.read_csv(r"C:\\Users\\HP\\Downloads\\train.csv\\train.csv")
    data['date'] = pd.to_datetime(data['date'])
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['weekday'] = data['date'].dt.weekday
    data = data.drop(columns=['date'])
    features = ['store', 'item', 'month', 'day', 'weekday']
    target = 'sales'

    # Train-test split
    X = data[features]
    y = data[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train)

    # Save models
    joblib.dump(rf, "random_forest_model.pkl")
    joblib.dump(gb, "gradient_boosting_model.pkl")
    return rf, gb

# Load or Train Models
rf_model, gb_model = train_and_save_models()

# Streamlit App
st.markdown("""
    <style>
        body {
            background-color: #1e3a8a;  /* Dark Blue Background */
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 15px 32px;
            border-radius: 12px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stMetric>div {
            background-color: rgba(0, 0, 0, 0.5);
            padding: 20px;
            border-radius: 10px;
            font-size: 24px;
        }
        .css-1v3fvcr {
            background-color: #1e3a8a;  /* Ensure all elements have dark blue background */
        }
    </style>
""", unsafe_allow_html=True)

st.title("Sales Prediction App")
st.markdown("""
This app predicts sales for a given store and item combination.  
Input the details below to get the predicted sales!
""")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    store = st.number_input("Store ID", min_value=1, step=1)
    item = st.number_input("Item ID", min_value=1, step=1)

with col2:
    date = st.date_input("Select Date")
    month = date.month
    day = date.day
    weekday = date.weekday()

# Create input DataFrame
input_data = pd.DataFrame({
    "store": [store],
    "item": [item],
    "month": [month],
    "day": [day],
    "weekday": [weekday]
})

# Predict Sales
if st.button("Predict"):
    rf_pred = rf_model.predict(input_data)[0]
    gb_pred = gb_model.predict(input_data)[0]
    ensemble_pred = (rf_pred + gb_pred) / 2

    # Display prediction
    st.metric(label="Predicted Sales", value=f"{ensemble_pred:.2f}")

    # Additional insights for shopkeepers
    st.markdown("""
    ### Key Takeaways:
    - Stock up inventory as per the predicted sales.
    - Plan promotions and discounts for higher sales periods.
    - Adjust resources for expected peak or low sales days.
    """)
