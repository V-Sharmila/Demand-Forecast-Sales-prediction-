# Sales Forecasting Using ARIMA and Machine Learning Models

This repository contains a project that forecasts sales for a retail dataset using both time series analysis (ARIMA) and machine learning techniques (Random Forest and Gradient Boosting). The implementation includes a user-friendly **Streamlit** application for visualization and interaction.

---

## **Table of Contents**

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Dataset Information](#dataset-information)
- [Steps in the Workflow](#steps-in-the-workflow)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Time Series Analysis](#2-time-series-analysis)
  - [3. Machine Learning Models](#3-machine-learning-models)
  - [4. Streamlit Application](#4-streamlit-application)
- [Results and Metrics](#results-and-metrics)
- [How to Run](#how-to-run)
- [Acknowledgments](#acknowledgments)

---

## **Introduction**

This project aims to forecast sales for a retail dataset that contains store and item-level sales over time. It leverages:
1. Time series modeling (ARIMA) for trend-based forecasting.
2. Machine learning models (Random Forest and Gradient Boosting) for feature-based forecasting.
3. A **Streamlit** application for an interactive user interface to visualize and analyze results.

---

## **Technologies Used**

- **Programming Language**: Python
- **Libraries**:
  - Data Processing: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Time Series Analysis: `statsmodels`
  - Machine Learning: `scikit-learn`
  - Web Application: `Streamlit`
- **Tools**: Jupyter Notebook, Streamlit

---

## **Project Structure**


├── data/
│   ├── train.csv      # Training dataset
│   ├── test.csv       # Test dataset
├── src/
│   ├── arima_model.py # ARIMA modeling script
│   ├── ml_models.py   # Machine learning script
│   ├── app.py         # Streamlit application script
├── outputs/
│   ├── plots/         # Generated plots
│   ├── results.csv    # Final predictions
├── requirements.txt   # Required Python packages
└── README.md          # Project documentation

---
## **Dataset Information**

The dataset contains historical sales data for various stores and items.

### **Columns:**
- `date`: The date of the sales record.
- `store`: An identifier for the store.
- `item`: An identifier for the item.
- `sales`: The number of items sold.

### **Key Details:**
- The data spans multiple dates, stores, and items, enabling both time series and feature-based modeling.
- **Train Data**: Contains historical data for model training.
- **Test Data**: Includes data for generating predictions, without sales figures.
---
## **Steps in the Workflow**

### **1. Data Preprocessing**
- Convert the `date` column to `datetime` format for proper handling.
- Filter and aggregate sales data by store and item pairs to create specific time series.
- Add additional features:
  - **`month`**: Extracted from the date for seasonal trends.
  - **`day`**: Extracted to capture daily variations.
  - **`weekday`**: Encoded as numerical values (e.g., Monday = 0).

### **2. Time Series Analysis**
- Aggregate sales data by the date to ensure time series continuity.
- Use the **ARIMA model** for forecasting:
  - Parameters `(p, d, q)` are tuned for the dataset.
  - Plot trends and forecasted values.
  - Include confidence intervals for visual clarity.

### **3. Machine Learning Models**
- Downsample training data for computational efficiency.
- Train **Random Forest** and **Gradient Boosting** regressors:
  - **Features**: `store`, `item`, `month`, `day`, `weekday`.
  - Hyperparameter adjustments to optimize performance for memory-constrained environments.
- Ensemble predictions:
  - Combine predictions from both models for better accuracy.

### **4. Streamlit Application**
- Built an interactive **Streamlit** app to:
  - Visualize time series trends for selected store-item pairs.
  - Display model predictions and performance metrics.
  - Allow user interaction to analyze specific combinations of stores and items.

---
## **Results and Metrics**

### **ARIMA Results:**
- Optimized parameters `(p, d, q)` based on ACF and PACF plots.
- Visualization of forecasted values with confidence intervals.

### **Machine Learning Model Metrics:**
- **Random Forest MSE**: `_value_`
- **Gradient Boosting MSE**: `_value_`
- **Ensemble MSE**: `_value_`

The ensemble approach demonstrated improved performance by leveraging the strengths of both Random Forest and Gradient Boosting models.

---

## **Acknowledgments**

Gratitude to:

- Python libraries like `pandas`, `statsmodels`, `scikit-learn`, and `Streamlit` for simplifying development.
- Open datasets that provided the foundation for this project.
- The open-source community for tools and resources that made this project possible.

