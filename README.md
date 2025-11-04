# kmeans_clustering

# Bank Customer Segmentation

A machine learning project that clusters bank customers into distinct segments based on their credit card usage patterns and financial behavior.

## Overview

This project uses K-Means clustering to segment bank customers into three distinct groups:
- **Conservative Customers** - Low usage, minimal spending
- **Active Spenders** - High-value frequent users with good payment behavior  
- **Credit Dependents** - High balances with poor payment behavior (Risk Zone)

## Project Structure

- `clustering-1.ipynb` - Jupyter notebook with data analysis, preprocessing, and model training
- `streamlit_app.py` - Interactive web application for customer classification
- `kmeans_model.joblib` - Trained clustering model
- `scaler.joblib` - Feature scaler for data preprocessing

## Features

- **Data Analysis**: Comprehensive EDA and outlier detection
- **Clustering**: K-Means algorithm with optimal cluster selection
- **Visualization**: PCA for 2D cluster visualization
- **Web Interface**: Streamlit app for real-time customer classification

## Installation

1. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib pillow
```

2. Download the dataset:
```python
import kagglehub
path = kagglehub.dataset_download("arjunbhasin2013/ccdata")
```

## Usage

### Model Training
Run the Jupyter notebook `clustering-1.ipynb` to:
- Preprocess the credit card data
- Train the K-Means model
- Evaluate cluster performance
- Generate visualizations

### Web Application
Launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```

The web app allows you to:
- Input customer financial data
- Get real-time cluster predictions
- View segment descriptions and recommendations

## Data Features

Key features used for clustering:
- **Core Features**: BALANCE, PURCHASES, ONEOFF_PURCHASES, INSTALLMENTS_PURCHASES
- **Credit Usage**: CREDIT_LIMIT, CASH_ADVANCE
- **Payment Behavior**: PAYMENTS, MINIMUM_PAYMENTS
- **Transaction Frequency**: Various frequency metrics

## Model Details

- **Algorithm**: K-Means Clustering (k=3)
- **Preprocessing**: StandardScaler for feature normalization
- **Evaluation**: Silhouette Score and Elbow Method
- **Dimensionality Reduction**: PCA for visualization

## Results

The model identifies three customer segments with distinct financial behaviors, enabling targeted marketing strategies and risk assessment for the bank.
