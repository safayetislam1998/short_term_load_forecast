# Enhanced Short-Term Load Forecasting using ProphetXGB-CatBoost Architecture

This repository contains the implementation of an enhanced short-term load forecasting system. It combines Facebook Prophet, XGBoost, and CatBoost models with multi-lag feature engineering techniques.

## Description

This project implements a novel hybrid approach for short-term electrical load forecasting, combining the strengths of:

Facebook Prophet for time series decomposition.
XGBoost for handling high-dimensional feature spaces.
CatBoost for meta-learning and ensemble optimization.

The architecture incorporates extensive feature engineering with multi-lag statistical components to enhance prediction accuracy.

## Key Features

- Multi-lag feature engineering expanding 17 base features to 390 engineered features.
- Hybrid model combining Prophet, XGBoost and CatBoost.
- Comprehensive evaluation metrics including RMSE, MAE, MAPE, R² and various deviance measures.
- Robust data preprocessing and validation.
- GPU acceleration support.
- Hyperparameter optimization using Optuna.

## Installation

1. Clone the repository and install the required packages:
    ```bash
    git clone https://github.com/safayetislam1998/short_term_load_forecast.git
    cd short_term_load_forecast
    pip install -r requirements.txt
    ```
2. Run the main script:
    ```bash
    python main.py
    ```

## Research Paper
The methodology and results are detailed in our paper "Enhanced Short-Term Load Forecasting with Multi-Lag Feature Engineering and ProphetXGB-CatBoost Architecture" (under review).
