import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from utils.preprocessing import prepare_data
from models.xgboost_model import train_xgboost
from models.prophet_model import train_prophet
from models.ensemble import evaluate_predictions
from catboost import CatBoostRegressor

def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)

def main():
    setup_gpu()
    df = pd.read_csv('data/continuous dataset.csv')
    
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, train_df, test_df = prepare_data(df, return_dfs=True)
    feature_names = feature_names.tolist()
    dtest = xgb.DMatrix(X_test_scaled, label=y_test, feature_names=feature_names)
    
    print("Training XGBoost model...")
    xgb_model, xgb_params = train_xgboost(X_train_scaled, X_test_scaled, y_train, y_test, feature_names)
    print("Best XGBoost parameters:", xgb_params)
    
    print("\nTraining Prophet model...")
    prophet_model, prophet_predictions = train_prophet(train_df, test_df)
    
    # Get XGBoost predictions
    xgb_predictions = xgb_model.predict(dtest)
    
    # Create ensemble predictions
    print("\nCreating ensemble predictions...")
    ensemble_predictions = (0.7 * prophet_predictions) + (0.3 * xgb_predictions)
    
    # Train meta-model
    print("\nTraining meta-model...")
    meta_features = np.column_stack((prophet_predictions, xgb_predictions))
    meta_model = CatBoostRegressor(iterations=100, depth=6, learning_rate=0.1, loss_function='RMSE')
    meta_model.fit(meta_features, y_test)
    meta_predictions = meta_model.predict(meta_features)
    
    # Evaluate all models
    print("\nModel Evaluation Results:")
    models = {
        'XGBoost': xgb_predictions,
        'Prophet': prophet_predictions,
        'Simple Ensemble': ensemble_predictions,
        'Meta-Model': meta_predictions
    }
    
    for model_name, predictions in models.items():
        metrics = evaluate_predictions(y_test, predictions)
        print(f"\n{model_name} Results:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"MAPE: {metrics['mape']:.2%}")
        print(f"R²: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()