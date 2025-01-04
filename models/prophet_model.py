from prophet import Prophet
import pandas as pd

def train_prophet(train_df, test_df):
    # Prepare training data for Prophet
    df_train_prophet = train_df.reset_index().rename(
        columns={'datetime': 'ds', 'nat_demand': 'y'}
    )
    model = Prophet()
    
    # Add all columns except datetime and nat_demand as regressors
    regressor_columns = [col for col in train_df.columns 
                        if col not in ['datetime', 'nat_demand']]
    
    for column in regressor_columns:
        model.add_regressor(column)
    model.fit(df_train_prophet)
    
    df_test_prophet = test_df.reset_index().rename(
        columns={'datetime': 'ds', 'nat_demand': 'y'}
    )
    
    predictions = model.predict(df_test_prophet)
    
    return model, predictions['yhat'].values