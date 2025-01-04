import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_features(df, windows=[12, 24, 128]):
    result_df = df.copy()
    feature_dfs = []
    
    for column in df.columns:
        if column == 'nat_demand':
            for window in windows:
                window_features = pd.DataFrame({
                    f"{column}_lag_{window}": df[column].shift(window),
                    f"{column}_ma_mean{window}": df[column].shift(1).rolling(window, min_periods=1).mean(),
                    f"{column}_std_std{window}": df[column].shift(1).rolling(window, min_periods=1).std(),
                    f"{column}_ewm_std{window}": df[column].shift(1).ewm(span=window, min_periods=1).std(),
                    f"{column}_ewm_mean{window}": df[column].shift(1).ewm(span=window, min_periods=1).mean()
                })
                feature_dfs.append(window_features)
                
        elif column not in ['datetime', 'holiday', 'school', 'Holiday_ID', 'nat_demand']:
            for window in windows:
                series_shifted = df[column].shift(1)
                rolling = series_shifted.rolling(window, min_periods=1)
                ewm = series_shifted.ewm(span=window, min_periods=1)
                
                rolling_min = rolling.min()
                rolling_max = rolling.max()
                
                window_features = pd.DataFrame({
                    f"{column}_lag_{window}": df[column].shift(window),
                    f"{column}_ma_mean{window}": rolling.mean(),
                    f"{column}_std_std{window}": rolling.std(),
                    f"{column}_ewm_std{window}": ewm.std(),
                    f"{column}_ewm_mean{window}": ewm.mean(),
                    f"{column}_min_max{window}": (series_shifted - rolling_min) / (rolling_max - rolling_min),
                    f"{column}_median{window}": rolling.median(),
                    f"{column}_skew{window}": rolling.skew(),
                    f"{column}_kurt{window}": rolling.kurt(),
                    f"{column}_p50{window}": rolling.quantile(0.5)
                })
                feature_dfs.append(window_features)
    
    if feature_dfs:
        all_features = pd.concat([result_df] + feature_dfs, axis=1)
        return all_features
    
    return result_df

def prepare_data(df, train_cutoff='2019-01-01', return_dfs=False):

    df = df.copy()
    
    if df['datetime'].dtype == 'object':
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    train_df = df[df['datetime'] < train_cutoff].copy()
    test_df = df[df['datetime'] >= train_cutoff].copy()
    
    print("Creating features for training data...")
    train_df = create_features(train_df)
    print("Creating features for test data...")
    test_df = create_features(test_df)
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    
    X_train = train_df.drop(columns=['nat_demand', 'datetime'])
    X_test = test_df.drop(columns=['nat_demand', 'datetime'])
    y_train = train_df['nat_demand']
    y_test = test_df['nat_demand']

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if return_dfs:
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns, train_df, test_df
    return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns