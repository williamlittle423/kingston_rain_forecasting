# data_preparation.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(file_pattern='climate-hourly_{}.csv', num_files=7):
    """
    Load multiple CSV files, concatenate them, and preprocess the climate data.

    Parameters:
        file_pattern (str): The pattern for file names, with '{}' as placeholder for index.
        num_files (int): Number of files to load.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
    """
    # Read multiple CSVs and concatenate
    df_list = []
    for i in range(num_files):
        file_name = file_pattern.format(i)
        df_list.append(pd.read_csv(file_name))
    climate_data = pd.concat(df_list, ignore_index=True)
    
    # Create a proper datetime index
    climate_data['LOCAL_DATETIME'] = pd.to_datetime(
        climate_data[['LOCAL_YEAR','LOCAL_MONTH','LOCAL_DAY','LOCAL_HOUR']]  
        .rename(columns={
            'LOCAL_YEAR': 'year',
            'LOCAL_MONTH': 'month',
            'LOCAL_DAY': 'day',
            'LOCAL_HOUR': 'hour'
        })
    )
    climate_data.set_index('LOCAL_DATETIME', inplace=True)
    climate_data.sort_index(inplace=True)
    
    # Resample hourly (so missing hours appear as NaN rows)
    climate_data = climate_data.resample('h').asfreq()
    
    # Filter the columns you need
    filtered_climate_data = climate_data[[
        'TEMP', 'DEW_POINT_TEMP', 'RELATIVE_HUMIDITY',
        'STATION_PRESSURE', 'WIND_SPEED', 'WIND_DIRECTION',
        'PRECIP_AMOUNT'
    ]].copy()
    
    # Drop rows with missing values
    filtered_climate_data.dropna(subset=[
        'TEMP','DEW_POINT_TEMP','RELATIVE_HUMIDITY',
        'STATION_PRESSURE','WIND_SPEED','WIND_DIRECTION',
        'PRECIP_AMOUNT'
    ], inplace=True)
    
    # Shift precip to the next hour
    filtered_climate_data['PRECIP_AMOUNT_NEXT_HOUR'] = filtered_climate_data['PRECIP_AMOUNT'].shift(-1)
    
    # Drop any rows where next-hour precip is NaN (last row(s))
    filtered_climate_data.dropna(subset=['PRECIP_AMOUNT_NEXT_HOUR'], inplace=True)
    
    # Convert next-hour precipitation to a binary label
    filtered_climate_data['RAIN_NEXT_HOUR'] = (
        filtered_climate_data['PRECIP_AMOUNT_NEXT_HOUR'] > 0
    ).astype(int)
    
    # Prepare features (X) and target (y)
    X = filtered_climate_data[[
        'TEMP','DEW_POINT_TEMP','RELATIVE_HUMIDITY',
        'STATION_PRESSURE','WIND_SPEED','WIND_DIRECTION', 'PRECIP_AMOUNT'
    ]].to_numpy()
    
    y = filtered_climate_data['RAIN_NEXT_HOUR'].to_numpy()
    
    return X, y

def split_and_scale(X, y, test_size=0.2):
    """
    Split the data into training and testing sets based on time, and scale the features.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        X_train_scaled (np.ndarray): Scaled training features.
        X_test_scaled (np.ndarray): Scaled testing features.
        y_train (np.ndarray): Training labels.
        y_test (np.ndarray): Testing labels.
        scaler (StandardScaler): Fitted scaler object.
    """
    # Time-based split (80% train, 20% test) — no shuffle
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Scale the training data, then transform test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def handle_class_imbalance(X, y, method='smote'):
    """
    Handle class imbalance using specified method.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target labels.
        method (str): Method to handle imbalance ('smote', 'none').

    Returns:
        X_res (np.ndarray): Resampled feature matrix.
        y_res (np.ndarray): Resampled target labels.
    """
    if method == 'smote':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    elif method == 'none':
        return X, y
    else:
        raise ValueError("Unsupported imbalance handling method.")
