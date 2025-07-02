import os
import pandas as pd
import numpy as np
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def download_data():
    """Download data from Kaggle"""
    # Set up the API
    api = KaggleApi()
    api.authenticate()
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    # Download dataset
    api.dataset_download_files('roustekbio/breast-cancer-csv', 
                             path=os.path.dirname(RAW_DATA_PATH), 
                             unzip=True)
    
    return RAW_DATA_PATH

def load_data():
    """Load and do initial inspection of the data"""
    df = pd.read_csv(RAW_DATA_PATH)
    pd.set_option('display.expand_frame_repr', False)
    print("\n=== Initial Data Preview ===")
    print(df.head())
    print("\n=== Data Shape ===")
    print(df.shape)
    print("\n=== Data Types ===")
    print(df.dtypes)
    return df

def clean_data(df):
    """Clean and preprocess the data"""
    # Convert 'bare_nucleoli' to numeric first
    df['bare_nucleoli'] = pd.to_numeric(df['bare_nucleoli'], errors='coerce')
    
    print("\n=== Missing Values Before Cleaning ===")
    print(df.isna().sum())
    print(f"Shape before cleaning: {df.shape}")
    
    # Drop missing values
    df = df.dropna().reset_index(drop=True)
    
    print("\n=== Missing Values After Cleaning ===")
    print(df.isna().sum())
    print(f"Shape after cleaning: {df.shape}")
    
    return df

# In dataPreprocess_and_FeatureEng_.py

def scale_features(df, target_col):
    """
    Scale features while preserving the target and ID columns.
    
    Args:
        df: Input DataFrame
        target_col: The name of the target column, which should not be scaled.
        
    Returns:
        scaled_df: DataFrame with scaled features
        scaler: Fitted scaler object
    """
    # Identify columns to preserve (target and ID)
    cols_to_preserve = [target_col]
    if 'id' in df.columns:
        cols_to_preserve.append('id')

    # Identify columns to scale (all columns that are not preserved)
    features_to_scale = [col for col in df.columns if col not in cols_to_preserve]
    
    scaler = StandardScaler()
    
    # Create a copy to work with
    scaled_df = df.copy()
    
    # Scale the identified features
    if features_to_scale:
        scaled_df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    
    return scaled_df, scaler

def split_data(df, target_col='class'):
    """
    Split data into train/test sets
    
    Args:
        df: Input DataFrame
        target_col: Column to use as target variable
        
    Returns:
        X_train, X_test, y_train, y_test: Split data
    """
    # Features are all columns except target and id
    X = df.drop(columns=[target_col, 'id'] if 'id' in df.columns else target_col)
    y = df[target_col]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
def save_processed_data(df, path=PROCESSED_DATA_PATH):
    """Save processed data to CSV"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nProcessed data saved to {path}")