# src/data_preprocessing.py
import pandas as pd

def preprocess_data(df):
    # Implement preprocessing steps like handling missing values, normalization, etc.
    df.fillna(method='ffill', inplace=True)
    df['cpu_usage'] = (df['cpu_usage'] - df['cpu_usage'].mean()) / df['cpu_usage'].std()
    df['memory_usage'] = (df['memory_usage'] - df['memory_usage'].mean()) / df['memory_usage'].std()
    df['cpu_memory_ratio'] = (df['cpu_memory_ratio'] - df['cpu_memory_ratio'].mean()) / df['cpu_memory_ratio'].std()
    return df

def save_processed_data(df, path):
    df.to_csv(path, index=False)
