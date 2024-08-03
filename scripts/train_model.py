# scripts/train_model.py
import sys
import os

# Append the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from src.data_preprocessing import preprocess_data, save_processed_data
from src.feature_engineering import create_features

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../config/config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_data_path = os.path.join(script_dir, '../', config['data']['raw_data_path'])
    processed_data_path = os.path.join(script_dir, '../', config['data']['processed_data_path'], 'processed_metrics.csv')
    model_save_path = os.path.join(script_dir, '../', config['model']['save_path'])

    # Load and preprocess data
    df = pd.read_csv(raw_data_path)
    df = preprocess_data(df)
    df = create_features(df)
    save_processed_data(df, processed_data_path)

    # Split data into training and test sets
    feature_columns = ['cpu_usage', 'memory_usage', 'cpu_memory_ratio']
    target_column = 'target_metric'  # Adjust this to match your dataset

    X = df[feature_columns]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, model_save_path)

if __name__ == '__main__':
    main()
