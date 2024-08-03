# scripts/evaluate.py
import sys
import os

# Append the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import yaml
import pandas as pd
from sklearn.metrics import mean_squared_error
from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features

def main():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../config/config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    raw_data_path = os.path.join(script_dir, '../', config['data']['raw_data_path'])
    model_save_path = os.path.join(script_dir, '../', config['model']['save_path'])

    # Load and preprocess data
    df = pd.read_csv(raw_data_path)
    df = preprocess_data(df)
    df = create_features(df)

    # Split data into features and target
    feature_columns = ['cpu_usage', 'memory_usage', 'cpu_memory_ratio']
    target_column = 'target_metric'  # Adjust this to match your dataset

    X = df[feature_columns]
    y = df[target_column]

    # Load model
    model = joblib.load(model_save_path)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Print predictions
    print("Predictions:")
    print(y_pred)

if __name__ == '__main__':
    main()
