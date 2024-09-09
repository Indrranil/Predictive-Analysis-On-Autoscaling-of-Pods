import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import load_config, load_raw_data, preprocess_data, split_data

def evaluate_model():
    """Evaluate the trained model."""
    # Load configuration file
    config = load_config('config/config.yaml')

    # Load and preprocess data
    data = load_raw_data(config['data']['processed_data_path'])
    processed_data = preprocess_data(data)

    # Extract features and target variable
    X = processed_data[['cpu_usage', 'memory_usage', 'cpu_memory_ratio']]
    y = processed_data['target_metric']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Load the trained model
    with open('models/model.pkl', 'rb') as f:
        model = joblib.load(f)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    evaluate_model()
