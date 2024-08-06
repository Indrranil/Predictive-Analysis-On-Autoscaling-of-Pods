import os
import sys
import joblib
import yaml
import pandas as pd
from sklearn.impute import SimpleImputer

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features

def main():
    # Get the absolute path to the config file
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get the absolute path to the model file
    model_save_path = os.path.join(os.path.dirname(__file__), '..', config['model']['save_path'])

    # Load model
    model = joblib.load(model_save_path)

    # Load data from CSV
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'processed_metrics.csv')
    sample_data = pd.read_csv(data_path)

    # Ensure required columns exist and create any necessary columns
    required_columns = ['cpu_usage', 'memory_usage']  # Update with actual required columns
    for col in required_columns:
        if col not in sample_data:
            raise ValueError(f"Required column '{col}' is missing from the data")

    # Create cpu_memory_ratio column if needed
    if 'cpu_memory_ratio' not in sample_data:
        sample_data['cpu_memory_ratio'] = sample_data['cpu_usage'] / sample_data['memory_usage']

    # Handle missing values
    sample_data = sample_data.dropna()  # or use imputation as needed

    # Preprocess and create features
    sample_data = preprocess_data(sample_data)
    sample_data = create_features(sample_data)

    # Ensure the features match the model's expectations
    model_features = model.feature_names_in_  # Assuming the model exposes this attribute
    missing_features = set(model_features) - set(sample_data.columns)
    extra_features = set(sample_data.columns) - set(model_features)

    if missing_features:
        raise ValueError(f"Feature mismatch: Missing features: {missing_features}")

    # Remove extra features from the sample_data
    sample_data = sample_data[model_features]

    # Make prediction
    prediction = model.predict(sample_data)
    print(f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    main()
