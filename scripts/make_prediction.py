# scripts/make_prediction.py
import joblib
import yaml
import pandas as pd
from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features

def main():
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    model_save_path = config['model']['save_path']

    # Load model
    model = joblib.load(model_save_path)

    # Sample input data
    sample_data = pd.DataFrame({
        'cpu_usage': [0.5],
        'memory_usage': [0.7],
    })

    # Preprocess and create features
    sample_data = preprocess_data(sample_data)
    sample_data = create_features(sample_data)

    # Make prediction
    prediction = model.predict(sample_data)
    print(f'Prediction: {prediction[0]}')

if __name__ == '__main__':
    main()
