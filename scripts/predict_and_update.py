import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import yaml
import pandas as pd
import subprocess
from src.data_preprocessing import preprocess_data
from src.feature_engineering import create_features

def load_model(model_path):
    return joblib.load(model_path)

def get_model_prediction(model, input_data):
    return model.predict(input_data)

def update_deployment_yaml(deployment_yaml_path, num_replicas):
    with open(deployment_yaml_path, 'r') as file:
        deployment = yaml.safe_load(file)

    deployment['spec']['replicas'] = num_replicas

    with open(deployment_yaml_path, 'w') as file:
        yaml.safe_dump(deployment, file)

def apply_deployment(deployment_yaml_path):
    subprocess.run(["kubectl", "apply", "-f", deployment_yaml_path])

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '../config/config.yaml')

    if not os.path.isfile(config_path):
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_path = os.path.join(script_dir, '../', config['model']['save_path'])
    deployment_yaml_path = os.path.join(script_dir, '../', config['deployment']['yaml_path'])
    raw_data_path = os.path.join(script_dir, '../', config['data']['raw_data_path'])

    model = load_model(model_path)

    # Load and preprocess the raw data
    df = pd.read_csv(raw_data_path)
    df = preprocess_data(df)
    df = create_features(df)

    # Using the last row of the data as the sample data for prediction
    sample_data = df[['cpu_usage', 'memory_usage', 'cpu_memory_ratio']].tail(1)

    num_replicas = int(get_model_prediction(model, sample_data)[0])
    print(f"Prediction: {num_replicas}")

    update_deployment_yaml(deployment_yaml_path, num_replicas)
    apply_deployment(deployment_yaml_path)

if __name__ == '__main__':
    main()
