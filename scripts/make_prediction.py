import pandas as pd
from joblib import load

def load_model(model_path):
    return load(model_path)

def make_predictions(model, data):
    features = data[['cpu_usage', 'memory_usage', 'cpu_memory_ratio']]
    predictions = model.predict(features)
    data['predicted_replicas'] = predictions
    return data

# Example usage
if __name__ == "__main__":
    from prepare_input_data import preprocess_real_time_data
    from k8s_interaction import get_pod_metrics

    # Load the model
    model = load_model("../models/model.pkl")

    # Get real-time pod metrics and preprocess them
    pod_metrics = get_pod_metrics()
    processed_data = preprocess_real_time_data(pod_metrics)

    # Make predictions
    prediction_results = make_predictions(model, processed_data)
    print(prediction_results)
