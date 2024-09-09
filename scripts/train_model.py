import os
import sys
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data_preprocessing import load_config, load_raw_data, preprocess_data

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(data):
    """Train the model on the given data."""
    # Extract features and target variable
    X = data[['cpu_usage', 'memory_usage', 'cpu_memory_ratio']]
    y = data['target_metric']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

    # Save the trained model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    # Construct the path to the configuration file
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

    # Load configuration file
    config = load_config(config_path)

    # Load and preprocess data
    data = load_raw_data(config['data']['processed_data_path'])
    processed_data = preprocess_data(data)

    # Train and save the model
    train_model(processed_data)
