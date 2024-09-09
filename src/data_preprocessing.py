import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def load_config(file_path):
    """Load configuration from the specified YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_raw_data(file_path):
    """Load raw data from the specified CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the raw data and return the processed DataFrame."""
    df = df.dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', inplace=True)

    df['cpu_usage'] = (df['cpu_usage'] - df['cpu_usage'].mean()) / df['cpu_usage'].std()

    df['memory_usage'] = (df['memory_usage'] - df['memory_usage'].mean()) / df['memory_usage'].std()

    return df

def save_processed_data(df, file_path):
    """Save the processed data to the specified CSV file."""
    df.to_csv(file_path, index=False)

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
