import pandas as pd
import numpy as np
import datetime

# Parameters
num_pods = 10
time_interval_minutes = 10
duration_hours = 72  # 3 days

# Generate timestamps
timestamps = pd.date_range(start=datetime.datetime.now(), periods=duration_hours*6, freq=f'{time_interval_minutes}T')

# Generate synthetic CPU and memory usage data
data = {
    'timestamp': np.tile(timestamps, num_pods),
    'pod_name': np.repeat([f'pod_{i+1}' for i in range(num_pods)], len(timestamps)),
    'cpu_usage': np.random.uniform(low=0.1, high=1.0, size=len(timestamps) * num_pods),  # CPU usage between 0.1 and 1.0 cores
    'memory_usage': np.random.uniform(low=100, high=1000, size=len(timestamps) * num_pods)  # Memory usage between 100MB and 1000MB
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate derived features
df['cpu_memory_ratio'] = df['cpu_usage'] / df['memory_usage']

# Simulate target metric (e.g., number of replicas needed)
df['target_metric'] = np.random.randint(low=1, high=5, size=len(df))

# Preprocess the data (handle missing values and normalization)
df.fillna(method='ffill', inplace=True)
df['cpu_usage'] = (df['cpu_usage'] - df['cpu_usage'].mean()) / df['cpu_usage'].std()
df['memory_usage'] = (df['memory_usage'] - df['memory_usage'].mean()) / df['memory_usage'].std()
df['cpu_memory_ratio'] = (df['cpu_memory_ratio'] - df['cpu_memory_ratio'].mean()) / df['cpu_memory_ratio'].std()

# Save to CSV
df.to_csv('synthetic_pod_usage_data.csv', index=False)

print("Synthetic data generated and saved to 'synthetic_pod_usage_data.csv'.")
