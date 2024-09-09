import pandas as pd
import matplotlib.pyplot as plt
import os

# Set file paths using os.path
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, 'data', 'predicted_data.csv')
output_path = os.path.join(base_path, 'graphs', 'actual_vs_predicted_pod_1.png')

# Load your data
data = pd.read_csv(data_path)

# Print column names to verify
print("Columns in the DataFrame:", data.columns)

# Filter data for Pod 1
pod_1_data = data[data['pod_name'] == 'pod_1']

# Adjust column names based on actual names
actual_column = 'actual_replicas'  # Update this
predicted_column = 'predicted_replicas'  # Update this

# Plot Actual vs. Predicted
plt.figure(figsize=(12, 6))
plt.plot(pod_1_data['timestamp'], pod_1_data[actual_column], label='Actual Replicas', marker='o')
plt.plot(pod_1_data['timestamp'], pod_1_data[predicted_column], label='Predicted Replicas', marker='x')
plt.xlabel('Timestamp')
plt.ylabel('Number of Replicas')
plt.title('Actual vs. Predicted Replica Counts for Pod 1')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_path)
plt.show()
