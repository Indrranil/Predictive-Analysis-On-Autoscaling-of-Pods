import pandas as pd
from datetime import datetime

def preprocess_real_time_data(pod_metrics):
    processed_data = []

    for pod in pod_metrics['items']:
        pod_name = pod['metadata']['name']
        containers = pod['containers']

        for container in containers:
            cpu_usage = container['usage']['cpu']
            memory_usage = container['usage']['memory']

            # Convert CPU usage to millicores (assumes value is in 'n' nanocores)
            cpu_numeric = int(cpu_usage.replace('n', '')) / 1e6

            # Convert memory usage to MiB
            if 'Ki' in memory_usage:
                memory_numeric = float(memory_usage.replace('Ki', '')) / 1024  # Convert KiB to MiB
            elif 'Mi' in memory_usage:
                memory_numeric = float(memory_usage.replace('Mi', ''))  # Already in MiB
            else:
                raise ValueError(f"Unexpected memory unit in usage: {memory_usage}")

            # CPU/Memory ratio
            cpu_memory_ratio = cpu_numeric / memory_numeric if memory_numeric != 0 else 0

            processed_data.append({
                'pod_name': pod_name,
                'cpu_usage': cpu_numeric,
                'memory_usage': memory_numeric,
                'cpu_memory_ratio': cpu_memory_ratio
            })

    # Convert the list of dictionaries into a pandas DataFrame
    processed_data_df = pd.DataFrame(processed_data)

    return processed_data_df

# Example usage
if __name__ == "__main__":
    from k8s_interaction import get_pod_metrics
    pod_metrics = get_pod_metrics()
    processed_data_df = preprocess_real_time_data(pod_metrics)
    print(processed_data_df)
