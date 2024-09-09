from kubernetes import client, config
from kubernetes.client.rest import ApiException

def get_pod_metrics():
    config.load_kube_config()
    try:
        metrics_v1 = client.CustomObjectsApi()
        pod_metrics = metrics_v1.list_cluster_custom_object(
            "metrics.k8s.io", "v1beta1", "pods"
        )
        if 'items' not in pod_metrics or len(pod_metrics['items']) == 0:
            print("No pod metrics found. Ensure that the Metrics Server is running and pods are deployed.")
            return None
        return pod_metrics
    except ApiException as e:
        print(f"Error fetching pod metrics: {e}")
        return None

# Example usage
if __name__ == "__main__":
    pod_metrics = get_pod_metrics()
    if pod_metrics:
        print("Successfully fetched pod metrics.")
    else:
        print("Failed to fetch pod metrics.")
