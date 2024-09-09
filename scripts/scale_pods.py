from kubernetes import client, config

def scale_pod(namespace, deployment_name, replicas):
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()

    # Ensure replicas is not negative
    replicas = max(0, replicas)

    # Retrieve the latest version of the deployment
    try:
        deployment = apps_v1.read_namespaced_deployment(name=deployment_name, namespace=namespace)

        # Update the replica count
        deployment.spec.replicas = replicas

        # Attempt to patch the deployment
        response = apps_v1.patch_namespaced_deployment(
            name=deployment_name,
            namespace=namespace,
            body=deployment
        )
        print(f"Scaled {deployment_name} to {replicas} replicas")

    except client.exceptions.ApiException as e:
        # Handle API exceptions, such as conflicts
        if e.status == 409:
            print(f"Conflict error occurred while scaling {deployment_name}. Retrying...")
            # Optionally, you could add logic here to retry the operation
        else:
            print(f"An error occurred: {e}")

# Example usage
if __name__ == "__main__":
    from make_prediction import make_predictions, load_model
    from prepare_input_data import preprocess_real_time_data
    from k8s_interaction import get_pod_metrics

    # Load the model
    model = load_model("../models/model.pkl")

    # Get real-time pod metrics and preprocess them
    pod_metrics = get_pod_metrics()
    processed_data = preprocess_real_time_data(pod_metrics)

    # Make predictions
    prediction_results = make_predictions(model, processed_data)

    # Scale the pods according to the predictions
    namespace = "default"  # Update with your namespace
    deployment_name = "nginx"  # Your deployment name

    for index, row in prediction_results.iterrows():
        # Ensure predicted replicas is non-negative
        replicas = max(0, int(row['predicted_replicas']))
        # Use the fixed deployment name instead of the pod name
        scale_pod(namespace, deployment_name, replicas)
