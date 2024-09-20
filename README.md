# Kubernetes Autoscaler with Predictive Analysis

This project implements a custom autoscaler for Kubernetes that uses machine learning to predict the number of replicas needed for pods based on real-time CPU and memory usage metrics. The autoscaler aims to enhance the standard Horizontal Pod Autoscaler (HPA) by making more informed decisions using predictive analysis.

## Features

- **Predictive Autoscaling**: Uses machine learning (currently linear regression) to predict pod scaling based on historical data.
- **Real-time Data Collection**: Continuously collects CPU and memory usage from the Kubernetes cluster and feeds it into the prediction model.
- **Customizable Metrics**: Allows tracking of various Kubernetes metrics such as `cpu_usage`, `memory_usage`, and `cpu_memory_ratio`.
- **Tested and Mocked Components**: Includes unit tests for core components using `pytest` and mocks for Kubernetes API interactions.

## Directory Structure

```bash
.
├── config                   # Configuration files
├── data                     # Raw and processed data
├── models                   # Trained machine learning models
├── scripts                  # Core Python scripts for autoscaler logic
├── src                      # Source code for data preprocessing and feature engineering
├── tests                    # Unit tests for various components
├── notebooks                # Jupyter notebooks (if any) for exploratory data analysis
└── requirements.txt          # Python dependencies

