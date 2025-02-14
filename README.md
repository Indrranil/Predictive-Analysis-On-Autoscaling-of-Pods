# 🚀 Predictive Analysis on Autoscaling of Pods  

## Introduction  
Kubernetes Horizontal Pod Autoscaler (HPA) is **reactive**, scaling only when resource usage crosses predefined thresholds. This approach leads to **delayed scaling decisions** and **resource inefficiencies**.  

This project introduces a **predictive autoscaling model** that leverages **machine learning on historical resource usage data** to proactively determine the optimal number of replicas for a deployment. By **forecasting future CPU and memory consumption**, this model ensures efficient autoscaling while reducing the risk of over-provisioning and under-provisioning.  

## Key Features  
 **Predictive Scaling** – Anticipates resource demands instead of reacting late  
 **Machine Learning Model** – Uses **Linear Regression** to estimate the required number of replicas  
 **Real-time Kubernetes Integration** – Fetches live metrics and adjusts scaling dynamically  
 **Improved Resource Utilization** – Reduces unnecessary replica allocation and cost overhead  
 **Minimal Latency** – Quick decision-making for optimal autoscaling  

---

## Flow
Below is the flow diagram illustrating how **metrics are collected, processed, and used for predictive scaling**:  

![Flow](/flow.png)  
