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

 ## 📄 Research Paper (Unpublished)

A detailed research paper accompanying this project explores the design, implementation, and evaluation of the predictive autoscaling model. Although the paper is **not formally published yet**, it is available for review:

👉 [Read the Research Paper on Google Docs](https://docs.google.com/document/d/1OGQ3mr1T1aBqd35AwfobJoTvlp1a5l50RgIrw4dkW-g/edit?usp=sharing)

Alternatively, you can [download the PDF version](docs/Predictive Analysis on Autoscaling of Pods .pdf) from this repository.


---

## Flow
Below is the flow diagram illustrating how **metrics are collected, processed, and used for predictive scaling**:  

![Flow](/flow.png)  
