# Time series Model comparison for residential electricity profile demand prediction
# Residential Electricity Demand Forecasting Using Deep Learning

## Overview

This project presents a comprehensive approach to forecasting residential electricity demand leveraging state-of-the-art deep learning architectures. Accurate load forecasting is critical for optimizing energy distribution, reducing waste, and enabling smarter community energy systems. The study benchmarks traditional recurrent models (LSTM, CNN) against a novel Transformer-based architecture designed to handle time-series data efficiently and scalably.

Developed as part of a National Science Foundation Research Internship at Florida Atlantic University, this work demonstrates significant improvements in prediction accuracy and model scalability, with potential applications in smart grid energy management.

---

## Key Contributions

- **Custom Transformer Architecture:** Developed a novel Transformer model tailored for electricity demand forecasting, incorporating dropout regularization and multi-head self-attention mechanisms.  
- **Performance Gains:** Achieved approximately 82% reduction in Root Mean Squared Error (RMSE) compared to baseline LSTM and CNN models.  
- **Robust Data Processing Pipeline:** Designed preprocessing steps to clean, normalize, and prepare large-scale, noisy time-series datasets using MinMaxScaler and StandardScaler.  
- **Comprehensive Evaluation:** Employed multiple metrics (RMSE, MAE, MAPE) to assess model performance across training and validation datasets.  
- **Visualization & Reporting:** Leveraged Matplotlib to produce detailed visualizations, including loss curves and forecast vs. actual comparisons, facilitating interpretability.

---

## Table of Contents

- [Installation](#installation)  
- [Data](#data)  
- [Methodology](#methodology)  
- [Usage](#usage)  
- [Evaluation Metrics](#evaluation-metrics)  
- [Results](#results)  
- [Future Work](#future-work)  
- [References](#references)  
- [License](#license)

---

## Installation

### Prerequisites

- Python 3.7 or higher  
- Git  
- Recommended: virtual environment (venv, conda, or similar)

### Setup Steps

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```
2. **Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

###Data###

The dataset consists of residential electricity consumption time-series, sourced from [Kaggle]. Data preprocessing includes handling missing values, outlier detection, and normalization using MinMaxScaler and StandardScaler to improve model convergence.

Data: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

###Methodology###

**Models Implemented**

-LSTM (Long Short-Term Memory): Baseline recurrent neural network for sequence prediction.

-CNN (Convolutional Neural Network): Adapted for time-series feature extraction and forecasting.

-Custom Transformer: Novel architecture integrating multi-head attention and dropout for robust, scalable forecasting.

###Training Details###

-Loss function: Mean Squared Error (MSE)

-Optimizer: AdamW with weight decay regularization

-Hyperparameters: Configurable learning rate, dropout rates, number of attention heads and layers

-Training performed on CPU Emvionment

###Usage###

Open and run the Jupyter notebook electricity_demand_forecasting.ipynb or use Google Colab for an interactive environment.

Steps include:

Data loading and preprocessing

Model training for LSTM, CNN, and Transformer

Evaluation and visualization of results

Hyperparameter tuning examples

###Evaluation Metrics###
Root Mean Squared Error (RMSE): Measures average prediction error magnitude.

Mean Absolute Error (MAE): Average of absolute differences between predictions and actual values.

Mean Absolute Percentage Error (MAPE): Relative error expressed as a percentage, useful for interpretability.

###Results###

Custom Transformer significantly outperformed baseline models, reducing RMSE by ~82%.

Visualizations illustrate convergence speed, prediction accuracy, and error distribution.

Results suggest Transformer’s superior capability in capturing temporal dependencies in electricity demand.

**Screenshots: **
![image](https://github.com/user-attachments/assets/6199a3cf-d2ec-4676-8294-a5b4c3e03005)

![image](https://github.com/user-attachments/assets/282cde82-ee0c-4cfa-92ed-7a90d4e16919)

![image](https://github.com/user-attachments/assets/f40366e7-7f98-439f-8d5b-f9007a4f30f9)

![image](https://github.com/user-attachments/assets/56d6d781-d84c-43f8-82ad-0890b3002d93)

![image](https://github.com/user-attachments/assets/6bc0b892-5ff4-4ed9-ad3a-73402f1d4ba2)

![image](https://github.com/user-attachments/assets/0b4c940b-43e8-412d-b0d4-50de7231b2ad)

![image](https://github.com/user-attachments/assets/ba9d26a6-9327-4fa1-b01f-72ddbcfd7e10)

###Future Work###

Extend model to incorporate exogenous variables such as weather, occupancy, or economic factors.

Deploy forecasting model as an API for real-time prediction in smart grid applications.

Investigate model interpretability using attention weights analysis.

Expand dataset to multiple regions for generalized forecasting.

###References###

Vaswani, Ashish, et al. “Attention Is All You Need.” NeurIPS, 2017.

Kingma, Diederik P., and Jimmy Ba. “Adam: A Method for Stochastic Optimization.” ICLR, 2015.

Keras Documentation

PyTorch Documentation

