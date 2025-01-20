# Precipitation Forecasting with PyTorch for Kingston, Ontario

![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Project Overview

This repository provides a pipeline for **precipitation forecasting** using a neural network model built with **PyTorch**. It incorporates **hyperparameter optimization** with **Optuna** and addresses class imbalance using **SMOTE** to improve predictive performance.

## Features

- **Data Preparation:** Loads and preprocesses hourly climate data from multiple CSV files.
- **Model Training:** Implements a configurable Multi-Layer Perceptron (MLP) for binary classification (Rain vs. No Rain).
- **Hyperparameter Optimization:** Utilizes Optuna to optimize model hyperparameters for better F1-Score on the "Rain" class.
- **Evaluation:** Provides metrics such as Precision, Recall, F1-Score, and Confusion Matrix.
- **Testing:** Includes a script to evaluate the trained model with predefined hyperparameters.
