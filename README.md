#  Breast Cancer Classification From Scratch

A machine learning project implementing classical ML algorithms completely from scratch using NumPy, without relying on scikit-learn.

# Project Overview

This project aims to classify breast cancer tumors as malignant or benign using handcrafted machine learning algorithms and a modular execution pipeline. 

# Implemented Algorithms:
 **Logistic Regression:** Optimized via Gradient Descent using binary cross-entropy loss.
 **Decision Tree:** Built recursively using Entropy and Information Gain metrics, featuring a custom manual label-counting node logic.
 **K-Nearest Neighbors (KNN):** Implemented using manual Euclidean distance calculations and a custom dictionary-based majority voting mechanism.


## 📂 Project Structure
```text
breast_cancer/
│
├── data/
│   └── data.csv                 
│
├── figures/                    
│
├── src/
│   ├── run.py                  
│   ├── logistic_regression.py   
│   ├── Decision_tree.py         
│   ├── knn.py                  
│   └── utils.py                 

Logistic Regression
Optimization: Iterative Gradient Descent (10,000 iterations).

Activation: Sigmoid function mapping outputs between 0 and 1.

Loss Tracking: Monitored step-by-step via Binary Cross-Entropy Cost.

 Decision Tree
Splitting Criterion: Evaluates feature cutoffs based on highest Information Gain.

Leaf Resolution: Custom dictionary-based frequency mapping replacing high-level collection libraries.

 K-Nearest Neighbors (KNN)
Distance Metric: Vectorized Euclidean distance calculation.
Classification: Manual majority vote tracking of the K closest neighbors.


Evaluation & PerformanceAll models are evaluated using a custom accuracy metric function calculating exact matching ratios over the dataset matrix:$$\text{Accuracy} = \frac{\sum (y_{\text{true}} == y_{\text{pred}})}{N}$$
