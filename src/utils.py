import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_preprocess_data(csv_filename='data.csv'):
    csv_path = os.path.join('data', csv_filename)
        
    data = pd.read_csv(csv_path)
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    # Extraction dynamique des colonnes de caractéristiques
    feature_headers = [col for col in data.columns if col not in ['id', 'diagnosis', 'Unnamed: 32']]
    
    X_raw = data[feature_headers].values
    y = data['diagnosis'].values
    
    return X_raw, y, feature_headers, data

def standardize_features(X_raw):

    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0)
    std[std == 0] = 1  
    
    X_standardized = (X_raw - mean) / std
    return X_standardized

def generate_plots(data, X_train, y_train, feature_headers):

    os.makedirs('figures', exist_ok=True)
    
    # Figure 1: Histogrammes
    plt.figure(figsize=(10, 8))
    data[feature_headers[:4]].hist(bins=20, figsize=(10, 8), color='teal', edgecolor='black')
    plt.suptitle("Feature Distributions (Checking for Normality)", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join('figures', '1_feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Nuage de points (Scatter Plot)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color='tab:blue', label='Benign (0)', s=25, alpha=0.7)
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color='tab:orange', label='Malignant (1)', s=25, alpha=0.7)
    plt.xlabel(feature_headers[0], fontsize=11)
    plt.ylabel(feature_headers[1], fontsize=11)
    plt.title('Breast Cancer Visual Distribution (First 2 Features)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join('figures', '2_class_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()