import numpy as np
from utils import load_and_preprocess_data, standardize_features, generate_plots
from logistic_regression import LogisticRegression
from Decision_tree import DecisionTree
from knn import KNN
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def main():

    X_raw, y_train, feature_headers, raw_dataframe = load_and_preprocess_data('data.csv')
    
    X_train = standardize_features(X_raw)
    print(f"Shape of data : {X_train.shape}")
    
    generate_plots(raw_dataframe, X_train, y_train, feature_headers)
    
    model1 = LogisticRegression(learning_rate=0.01, iterations=500)
    
    model1.fit(X_train, y_train)

    predictions = model1.predict(X_train)
    lr_acc = accuracy(y_train, predictions) * 100

    print(f"\nAccuracy of LogisticRegression : {lr_acc:.2f}%") 

    print("\nDecision Tree")

    model2 = DecisionTree(max_depth=5)
    model2.fit(X_train, y_train)
    dt_predictions = model2.predict(X_train)
    dt_acc = accuracy(y_train, dt_predictions) * 100
    print(f" Accuracy of DecisionTree : {dt_acc:.2f}%")

    print("\n K-Nearest Neighbors ")

    knn_clf = KNN(k=3)
    knn_clf.fit(X_train, y_train)
    
    knn_predictions = knn_clf.predict(X_train)
    knn_acc = accuracy(y_train, knn_predictions) * 100
    print(f"Accuracy of KNN : {knn_acc:.2f}%")



if __name__ == "__main__":
    main()