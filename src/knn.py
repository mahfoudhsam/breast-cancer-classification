import numpy as np

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        # 1. Compute the distance between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
    
        # 2. Get the closest k indices
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 3. Manual Majority Vote (Replaces Counter)
        label_counts = {}
        for label in k_nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        
        # 4. Find the label with the maximum count
        most_common_label = None
        max_count = -1
        
        for label, count in label_counts.items():
            if count > max_count:
                max_count = count
                most_common_label = label
                
        return most_common_label