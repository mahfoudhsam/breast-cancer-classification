import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=10000):
        self.alpha = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None

    def _sigmoid(self, z):
        # Sécurité pour éviter les overflows d'exponentielle
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y, w, b):
        cost_sum = 0
        m = y.shape[0]
        for i in range(m):
            z = np.dot(w, X[i]) + b
            g = self._sigmoid(z)
            # Ajout d'un epsilon (1e-15) pour éviter log(0)
            cost_sum += -y[i] * np.log(g + 1e-15) - (1 - y[i]) * np.log(1 - g + 1e-15)
        return (1 / m) * cost_sum

    def _gradient_function(self, X, y, w, b):
        m, n = X.shape
        grad_w = np.zeros(n)
        grad_b = 0
        
        for i in range(m):
            z = np.dot(w, X[i]) + b
            g = self._sigmoid(z)
            
            grad_b += (g - y[i])
            for j in range(n):
                grad_w[j] += (g - y[i]) * X[i, j]
                
        grad_b = (1 / m) * grad_b
        grad_w = (1 / m) * grad_w
        return grad_b, grad_w

    def fit(self, X, y):
        """Entraîne le modèle en ajustant les poids w et le biais b via Descente de Gradient."""
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0
        
        print(f"LogisticRegression ({self.iterations} itérations)...")
        for i in range(self.iterations):
            grad_b, grad_w = self._gradient_function(X, y, self.w, self.b)
            
            # Mise à jour des paramètres
            self.w = self.w - self.alpha * grad_w
            self.b = self.b - self.alpha * grad_b
            
            if i % 1000 == 0:
                current_cost = self._cost_function(X, y, self.w, self.b)
                print(f"  Iteration {i:5d} | cost : {current_cost:.4f}")
        
    def predict(self, X):
        """Prédit les classes (0 ou 1) pour un ensemble de données."""
        if self.w is None or self.b is None:
            raise ValueError("Le modèle doit être entraîné avec la méthode .fit() avant de faire des prédictions.")
            
        m, n = X.shape
        preds = np.zeros(m)
        for i in range(m):
            z = np.dot(self.w, X[i]) + self.b 
            g = self._sigmoid(z)
            preds[i] = 1 if g >= 0.5 else 0
        return preds