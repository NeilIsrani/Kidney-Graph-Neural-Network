import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

class AdaptiveKNN:
    def __init__(self, k_range=(1, 10), metric='euclidean'):
        self.k_range = k_range
        self.metric = metric
        self.best_k = None
        self.scaler = StandardScaler()
        self.knn = None
        
    def fit(self, X, y):
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Find best k
        best_score = 0
        for k in range(self.k_range[0], self.k_range[1] + 1):
            knn = KNeighborsClassifier(n_neighbors=k, metric=self.metric)
            knn.fit(X_train, y_train)
            score = knn.score(X_val, y_val)
            
            if score > best_score:
                best_score = score
                self.best_k = k
                self.knn = knn
                
        # Retrain on full data with best k
        self.knn = KNeighborsClassifier(n_neighbors=self.best_k, metric=self.metric)
        self.knn.fit(X_scaled, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.knn.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.knn.predict_proba(X_scaled)

class AdaptiveBayesian:
    def __init__(self, var_smoothing_range=(1e-9, 1e-3)):
        self.var_smoothing_range = var_smoothing_range
        self.best_var_smoothing = None
        self.scaler = StandardScaler()
        self.nb = None
        
    def fit(self, X, y):
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Find best var_smoothing
        best_score = 0
        for var_smoothing in np.logspace(
            np.log10(self.var_smoothing_range[0]),
            np.log10(self.var_smoothing_range[1]),
            num=10
        ):
            nb = GaussianNB(var_smoothing=var_smoothing)
            nb.fit(X_train, y_train)
            score = nb.score(X_val, y_val)
            
            if score > best_score:
                best_score = score
                self.best_var_smoothing = var_smoothing
                self.nb = nb
                
        # Retrain on full data with best var_smoothing
        self.nb = GaussianNB(var_smoothing=self.best_var_smoothing)
        self.nb.fit(X_scaled, y)
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.nb.predict(X_scaled)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.nb.predict_proba(X_scaled)

def load_data():
    # Load gene expression data
    gene_data = pd.read_csv('spot_gene_expression.csv')
    ground_truth = pd.read_csv('ground_truth.csv')
    
    # Merge data
    data = pd.merge(gene_data, ground_truth, on=['fov', 'spot_id'])
    
    # Prepare features and target
    X = data.drop(['fov', 'spot_id', 'cell_type'], axis=1)
    y = data['cell_type']
    
    return X, y

def main():
    # Load data
    X, y = load_data()
    
    # Initialize and train adaptive KNN
    aknn = AdaptiveKNN()
    aknn.fit(X, y)
    print(f"Best k for KNN: {aknn.best_k}")
    
    # Initialize and train adaptive Bayesian
    abayes = AdaptiveBayesian()
    abayes.fit(X, y)
    print(f"Best var_smoothing for Bayesian: {abayes.best_var_smoothing}")
    
    # Make predictions
    y_pred_knn = aknn.predict(X)
    y_pred_bayes = abayes.predict(X)
    
    # Calculate accuracies
    acc_knn = accuracy_score(y, y_pred_knn)
    acc_bayes = accuracy_score(y, y_pred_bayes)
    
    print(f"KNN Accuracy: {acc_knn:.4f}")
    print(f"Bayesian Accuracy: {acc_bayes:.4f}")

if __name__ == "__main__":
    main() 