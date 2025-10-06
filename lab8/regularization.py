"""
Logistic Regression with Different Regularization Techniques
Implementation from scratch with 5-fold cross validation on drug_200.csv dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class LogisticRegressionScratch:
    """Logistic Regression implementation from scratch with regularization options"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization=None, lambda_reg=0.01, l1_ratio=0.5):
        """
        Initialize Logistic Regression
        
        Parameters:
        - learning_rate: Learning rate for gradient descent
        - max_iterations: Maximum number of iterations
        - regularization: Type of regularization ('l1', 'l2', 'elastic_net', or None)
        - lambda_reg: Regularization strength
        - l1_ratio: Ratio for elastic net (0=Ridge, 1=Lasso)
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.l1_ratio = l1_ratio
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def sigmoid(self, z):
        """Sigmoid activation function with clipping to prevent overflow"""
        z = np.clip(z, -250, 250)  # Prevent overflow
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, y_true, y_pred, weights):
        """Compute cost function with regularization"""
        m = len(y_true)
        
        # Cross-entropy loss
        epsilon = 1e-15  # Small constant to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # Add regularization
        if self.regularization == 'l2':
            cost += self.lambda_reg * np.sum(weights ** 2) / (2 * m)
        elif self.regularization == 'l1':
            cost += self.lambda_reg * np.sum(np.abs(weights)) / m
        elif self.regularization == 'elastic_net':
            l1_penalty = self.lambda_reg * self.l1_ratio * np.sum(np.abs(weights)) / m
            l2_penalty = self.lambda_reg * (1 - self.l1_ratio) * np.sum(weights ** 2) / (2 * m)
            cost += l1_penalty + l2_penalty
            
        return cost
    
    def fit(self, X, y, random_seed=None):
        """Train the logistic regression model"""
        # Initialize parameters with seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.1, n_features)  # Larger initial weights
        self.bias = 0
        self.cost_history = []
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        m = len(y)
        
        # Training loop
        for i in range(self.max_iterations):
            # Forward propagation
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred, self.weights)
            self.cost_history.append(cost)
            
            # Backward propagation
            dw = np.dot(X.T, (y_pred - y)) / m
            db = np.mean(y_pred - y)
            
            # Add regularization to gradients
            if self.regularization == 'l2':
                dw += self.lambda_reg * self.weights / m
            elif self.regularization == 'l1':
                dw += self.lambda_reg * np.sign(self.weights) / m
            elif self.regularization == 'elastic_net':
                l1_grad = self.lambda_reg * self.l1_ratio * np.sign(self.weights) / m
                l2_grad = self.lambda_reg * (1 - self.l1_ratio) * self.weights / m
                dw += l1_grad + l2_grad
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence (less aggressive convergence criteria)
            if i > 100 and abs(self.cost_history[-2] - self.cost_history[-1]) < 1e-10:
                print(f"Converged at iteration {i}")
                break
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X):
        """Make predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

class MultiClassLogisticRegression:
    """Multi-class logistic regression using One-vs-Rest strategy"""
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, regularization=None, lambda_reg=0.01, l1_ratio=0.5):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.l1_ratio = l1_ratio
        self.classifiers = {}
        self.classes = None
        
    def fit(self, X, y, random_seed=None):
        """Train the multi-class classifier"""
        self.classes = np.unique(y)
        
        for i, class_label in enumerate(self.classes):
            # Create binary labels for current class vs all others
            binary_y = (y == class_label).astype(int)
            
            # Train binary classifier for this class
            classifier = LogisticRegressionScratch(
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations,
                regularization=self.regularization,
                lambda_reg=self.lambda_reg,
                l1_ratio=self.l1_ratio
            )
            # Use different seed for each class to avoid identical initialization
            seed = random_seed + i if random_seed is not None else None
            classifier.fit(X, binary_y, random_seed=seed)
            self.classifiers[class_label] = classifier
    
    def predict_proba(self, X):
        """Predict probabilities for all classes"""
        probabilities = np.zeros((len(X), len(self.classes)))
        
        for i, class_label in enumerate(self.classes):
            probabilities[:, i] = self.classifiers[class_label].predict_proba(X)
        
        # Normalize probabilities (softmax-like)
        probabilities = probabilities / (np.sum(probabilities, axis=1, keepdims=True) + 1e-15)
        return probabilities
    
    def predict(self, X):
        """Make predictions"""
        probabilities = self.predict_proba(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        return self.classes[predicted_indices]

def load_and_preprocess_data():
    """Load and preprocess the drug dataset"""
    # Load data
    data = pd.read_csv('c:\\Users\\acer\\Desktop\\U23AI118\\SEM 5\\ML-Lab\\lab7\\drug_200.csv')
    
    print("Dataset shape:", data.shape)
    print("\nDataset info:")
    print(data.info())
    print("\nTarget distribution:")
    print(data['Drug'].value_counts())
    
    # Separate features and target
    X = data.drop('Drug', axis=1)
    y = data['Drug']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Sex', 'BP', 'Cholesterol']
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Add polynomial features to create more complex feature space
    # This will make regularization effects more visible
    X_poly = X.copy()
    X_poly['Age_squared'] = X['Age'] ** 2
    X_poly['Na_K_squared'] = X['Na_to_K'] ** 2
    X_poly['Age_Na_K'] = X['Age'] * X['Na_to_K']
    X_poly['BP_Chol'] = X['BP'] * X['Cholesterol']
    
    print(f"Features after polynomial expansion: {X_poly.shape[1]}")
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    return X_scaled, y_encoded, target_encoder, scaler, label_encoders

def evaluate_model(y_true, y_pred, target_encoder):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def cross_validate_model(X, y, model_params, target_encoder, cv_folds=5):
    """Perform k-fold cross validation"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    metrics = defaultdict(list)
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Training fold {fold + 1}/{cv_folds}...")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model with different seed for each fold
        model = MultiClassLogisticRegression(**model_params)
        model.fit(X_train, y_train, random_seed=42 + fold)
        
        # Debug: Print some weight statistics for first fold
        if fold == 0:
            for class_label, classifier in model.classifiers.items():
                weight_norm = np.linalg.norm(classifier.weights)
                final_cost = classifier.cost_history[-1] if classifier.cost_history else 0
                # Count near-zero weights (for sparsity analysis)
                near_zero_weights = np.sum(np.abs(classifier.weights) < 1e-3)
                print(f"  Class {class_label}: Weight norm = {weight_norm:.4f}, Final cost = {final_cost:.4f}, Sparse weights = {near_zero_weights}/{len(classifier.weights)}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        fold_metrics = evaluate_model(y_test, y_pred, target_encoder)
        
        for metric, value in fold_metrics.items():
            metrics[metric].append(value)
        
        fold_results.append({
            'fold': fold + 1,
            'metrics': fold_metrics,
            'predictions': y_pred,
            'true_labels': y_test
        })
    
    # Calculate mean and std for each metric
    final_metrics = {}
    for metric, values in metrics.items():
        final_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return final_metrics, fold_results

def main():
    """Main function to run the experiment"""
    print("Loading and preprocessing data...")
    X, y, target_encoder, scaler, label_encoders = load_and_preprocess_data()
    
    # Define different regularization configurations
    regularization_configs = {
        'No Regularization': {
            'regularization': None,
            'lambda_reg': 0,
            'learning_rate': 0.1,
            'max_iterations': 2000
        },
        'L2 Regularization (Ridge)': {
            'regularization': 'l2',
            'lambda_reg': 1.0,
            'learning_rate': 0.1,
            'max_iterations': 2000
        },
        'L1 Regularization (Lasso)': {
            'regularization': 'l1',
            'lambda_reg': 0.5,
            'learning_rate': 0.1,
            'max_iterations': 2000
        },
        'Elastic Net': {
            'regularization': 'elastic_net',
            'lambda_reg': 0.2,
            'l1_ratio': 0.5,
            'learning_rate': 0.1,
            'max_iterations': 2000
        }
    }
    
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION WITH REGULARIZATION")
    print("="*50)
    
    all_results = {}
    
    for reg_name, params in regularization_configs.items():
        print(f"\n{reg_name}:")
        print("-" * 30)
        
        # Perform cross-validation
        metrics, fold_results = cross_validate_model(X, y, params, target_encoder)
        all_results[reg_name] = metrics
        
        # Print results
        for metric, values in metrics.items():
            print(f"{metric.capitalize()}: {values['mean']:.4f} (+/- {values['std']:.4f})")
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 2, i)
        
        reg_names = list(all_results.keys())
        means = [all_results[name][metric]['mean'] for name in reg_names]
        stds = [all_results[name][metric]['std'] for name in reg_names]
        
        bars = plt.bar(range(len(reg_names)), means, yerr=stds, capsize=5, alpha=0.7)
        plt.xlabel('Regularization Type')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Comparison')
        plt.xticks(range(len(reg_names)), reg_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('c:\\Users\\acer\\Desktop\\U23AI118\\SEM 5\\ML-Lab\\lab8\\regularization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE - LOGISTIC REGRESSION REGULARIZATION COMPARISON")
    print("="*80)
    print(f"{'Regularization Type':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for reg_name, metrics in all_results.items():
        print(f"{reg_name:<25} "
              f"{metrics['accuracy']['mean']:.4f}    "
              f"{metrics['precision']['mean']:.4f}     "
              f"{metrics['recall']['mean']:.4f}    "
              f"{metrics['f1_score']['mean']:.4f}")

if __name__ == "__main__":
    main()
