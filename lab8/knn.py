"""
K-Nearest Neighbors Classification Implementation from Scratch
Implementation with 5-fold cross validation on drug_200.csv dataset
Testing with K=1, 3, 5
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

class KNNClassifier:
    """K-Nearest Neighbors classifier implemented from scratch"""
    
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initialize KNN Classifier
        
        Parameters:
        - k: Number of nearest neighbors
        - distance_metric: Distance metric to use ('euclidean', 'manhattan')
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store training data (KNN is a lazy learner)"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))
        else:
            raise ValueError("Unsupported distance metric")
    
    def get_neighbors(self, test_point):
        """Get k nearest neighbors for a test point"""
        distances = []
        
        for i, train_point in enumerate(self.X_train):
            distance = self.calculate_distance(test_point, train_point)
            distances.append((distance, self.y_train[i]))
        
        # Sort by distance and get k nearest
        distances.sort(key=lambda x: x[0])
        neighbors = [distances[i][1] for i in range(min(self.k, len(distances)))]
        
        return neighbors
    
    def predict_single(self, test_point):
        """Predict class for a single test point"""
        neighbors = self.get_neighbors(test_point)
        
        # Vote among neighbors
        vote_count = Counter(neighbors)
        prediction = vote_count.most_common(1)[0][0]
        
        return prediction
    
    def predict(self, X_test):
        """Make predictions for test data"""
        predictions = []
        X_test = np.array(X_test)
        
        for test_point in X_test:
            prediction = self.predict_single(test_point)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_proba(self, X_test):
        """Predict probabilities for test data"""
        probabilities = []
        X_test = np.array(X_test)
        
        # Get all unique classes
        unique_classes = np.unique(self.y_train)
        
        for test_point in X_test:
            neighbors = self.get_neighbors(test_point)
            vote_count = Counter(neighbors)
            
            # Calculate probabilities
            prob_dict = {}
            total_neighbors = len(neighbors)
            
            for class_label in unique_classes:
                prob_dict[class_label] = vote_count.get(class_label, 0) / total_neighbors
            
            # Convert to array in sorted order of classes
            class_probs = [prob_dict[class_label] for class_label in sorted(unique_classes)]
            probabilities.append(class_probs)
        
        return np.array(probabilities)

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
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Standardize features (important for KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
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

def cross_validate_knn(X, y, k_value, target_encoder, cv_folds=5):
    """Perform k-fold cross validation for KNN"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    metrics = defaultdict(list)
    fold_results = []
    confusion_matrices = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Training KNN (K={k_value}) fold {fold + 1}/{cv_folds}...")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = KNNClassifier(k=k_value)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        fold_metrics = evaluate_model(y_test, y_pred, target_encoder)
        
        for metric, value in fold_metrics.items():
            metrics[metric].append(value)
        
        # Store confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(cm)
        
        fold_results.append({
            'fold': fold + 1,
            'metrics': fold_metrics,
            'predictions': y_pred,
            'true_labels': y_test,
            'confusion_matrix': cm
        })
    
    # Calculate mean and std for each metric
    final_metrics = {}
    for metric, values in metrics.items():
        final_metrics[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return final_metrics, fold_results, confusion_matrices

def plot_confusion_matrix_average(confusion_matrices, target_encoder, k_value):
    """Plot average confusion matrix across all folds"""
    # Calculate average confusion matrix
    avg_cm = np.mean(confusion_matrices, axis=0)
    
    plt.figure(figsize=(8, 6))
    class_names = target_encoder.classes_
    
    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Average Confusion Matrix - KNN (K={k_value})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    return avg_cm

def analyze_feature_importance_knn(X, y, k_value, target_encoder, feature_names):
    """Analyze feature importance by measuring performance drop when features are removed"""
    print(f"\nAnalyzing feature importance for KNN (K={k_value})...")
    
    # Baseline performance with all features
    baseline_metrics, _, _ = cross_validate_knn(X, y, k_value, target_encoder, cv_folds=3)
    baseline_accuracy = baseline_metrics['accuracy']['mean']
    
    feature_importance = {}
    
    for i, feature_name in enumerate(feature_names):
        # Remove one feature and test performance
        X_reduced = np.delete(X, i, axis=1)
        reduced_metrics, _, _ = cross_validate_knn(X_reduced, y, k_value, target_encoder, cv_folds=3)
        reduced_accuracy = reduced_metrics['accuracy']['mean']
        
        # Importance = drop in accuracy when feature is removed
        importance = baseline_accuracy - reduced_accuracy
        feature_importance[feature_name] = importance
        
        print(f"Without {feature_name}: Accuracy = {reduced_accuracy:.4f} (drop: {importance:.4f})")
    
    return feature_importance

def main():
    """Main function to run the KNN experiment"""
    print("Loading and preprocessing data...")
    X, y, target_encoder, scaler, label_encoders = load_and_preprocess_data()
    
    feature_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
    
    # Test different K values
    k_values = [1, 3, 5]
    
    print("\n" + "="*50)
    print("K-NEAREST NEIGHBORS CLASSIFICATION")
    print("="*50)
    
    all_results = {}
    all_confusion_matrices = {}
    
    for k in k_values:
        print(f"\n{'='*20} K = {k} {'='*20}")
        
        # Perform cross-validation
        metrics, fold_results, confusion_matrices = cross_validate_knn(X, y, k, target_encoder)
        all_results[f'KNN_K{k}'] = metrics
        all_confusion_matrices[f'KNN_K{k}'] = confusion_matrices
        
        # Print results
        print(f"\nResults for K = {k}:")
        print("-" * 30)
        for metric, values in metrics.items():
            print(f"{metric.capitalize()}: {values['mean']:.4f} (+/- {values['std']:.4f})")
        
        # Plot confusion matrix for this K value
        avg_cm = plot_confusion_matrix_average(confusion_matrices, target_encoder, k)
        plt.savefig(f'c:\\Users\\acer\\Desktop\\U23AI118\\SEM 5\\ML-Lab\\lab8\\knn_confusion_matrix_k{k}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # Create comparison plot
    plt.figure(figsize=(15, 10))
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(2, 2, i)
        
        k_names = [f'K={k}' for k in k_values]
        means = [all_results[f'KNN_K{k}'][metric]['mean'] for k in k_values]
        stds = [all_results[f'KNN_K{k}'][metric]['std'] for k in k_values]
        
        bars = plt.bar(range(len(k_names)), means, yerr=stds, capsize=5, alpha=0.7, color='skyblue')
        plt.xlabel('K Value')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs K Value')
        plt.xticks(range(len(k_names)), k_names)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.01,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('c:\\Users\\acer\\Desktop\\U23AI118\\SEM 5\\ML-Lab\\lab8\\knn_k_comparison.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Feature importance analysis for best K
    best_k = k_values[np.argmax([all_results[f'KNN_K{k}']['accuracy']['mean'] for k in k_values])]
    print(f"\nBest K value: {best_k}")
    
    feature_importance = analyze_feature_importance_knn(X, y, best_k, target_encoder, feature_names)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    features = list(feature_importance.keys())
    importance_values = list(feature_importance.values())
    
    bars = plt.bar(features, importance_values, alpha=0.7, color='lightcoral')
    plt.xlabel('Features')
    plt.ylabel('Importance (Accuracy Drop)')
    plt.title(f'Feature Importance - KNN (K={best_k})')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, importance in zip(bars, importance_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(importance_values)*0.01,
                f'{importance:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('c:\\Users\\acer\\Desktop\\U23AI118\\SEM 5\\ML-Lab\\lab8\\knn_feature_importance.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE - K-NEAREST NEIGHBORS COMPARISON")
    print("="*70)
    print(f"{'K Value':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 70)
    
    for k in k_values:
        metrics = all_results[f'KNN_K{k}']
        print(f"{k:<10} "
              f"{metrics['accuracy']['mean']:.4f}    "
              f"{metrics['precision']['mean']:.4f}     "
              f"{metrics['recall']['mean']:.4f}    "
              f"{metrics['f1_score']['mean']:.4f}")
    
    # Detailed classification report for best K
    print(f"\nDetailed Classification Report for Best K={best_k}:")
    print("-" * 50)
    
    # Get predictions from the best K model for detailed analysis
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_y_true = []
    all_y_pred = []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = KNNClassifier(k=best_k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    # Convert back to original labels for the report
    y_true_original = target_encoder.inverse_transform(all_y_true)
    y_pred_original = target_encoder.inverse_transform(all_y_pred)
    
    print(classification_report(y_true_original, y_pred_original))
    
    return all_results

if __name__ == "__main__":
    results = main()
