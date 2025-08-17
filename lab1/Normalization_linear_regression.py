import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple, List
import os
from scipy import stats

warnings.filterwarnings('ignore')
plt.style.use('default')

class SimpleLinearRegressionFromScratch:
    """Simple Linear Regression implemented from scratch with batch and online learning"""
    
    def __init__(self, learning_rate: float = 0.001, max_epochs: int = 1000, tolerance: float = 1e-6, normalization: str = 'standard'):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.normalization = normalization  # 'standard' or 'minmax'
        
        # Model parameters
        self.weight = None
        self.bias = None
        
        # Training history
        self.cost_history = []
        self.epoch_history = []
        
        # Data normalization parameters - Standard Scaling
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        
        # Data normalization parameters - Min-Max Scaling
        self.X_min = None
        self.X_max = None
        self.y_min = None
        self.y_max = None
    
    def _normalize_data_standard(self, X: np.ndarray, y: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features and target using standard scaling (z-score normalization)"""
        if fit:
            self.X_mean = np.mean(X)
            self.X_std = np.std(X)
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
        
        X_norm = (X - self.X_mean) / self.X_std if self.X_std != 0 else X - self.X_mean
        y_norm = (y - self.y_mean) / self.y_std if self.y_std != 0 else y - self.y_mean
        
        return X_norm, y_norm
    
    def _normalize_data_minmax(self, X: np.ndarray, y: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features and target using min-max scaling"""
        if fit:
            self.X_min = np.min(X)
            self.X_max = np.max(X)
            self.y_min = np.min(y)
            self.y_max = np.max(y)
        
        X_norm = (X - self.X_min) / (self.X_max - self.X_min) if (self.X_max - self.X_min) != 0 else X - self.X_min
        y_norm = (y - self.y_min) / (self.y_max - self.y_min) if (self.y_max - self.y_min) != 0 else y - self.y_min
        
        return X_norm, y_norm
    
    def _normalize_data(self, X: np.ndarray, y: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize data using specified method"""
        if self.normalization == 'standard':
            return self._normalize_data_standard(X, y, fit)
        elif self.normalization == 'minmax':
            return self._normalize_data_minmax(X, y, fit)
        else:
            raise ValueError("Normalization must be 'standard' or 'minmax'")
    
    def _denormalize_predictions_standard(self, y_pred_norm: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale (standard scaling)"""
        return y_pred_norm * self.y_std + self.y_mean
    
    def _denormalize_predictions_minmax(self, y_pred_norm: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale (min-max scaling)"""
        return y_pred_norm * (self.y_max - self.y_min) + self.y_min
    
    def _denormalize_predictions(self, y_pred_norm: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale"""
        if self.normalization == 'standard':
            return self._denormalize_predictions_standard(y_pred_norm)
        elif self.normalization == 'minmax':
            return self._denormalize_predictions_minmax(y_pred_norm)
    
    def get_normalization_stats(self) -> dict:
        """Get normalization statistics for analysis"""
        stats = {'method': self.normalization}
        if self.normalization == 'standard':
            stats.update({
                'X_mean': self.X_mean,
                'X_std': self.X_std,
                'y_mean': self.y_mean,
                'y_std': self.y_std
            })
        elif self.normalization == 'minmax':
            stats.update({
                'X_min': self.X_min,
                'X_max': self.X_max,
                'y_min': self.y_min,
                'y_max': self.y_max
            })
        return stats
    
    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error"""
        return np.mean((y_pred - y_true) ** 2)
    
    def fit_batch(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """Train using batch gradient descent"""
        if verbose:
            print(f"Training with Batch Gradient Descent ({self.normalization} normalization)...")
        
        # Normalize data
        X_norm, y_norm = self._normalize_data(X, y, fit=True)
        
        # Initialize parameters
        self.weight = np.random.normal(0, 0.01)
        self.bias = np.random.normal(0, 0.01)
        
        self.cost_history = []
        self.epoch_history = []
        
        prev_cost = float('inf')
        m = len(y_norm)
        
        for epoch in range(self.max_epochs):
            # Forward pass
            y_pred = self.weight * X_norm + self.bias
            
            # Compute cost
            current_cost = self._compute_cost(y_norm, y_pred)
            self.cost_history.append(current_cost)
            self.epoch_history.append(epoch)
            
            # Compute gradients
            dw = (1/m) * np.sum((y_pred - y_norm) * X_norm)
            db = (1/m) * np.sum(y_pred - y_norm)
            
            # Update parameters
            self.weight -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if abs(prev_cost - current_cost) < self.tolerance:
                if verbose:
                    print(f"Converged at epoch {epoch} with cost {current_cost:.6f}")
                break
            
            prev_cost = current_cost
            
            if verbose and epoch % 200 == 0:
                print(f"Epoch {epoch}: Cost = {current_cost:.6f}")
        
        if verbose:
            print(f"Batch training completed. Final cost: {current_cost:.6f}")
    
    def fit_online(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """Train using online (stochastic) gradient descent"""
        if verbose:
            print("Training with Online Gradient Descent...")
        
        # Normalize data
        X_norm, y_norm = self._normalize_data(X, y, fit=True)
        
        # Initialize parameters
        self.weight = np.random.normal(0, 0.01)
        self.bias = np.random.normal(0, 0.01)
        
        self.cost_history = []
        self.epoch_history = []
        
        m = len(y_norm)
        
        for epoch in range(self.max_epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X_norm[indices]
            y_shuffled = y_norm[indices]
            
            epoch_cost = 0
            
            # Process one sample at a time
            for i in range(m):
                # Forward pass for single sample
                y_pred_i = self.weight * X_shuffled[i] + self.bias
                
                # Compute cost for this sample
                sample_cost = (y_pred_i - y_shuffled[i]) ** 2
                epoch_cost += sample_cost
                
                # Compute gradients
                dw = (y_pred_i - y_shuffled[i]) * X_shuffled[i]
                db = y_pred_i - y_shuffled[i]
                
                # Update parameters
                self.weight -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Average cost for the epoch
            avg_cost = epoch_cost / m
            self.cost_history.append(avg_cost)
            self.epoch_history.append(epoch)
            
            if verbose and epoch % 200 == 0:
                print(f"Epoch {epoch}: Average Cost = {avg_cost:.6f}")
        
        if verbose:
            print(f"Online training completed. Final average cost: {avg_cost:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        if self.normalization == 'standard':
            X_norm = (X - self.X_mean) / self.X_std if self.X_std != 0 else X - self.X_mean
        elif self.normalization == 'minmax':
            X_norm = (X - self.X_min) / (self.X_max - self.X_min) if (self.X_max - self.X_min) != 0 else X - self.X_min
        else:
            raise ValueError("Invalid normalization method")
        
        y_pred_norm = self.weight * X_norm + self.bias
        return self._denormalize_predictions(y_pred_norm)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def mean_squared_error(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate Mean Squared Error"""
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

class DatasetAnalyzer:
    """Comprehensive dataset analysis and modeling"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.df = None
        self.df_original = None  # Store original data
        self.feature_col = None
        self.target_col = None
        self.outliers_removed = 0
        
        # Store normalization and outlier detection results
        self.normalization_results = {}
        self.outlier_detection_results = {}
    
    def load_data(self) -> bool:
        """Load dataset"""
        try:
            file_path = f'c:\\Users\\acer\\Desktop\\U23AI118\\SEM 5\\ML-Lab\\lab1\\{self.dataset_name}.csv'
            self.df = pd.read_csv(file_path)
            
            # Set feature and target columns based on dataset
            if self.dataset_name.lower() == 'housing':
                self.feature_col = 'area'
                self.target_col = 'price'
                # Convert categorical variables to numeric
                categorical_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                                   'airconditioning', 'prefarea']
                for var in categorical_vars:
                    if var in self.df.columns:
                        self.df[var] = self.df[var].map({'yes': 1, 'no': 0})
                
                if 'furnishingstatus' in self.df.columns:
                    furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
                    self.df['furnishingstatus'] = self.df['furnishingstatus'].map(furnishing_map)
            
            else:  # advertising dataset
                self.feature_col = 'TV'
                self.target_col = 'Sales'
            
            print(f"✅ {self.dataset_name} dataset loaded successfully!")
            print(f"   Shape: {self.df.shape}")
            print(f"   Feature: {self.feature_col}, Target: {self.target_col}")
            return True
            
        except FileNotFoundError:
            print(f"❌ Error: {self.dataset_name}.csv not found!")
            return False
    
    def explore_data(self):
        """Explore and visualize dataset"""
        print(f"\n{'='*60}")
        print(f"EXPLORING {self.dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        # Basic statistics
        print("\nDataset Info:")
        print(self.df.info())
        print("\nBasic Statistics:")
        print(self.df.describe())
        
        # Correlation analysis
        correlation = self.df[self.feature_col].corr(self.df[self.target_col])
        print(f"\nCorrelation between {self.feature_col} and {self.target_col}: {correlation:.4f}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.dataset_name.title()} Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Target distribution
        axes[0,0].hist(self.df[self.target_col], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title(f'{self.target_col} Distribution')
        axes[0,0].set_xlabel(self.target_col)
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Feature vs Target scatter plot
        axes[0,1].scatter(self.df[self.feature_col], self.df[self.target_col], alpha=0.6, color='red')
        axes[0,1].set_title(f'{self.target_col} vs {self.feature_col}')
        axes[0,1].set_xlabel(self.feature_col)
        axes[0,1].set_ylabel(self.target_col)
        axes[0,1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.df[self.feature_col], self.df[self.target_col], 1)
        p = np.poly1d(z)
        axes[0,1].plot(self.df[self.feature_col], p(self.df[self.feature_col]), "b--", linewidth=2)
        
        # 3. Box plot
        axes[1,0].boxplot(self.df[self.target_col])
        axes[1,0].set_title(f'{self.target_col} Box Plot')
        axes[1,0].set_ylabel(self.target_col)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Correlation matrix for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1,1], fmt='.2f', square=True)
            axes[1,1].set_title('Correlation Matrix')
        else:
            axes[1,1].text(0.5, 0.5, 'Not enough numeric\ncolumns for correlation', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_feature_importance(self):
        """Analyze feature importance for multi-feature datasets"""
        if self.dataset_name.lower() != 'housing':
            return
            
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        
        # Get all numeric features except target
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in numeric_features:
            numeric_features.remove(self.target_col)
        
        if len(numeric_features) <= 1:
            print("Not enough features for importance analysis.")
            return
        
        # Simple correlation-based importance
        correlations = []
        for feature in numeric_features:
            corr = abs(self.df[feature].corr(self.df[self.target_col]))
            correlations.append(corr)
        
        # Normalize to percentages
        total_corr = sum(correlations)
        importance_pct = [100 * corr / total_corr for corr in correlations]
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': numeric_features,
            'Correlation': correlations,
            'Importance (%)': importance_pct
        }).sort_values('Importance (%)', ascending=False)
        
        # Visualize
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance (%)'], color='lightcoral')
        plt.xlabel('Relative Importance (%)')
        plt.title('Feature Importance for Housing Price Prediction\n(Based on Correlation with Price)')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        # Print results
        print("\nFeature Importance Rankings:")
        print("-" * 50)
        for _, row in importance_df.iterrows():
            print(f"{row['Feature']:<15}: {row['Importance (%)']:>6.1f}% (corr: {row['Correlation']:>5.3f})")
    
    def split_data(self, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets"""
        X = self.df[self.feature_col].values
        y = self.df[self.target_col].values
        
        # Simple random split
        np.random.seed(42)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
    
    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5) -> np.ndarray:
        """Detect outliers using Interquartile Range (IQR) method"""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        return outliers
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> np.ndarray:
        """Detect outliers using Z-score method"""
        z_scores = np.abs(stats.zscore(self.df[column]))
        outliers = z_scores > threshold
        return outliers
    
    def detect_outliers_modified_zscore(self, column: str, threshold: float = 3.5) -> np.ndarray:
        """Detect outliers using Modified Z-score method (more robust)"""
        median = np.median(self.df[column])
        mad = np.median(np.abs(self.df[column] - median))
        modified_z_scores = 0.6745 * (self.df[column] - median) / mad
        outliers = np.abs(modified_z_scores) > threshold
        return outliers
    
    def visualize_outliers(self):
        """Visualize outliers before removal"""
        print(f"\n{'='*60}")
        print("OUTLIER DETECTION ANALYSIS")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.dataset_name.title()} - Outlier Detection Visualization', fontsize=16)
        
        # Original data for both feature and target
        columns_to_analyze = [self.feature_col, self.target_col]
        
        for idx, column in enumerate(columns_to_analyze):
            # Box plot
            axes[idx, 0].boxplot(self.df[column])
            axes[idx, 0].set_title(f'{column} - Box Plot')
            axes[idx, 0].set_ylabel(column)
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Histogram with outliers highlighted
            axes[idx, 1].hist(self.df[column], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Mark outliers using IQR
            iqr_outliers = self.detect_outliers_iqr(column)
            if iqr_outliers.any():
                outlier_values = self.df[column][iqr_outliers]
                axes[idx, 1].hist(outlier_values, bins=20, alpha=0.8, color='red', 
                                edgecolor='black', label=f'IQR Outliers ({iqr_outliers.sum()})')
                axes[idx, 1].legend()
            
            axes[idx, 1].set_title(f'{column} - Distribution with Outliers')
            axes[idx, 1].set_xlabel(column)
            axes[idx, 1].set_ylabel('Frequency')
            axes[idx, 1].grid(True, alpha=0.3)
            
            # Z-score plot
            z_scores = np.abs(stats.zscore(self.df[column]))
            axes[idx, 2].scatter(range(len(z_scores)), z_scores, alpha=0.6, color='blue')
            axes[idx, 2].axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='Z-score = 3')
            axes[idx, 2].axhline(y=2.0, color='orange', linestyle='--', linewidth=1, label='Z-score = 2')
            axes[idx, 2].set_title(f'{column} - Z-scores')
            axes[idx, 2].set_xlabel('Data Point Index')
            axes[idx, 2].set_ylabel('|Z-score|')
            axes[idx, 2].legend()
            axes[idx, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print outlier statistics
        print("\nOutlier Detection Summary:")
        print("-" * 50)
        
        for column in columns_to_analyze:
            iqr_outliers = self.detect_outliers_iqr(column)
            zscore_outliers = self.detect_outliers_zscore(column)
            modified_zscore_outliers = self.detect_outliers_modified_zscore(column)
            
            print(f"\n{column.upper()}:")
            print(f"  IQR Method (1.5×IQR):     {iqr_outliers.sum():>3} outliers ({iqr_outliers.mean()*100:.1f}%)")
            print(f"  Z-score Method (|z|>3):   {zscore_outliers.sum():>3} outliers ({zscore_outliers.mean()*100:.1f}%)")
            print(f"  Modified Z-score (|z|>3.5): {modified_zscore_outliers.sum():>3} outliers ({modified_zscore_outliers.mean()*100:.1f}%)")
    
    def remove_outliers(self, method: str = 'iqr', feature_threshold: float = 1.5, target_threshold: float = 1.5) -> int:
        """Remove outliers from the dataset"""
        original_size = len(self.df)
        
        if method.lower() == 'iqr':
            feature_outliers = self.detect_outliers_iqr(self.feature_col, feature_threshold)
            target_outliers = self.detect_outliers_iqr(self.target_col, target_threshold)
        elif method.lower() == 'zscore':
            feature_outliers = self.detect_outliers_zscore(self.feature_col, feature_threshold)
            target_outliers = self.detect_outliers_zscore(self.target_col, target_threshold)
        elif method.lower() == 'modified_zscore':
            feature_outliers = self.detect_outliers_modified_zscore(self.feature_col, feature_threshold)
            target_outliers = self.detect_outliers_modified_zscore(self.target_col, target_threshold)
        else:
            raise ValueError("Method must be 'iqr', 'zscore', or 'modified_zscore'")
        
        # Combine outliers (remove if outlier in either feature or target)
        combined_outliers = feature_outliers | target_outliers
        
        # Remove outliers
        self.df = self.df[~combined_outliers].reset_index(drop=True)
        
        self.outliers_removed = original_size - len(self.df)
        
        print(f"\n🔍 Outlier Removal Results:")
        print(f"   Method: {method.upper()}")
        print(f"   Original dataset size: {original_size}")
        print(f"   Outliers removed: {self.outliers_removed}")
        print(f"   Final dataset size: {len(self.df)}")
        print(f"   Percentage removed: {(self.outliers_removed/original_size)*100:.2f}%")
        
        return self.outliers_removed
    
    def compare_with_without_outliers(self):
        """Compare model performance with and without outliers"""
        print(f"\n{'='*60}")
        print("COMPARING PERFORMANCE: WITH vs WITHOUT OUTLIERS")
        print(f"{'='*60}")
        
        # Train model with outliers (original data)
        X_orig = self.df_original[self.feature_col].values
        y_orig = self.df_original[self.target_col].values
        
        # Simple split for original data
        np.random.seed(42)
        n_samples_orig = len(X_orig)
        n_test_orig = int(n_samples_orig * 0.2)
        indices_orig = np.random.permutation(n_samples_orig)
        test_indices_orig = indices_orig[:n_test_orig]
        train_indices_orig = indices_orig[n_test_orig:]
        
        X_train_orig = X_orig[train_indices_orig]
        X_test_orig = X_orig[test_indices_orig]
        y_train_orig = y_orig[train_indices_orig]
        y_test_orig = y_orig[test_indices_orig]
        
        # Train model without outliers (cleaned data)
        X_clean = self.df[self.feature_col].values
        y_clean = self.df[self.target_col].values
        
        # Simple split for cleaned data
        n_samples_clean = len(X_clean)
        n_test_clean = int(n_samples_clean * 0.2)
        indices_clean = np.random.permutation(n_samples_clean)
        test_indices_clean = indices_clean[:n_test_clean]
        train_indices_clean = indices_clean[n_test_clean:]
        
        X_train_clean = X_clean[train_indices_clean]
        X_test_clean = X_clean[test_indices_clean]
        y_train_clean = y_clean[train_indices_clean]
        y_test_clean = y_clean[test_indices_clean]
        
        # Train models
        model_with_outliers = SimpleLinearRegressionFromScratch(learning_rate=0.01, max_epochs=1000)
        model_with_outliers.fit_batch(X_train_orig, y_train_orig, verbose=False)
        
        model_without_outliers = SimpleLinearRegressionFromScratch(learning_rate=0.01, max_epochs=1000)
        model_without_outliers.fit_batch(X_train_clean, y_train_clean, verbose=False)
        
        # Evaluate models
        print("\nModel Comparison Results:")
        print("-" * 70)
        print(f"{'Dataset':<20} {'Train R²':<10} {'Test R²':<10} {'Train MSE':<12} {'Test MSE':<12}")
        print("-" * 70)
        
        # With outliers
        train_r2_orig = model_with_outliers.score(X_train_orig, y_train_orig)
        test_r2_orig = model_with_outliers.score(X_test_orig, y_test_orig)
        train_mse_orig = model_with_outliers.mean_squared_error(X_train_orig, y_train_orig)
        test_mse_orig = model_with_outliers.mean_squared_error(X_test_orig, y_test_orig)
        
        print(f"{'With Outliers':<20} {train_r2_orig:<10.4f} {test_r2_orig:<10.4f} {train_mse_orig:<12.2f} {test_mse_orig:<12.2f}")
        
        # Without outliers
        train_r2_clean = model_without_outliers.score(X_train_clean, y_train_clean)
        test_r2_clean = model_without_outliers.score(X_test_clean, y_test_clean)
        train_mse_clean = model_without_outliers.mean_squared_error(X_train_clean, y_train_clean)
        test_mse_clean = model_without_outliers.mean_squared_error(X_test_clean, y_test_clean)
        
        print(f"{'Without Outliers':<20} {train_r2_clean:<10.4f} {test_r2_clean:<10.4f} {train_mse_clean:<12.2f} {test_mse_clean:<12.2f}")
        
        # Calculate improvements
        r2_improvement = test_r2_clean - test_r2_orig
        mse_improvement = ((test_mse_orig - test_mse_clean) / test_mse_orig) * 100
        
        print("\nImprovement Summary:")
        print(f"• R² Score improvement: {r2_improvement:+.4f}")
        print(f"• MSE reduction: {mse_improvement:.1f}%")
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'{self.dataset_name.title()} - Impact of Outlier Removal', fontsize=16)
        
        # With outliers
        axes[0].scatter(X_test_orig, y_test_orig, alpha=0.6, color='red', label='Actual', s=50)
        y_pred_orig = model_with_outliers.predict(X_test_orig)
        axes[0].scatter(X_test_orig, y_pred_orig, alpha=0.6, color='blue', label='Predicted', s=30)
        axes[0].set_title(f'With Outliers\nR² = {test_r2_orig:.4f}')
        axes[0].set_xlabel(self.feature_col)
        axes[0].set_ylabel(self.target_col)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Without outliers
        axes[1].scatter(X_test_clean, y_test_clean, alpha=0.6, color='red', label='Actual', s=50)
        y_pred_clean = model_without_outliers.predict(X_test_clean)
        axes[1].scatter(X_test_clean, y_pred_clean, alpha=0.6, color='blue', label='Predicted', s=30)
        axes[1].set_title(f'Without Outliers\nR² = {test_r2_clean:.4f}')
        axes[1].set_xlabel(self.feature_col)
        axes[1].set_ylabel(self.target_col)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'with_outliers': {'r2': test_r2_orig, 'mse': test_mse_orig},
            'without_outliers': {'r2': test_r2_clean, 'mse': test_mse_clean},
            'improvement': {'r2': r2_improvement, 'mse_reduction_pct': mse_improvement}
        }
    
    def compare_normalization_methods(self, X: np.ndarray, y: np.ndarray):
        """Compare standard scaling vs min-max scaling with detailed computations"""
        print(f"\n{'='*70}")
        print("NORMALIZATION METHODS COMPARISON")
        print(f"{'='*70}")
        
        # Original data statistics
        print("\n📊 ORIGINAL DATA STATISTICS:")
        print("-" * 50)
        print(f"Feature ({self.feature_col}):")
        print(f"  Mean: {np.mean(X):.4f}")
        print(f"  Std:  {np.std(X):.4f}")
        print(f"  Min:  {np.min(X):.4f}")
        print(f"  Max:  {np.max(X):.4f}")
        print(f"  Range: {np.max(X) - np.min(X):.4f}")
        
        print(f"\nTarget ({self.target_col}):")
        print(f"  Mean: {np.mean(y):.4f}")
        print(f"  Std:  {np.std(y):.4f}")
        print(f"  Min:  {np.min(y):.4f}")
        print(f"  Max:  {np.max(y):.4f}")
        print(f"  Range: {np.max(y) - np.min(y):.4f}")
        
        # Standard Scaling
        X_mean, X_std = np.mean(X), np.std(X)
        y_mean, y_std = np.mean(y), np.std(y)
        X_standard = (X - X_mean) / X_std
        y_standard = (y - y_mean) / y_std
        
        # Min-Max Scaling
        X_min, X_max = np.min(X), np.max(X)
        y_min, y_max = np.min(y), np.max(y)
        X_minmax = (X - X_min) / (X_max - X_min)
        y_minmax = (y - y_min) / (y_max - y_min)
        
        # Store results for later use
        self.normalization_results = {
            'standard': {
                'X_norm': X_standard, 'y_norm': y_standard,
                'X_mean': X_mean, 'X_std': X_std, 'y_mean': y_mean, 'y_std': y_std
            },
            'minmax': {
                'X_norm': X_minmax, 'y_norm': y_minmax,
                'X_min': X_min, 'X_max': X_max, 'y_min': y_min, 'y_max': y_max
            }
        }
        
        print(f"\n🔄 NORMALIZATION COMPUTATIONS:")
        print("-" * 50)
        
        print("STANDARD SCALING (Z-score normalization):")
        print(f"  Formula: (x - μ) / σ")
        print(f"  Feature parameters: μ = {X_mean:.4f}, σ = {X_std:.4f}")
        print(f"  Target parameters:  μ = {y_mean:.4f}, σ = {y_std:.4f}")
        print(f"  Normalized feature range: [{np.min(X_standard):.4f}, {np.max(X_standard):.4f}]")
        print(f"  Normalized target range:  [{np.min(y_standard):.4f}, {np.max(y_standard):.4f}]")
        
        print(f"\nMIN-MAX SCALING:")
        print(f"  Formula: (x - min) / (max - min)")
        print(f"  Feature parameters: min = {X_min:.4f}, max = {X_max:.4f}")
        print(f"  Target parameters:  min = {y_min:.4f}, max = {y_max:.4f}")
        print(f"  Normalized feature range: [{np.min(X_minmax):.4f}, {np.max(X_minmax):.4f}]")
        print(f"  Normalized target range:  [{np.min(y_minmax):.4f}, {np.max(y_minmax):.4f}]")
        
        # Sample transformations
        print(f"\n📋 SAMPLE TRANSFORMATIONS (first 5 values):")
        print("-" * 70)
        print(f"{'Original':<12} {'Standard':<12} {'Min-Max':<12} {'Feature'}")
        print("-" * 70)
        for i in range(min(5, len(X))):
            print(f"{X[i]:<12.2f} {X_standard[i]:<12.4f} {X_minmax[i]:<12.4f} {self.feature_col}")
        
        print(f"\n{'Original':<12} {'Standard':<12} {'Min-Max':<12} {'Target'}")
        print("-" * 70)
        for i in range(min(5, len(y))):
            print(f"{y[i]:<12.2f} {y_standard[i]:<12.4f} {y_minmax[i]:<12.4f} {self.target_col}")
        
        # Visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.dataset_name.title()} - Normalization Methods Comparison', fontsize=16)
        
        # Original data
        axes[0,0].hist(X, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].set_title(f'Original {self.feature_col}')
        axes[0,0].set_xlabel(self.feature_col)
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        axes[1,0].hist(y, bins=20, alpha=0.7, color='green', edgecolor='black')
        axes[1,0].set_title(f'Original {self.target_col}')
        axes[1,0].set_xlabel(self.target_col)
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)
        
        # Standard scaling
        axes[0,1].hist(X_standard, bins=20, alpha=0.7, color='red', edgecolor='black')
        axes[0,1].set_title(f'Standard Scaled {self.feature_col}')
        axes[0,1].set_xlabel('Standardized Values')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        axes[1,1].hist(y_standard, bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1,1].set_title(f'Standard Scaled {self.target_col}')
        axes[1,1].set_xlabel('Standardized Values')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].grid(True, alpha=0.3)
        
        # Min-max scaling
        axes[0,2].hist(X_minmax, bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[0,2].set_title(f'Min-Max Scaled {self.feature_col}')
        axes[0,2].set_xlabel('Normalized Values [0,1]')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].grid(True, alpha=0.3)
        
        axes[1,2].hist(y_minmax, bins=20, alpha=0.7, color='brown', edgecolor='black')
        axes[1,2].set_title(f'Min-Max Scaled {self.target_col}')
        axes[1,2].set_xlabel('Normalized Values [0,1]')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return self.normalization_results
    
    def compare_outlier_detection_methods(self, detailed_analysis: bool = True):
        """Compare IQR vs Z-score outlier detection with detailed computations"""
        print(f"\n{'='*70}")
        print("OUTLIER DETECTION METHODS COMPARISON")
        print(f"{'='*70}")
        
        columns_to_analyze = [self.feature_col, self.target_col]
        comparison_results = {}
        
        for column in columns_to_analyze:
            print(f"\n🔍 ANALYZING {column.upper()}:")
            print("-" * 50)
            
            data = self.df[column].values
            
            # IQR Method Calculations
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound_iqr = Q1 - 1.5 * IQR
            upper_bound_iqr = Q3 + 1.5 * IQR
            iqr_outliers = (data < lower_bound_iqr) | (data > upper_bound_iqr)
            
            # Z-score Method Calculations
            mean_val = np.mean(data)
            std_val = np.std(data)
            z_scores = np.abs((data - mean_val) / std_val)
            zscore_outliers = z_scores > 3.0
            
            comparison_results[column] = {
                'iqr': {
                    'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
                    'lower_bound': lower_bound_iqr, 'upper_bound': upper_bound_iqr,
                    'outliers': iqr_outliers, 'count': iqr_outliers.sum(),
                    'outlier_values': data[iqr_outliers]
                },
                'zscore': {
                    'mean': mean_val, 'std': std_val, 'threshold': 3.0,
                    'z_scores': z_scores, 'outliers': zscore_outliers, 'count': zscore_outliers.sum(),
                    'outlier_values': data[zscore_outliers]
                }
            }
            
            print(f"IQR METHOD CALCULATIONS:")
            print(f"  Q1 (25th percentile): {Q1:.4f}")
            print(f"  Q3 (75th percentile): {Q3:.4f}")
            print(f"  IQR (Q3 - Q1):       {IQR:.4f}")
            print(f"  Lower bound (Q1 - 1.5×IQR): {lower_bound_iqr:.4f}")
            print(f"  Upper bound (Q3 + 1.5×IQR): {upper_bound_iqr:.4f}")
            print(f"  Outliers detected: {iqr_outliers.sum()} ({iqr_outliers.mean()*100:.1f}%)")
            
            print(f"\nZ-SCORE METHOD CALCULATIONS:")
            print(f"  Mean (μ): {mean_val:.4f}")
            print(f"  Std (σ):  {std_val:.4f}")
            print(f"  Threshold: |z| > 3.0")
            print(f"  Max |z-score|: {np.max(z_scores):.4f}")
            print(f"  Outliers detected: {zscore_outliers.sum()} ({zscore_outliers.mean()*100:.1f}%)")
            
            if detailed_analysis and (iqr_outliers.sum() > 0 or zscore_outliers.sum() > 0):
                print(f"\n📋 OUTLIER VALUES DETECTED:")
                
                if iqr_outliers.sum() > 0:
                    print(f"  IQR outliers: {sorted(data[iqr_outliers])}")
                
                if zscore_outliers.sum() > 0:
                    print(f"  Z-score outliers: {sorted(data[zscore_outliers])}")
                
                # Find overlapping outliers
                overlap = iqr_outliers & zscore_outliers
                if overlap.sum() > 0:
                    print(f"  Both methods agree on: {sorted(data[overlap])}")
                
                # Find method-specific outliers
                iqr_only = iqr_outliers & ~zscore_outliers
                zscore_only = zscore_outliers & ~iqr_outliers
                
                if iqr_only.sum() > 0:
                    print(f"  IQR only: {sorted(data[iqr_only])}")
                if zscore_only.sum() > 0:
                    print(f"  Z-score only: {sorted(data[zscore_only])}")
        
        # Store results
        self.outlier_detection_results = comparison_results
        
        # Visualization
        self.visualize_outlier_comparison(comparison_results)
        
        return comparison_results
    
    def visualize_outlier_comparison(self, comparison_results):
        """Visualize outlier detection comparison"""
        columns = list(comparison_results.keys())
        fig, axes = plt.subplots(len(columns), 3, figsize=(18, 6*len(columns)))
        if len(columns) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{self.dataset_name.title()} - Outlier Detection Methods Comparison', fontsize=16)
        
        for i, column in enumerate(columns):
            data = self.df[column].values
            results = comparison_results[column]
            
            # Box plot comparison
            axes[i,0].boxplot([data], labels=[column])
            axes[i,0].set_title(f'{column} - Box Plot\n(IQR: {results["iqr"]["count"]} outliers)')
            axes[i,0].grid(True, alpha=0.3)
            
            # Histogram with IQR outliers
            axes[i,1].hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', label='Normal')
            if results['iqr']['count'] > 0:
                axes[i,1].hist(results['iqr']['outlier_values'], bins=30, alpha=0.8, 
                              color='red', edgecolor='black', label=f'IQR Outliers ({results["iqr"]["count"]})')
            
            # Mark IQR bounds
            axes[i,1].axvline(results['iqr']['lower_bound'], color='red', linestyle='--', alpha=0.7, label='IQR Bounds')
            axes[i,1].axvline(results['iqr']['upper_bound'], color='red', linestyle='--', alpha=0.7)
            axes[i,1].set_title(f'{column} - IQR Method')
            axes[i,1].set_xlabel(column)
            axes[i,1].set_ylabel('Frequency')
            axes[i,1].legend()
            axes[i,1].grid(True, alpha=0.3)
            
            # Z-score plot
            z_scores = results['zscore']['z_scores']
            normal_mask = ~results['zscore']['outliers']
            outlier_mask = results['zscore']['outliers']
            
            axes[i,2].scatter(range(len(z_scores)), z_scores, alpha=0.6, color='blue', label='Normal')
            if outlier_mask.sum() > 0:
                outlier_indices = np.where(outlier_mask)[0]
                axes[i,2].scatter(outlier_indices, z_scores[outlier_mask], alpha=0.8, 
                                 color='red', s=60, label=f'Z-score Outliers ({results["zscore"]["count"]})')
            
            axes[i,2].axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='Threshold = 3')
            axes[i,2].axhline(y=-3.0, color='red', linestyle='--', linewidth=2)
            axes[i,2].set_title(f'{column} - Z-score Method')
            axes[i,2].set_xlabel('Data Point Index')
            axes[i,2].set_ylabel('|Z-score|')
            axes[i,2].legend()
            axes[i,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_normalization_performance(self, X_train, X_test, y_train, y_test):
        """Compare model performance with different normalization methods"""
        print(f"\n{'='*70}")
        print("NORMALIZATION METHODS PERFORMANCE COMPARISON")
        print(f"{'='*70}")
        
        normalization_methods = ['standard', 'minmax']
        results = {}
        
        for method in normalization_methods:
            print(f"\n🔄 Training with {method.upper()} normalization...")
            
            # Train model with specific normalization
            model = SimpleLinearRegressionFromScratch(
                learning_rate=0.01, 
                max_epochs=1000, 
                normalization=method
            )
            model.fit_batch(X_train, y_train, verbose=False)
            
            # Evaluate model
            train_r2 = model.score(X_train, y_train)
            test_r2 = model.score(X_test, y_test)
            train_mse = model.mean_squared_error(X_train, y_train)
            test_mse = model.mean_squared_error(X_test, y_test)
            
            # Get normalization statistics
            norm_stats = model.get_normalization_stats()
            
            results[method] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'norm_stats': norm_stats,
                'final_cost': model.cost_history[-1] if model.cost_history else 0,
                'epochs_to_converge': len(model.cost_history)
            }
        
        # Display detailed comparison
        print("\n📊 PERFORMANCE COMPARISON:")
        print("-" * 80)
        print(f"{'Method':<12} {'Train R²':<10} {'Test R²':<10} {'Train MSE':<12} {'Test MSE':<12} {'Epochs':<8}")
        print("-" * 80)
        
        for method, result in results.items():
            print(f"{method.title():<12} {result['train_r2']:<10.4f} {result['test_r2']:<10.4f} "
                  f"{result['train_mse']:<12.2f} {result['test_mse']:<12.2f} {result['epochs_to_converge']:<8}")
        
        # Display normalization parameters
        print(f"\n🔧 NORMALIZATION PARAMETERS:")
        print("-" * 50)
        for method, result in results.items():
            stats = result['norm_stats']
            print(f"\n{method.upper()} SCALING:")
            if method == 'standard':
                print(f"  Feature: μ = {stats['X_mean']:.4f}, σ = {stats['X_std']:.4f}")
                print(f"  Target:  μ = {stats['y_mean']:.4f}, σ = {stats['y_std']:.4f}")
            else:  # minmax
                print(f"  Feature: min = {stats['X_min']:.4f}, max = {stats['X_max']:.4f}")
                print(f"  Target:  min = {stats['y_min']:.4f}, max = {stats['y_max']:.4f}")
        
        # Visualize comparison
        self.visualize_normalization_performance(X_test, y_test, results)
        
        return results
    
    def visualize_normalization_performance(self, X_test, y_test, results):
        """Visualize normalization performance comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.dataset_name.title()} - Normalization Performance Comparison', fontsize=16)
        
        methods = list(results.keys())
        colors = ['blue', 'red']
        
        for i, (method, color) in enumerate(zip(methods, colors)):
            result = results[method]
            model = result['model']
            
            # Predictions vs Actual
            y_pred = model.predict(X_test)
            
            axes[0, i].scatter(y_test, y_pred, alpha=0.6, color=color, s=50)
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            axes[0, i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
            
            axes[0, i].set_xlabel('Actual Values')
            axes[0, i].set_ylabel('Predicted Values')
            axes[0, i].set_title(f'{method.title()} Scaling\nR² = {result["test_r2"]:.4f}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Training curves
            axes[1, i].plot(model.epoch_history, model.cost_history, color=color, linewidth=2)
            axes[1, i].set_xlabel('Epochs')
            axes[1, i].set_ylabel('Normalized Cost')
            axes[1, i].set_title(f'{method.title()} - Training Curve')
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline with comparisons"""
        if not self.load_data():
            return None
        
        # Store original data
        self.df_original = self.df.copy()
            
        # Explore data
        self.explore_data()
        
        # Feature importance for housing
        if self.dataset_name.lower() == 'housing':
            self.analyze_feature_importance()
        
        # Get original data for comparisons
        X_original = self.df[self.feature_col].values
        y_original = self.df[self.target_col].values
        
        # NORMALIZATION COMPARISON
        normalization_results = self.compare_normalization_methods(X_original, y_original)
        
        # OUTLIER DETECTION COMPARISON
        outlier_results = self.compare_outlier_detection_methods(detailed_analysis=True)
        
        # Remove outliers using IQR method (for consistency with original code)
        outliers_removed = self.remove_outliers(method='iqr', feature_threshold=1.5, target_threshold=1.5)
        
        # Compare performance with and without outliers
        if outliers_removed > 0:
            comparison_results = self.compare_with_without_outliers()
        
        # Split cleaned data
        X_train, X_test, y_train, y_test = self.split_data()
        print(f"\nCleaned data split: {len(X_train)} training, {len(X_test)} test samples")
        
        # NORMALIZATION PERFORMANCE COMPARISON
        normalization_performance = self.compare_normalization_performance(X_train, X_test, y_train, y_test)
        
        # Train models on cleaned data
        print(f"\n{'='*60}")
        print("MODEL TRAINING ON CLEANED DATA")
        print(f"{'='*60}")
        
        # Batch model
        batch_model = SimpleLinearRegressionFromScratch(learning_rate=0.01, max_epochs=1000)
        batch_model.fit_batch(X_train, y_train, verbose=False)
        
        # Online model
        online_model = SimpleLinearRegressionFromScratch(learning_rate=0.001, max_epochs=500)
        online_model.fit_online(X_train, y_train, verbose=False)
        
        # Evaluate models
        models = {'Batch': batch_model, 'Online': online_model}
        results = {}
        
        print("\nModel Performance Comparison:")
        print("-" * 65)
        print(f"{'Model':<12} {'Train R²':<10} {'Test R²':<10} {'Train MSE':<12} {'Test MSE':<12}")
        print("-" * 65)
        
        for name, model in models.items():
            train_r2 = model.score(X_train, y_train)
            test_r2 = model.score(X_test, y_test)
            train_mse = model.mean_squared_error(X_train, y_train)
            test_mse = model.mean_squared_error(X_test, y_test)
            
            results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_pred': model.predict(X_train),
                'test_pred': model.predict(X_test)
            }
            
            print(f"{name:<12} {train_r2:<10.4f} {test_r2:<10.4f} {train_mse:<12.2f} {test_mse:<12.2f}")
        
        # Visualize results
        self.visualize_results(X_train, X_test, y_train, y_test, results)
        
        # Make predictions on new data
        self.demonstrate_predictions(models)
        
        return {
            'normalization_results': normalization_results,
            'outlier_results': outlier_results,
            'normalization_performance': normalization_performance
        }

    def visualize_results(self, X_train, X_test, y_train, y_test, results):
        """Visualize model results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.dataset_name.title()} - Model Comparison Results', fontsize=16)
        
        colors = ['blue', 'red']
        models = list(results.keys())
        
        for i, (model_name, color) in enumerate(zip(models, colors)):
            model_results = results[model_name]
            
            # Actual vs Predicted
            axes[0, i].scatter(y_test, model_results['test_pred'], alpha=0.6, color=color, s=50)
            
            # Perfect prediction line
            min_val = min(y_test.min(), model_results['test_pred'].min())
            max_val = max(y_test.max(), model_results['test_pred'].max())
            axes[0, i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
            
            axes[0, i].set_xlabel('Actual Values')
            axes[0, i].set_ylabel('Predicted Values')
            axes[0, i].set_title(f'{model_name} Learning\nR² = {model_results["test_r2"]:.4f}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Residual plots
            residuals = y_test - model_results['test_pred']
            axes[1, i].scatter(model_results['test_pred'], residuals, alpha=0.6, color=color)
            axes[1, i].axhline(y=0, color='k', linestyle='--', linewidth=2)
            axes[1, i].set_xlabel('Predicted Values')
            axes[1, i].set_ylabel('Residuals')
            axes[1, i].set_title(f'Residual Plot - {model_name}')
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Training curves comparison
        plt.figure(figsize=(12, 6))
        for model_name in models:
            model = results[model_name]['model']
            plt.plot(model.epoch_history, model.cost_history, label=f'{model_name} Learning', linewidth=2)
        
        plt.xlabel('Epochs')
        plt.ylabel('Cost (Normalized)')
        plt.title(f'Training Curves Comparison - {self.dataset_name.title()}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.show()
    
    def demonstrate_predictions(self, models):
        """Demonstrate predictions on new data"""
        print(f"\n{'='*60}")
        print("PREDICTIONS ON NEW DATA")
        print(f"{'='*60}")
        
        if self.dataset_name.lower() == 'housing':
            new_data = np.array([3000, 5000, 7000, 10000])
            print("Predicting house prices for new areas (sq ft):")
            unit = "INR"
        else:
            new_data = np.array([50, 100, 150, 200])
            print("Predicting sales for new TV advertising spending:")
            unit = "units"
        
        print(f"\n{self.feature_col:<10} {'Batch Model':<15} {'Online Model':<15}")
        print("-" * 45)
        
        for value in new_data:
            batch_pred = models['Batch'].predict(np.array([value]))[0]
            online_pred = models['Online'].predict(np.array([value]))[0]
            print(f"{value:<10.0f} {batch_pred:<15.2f} {online_pred:<15.2f}")

def get_user_input():
    """Get user choices for analysis parameters"""
    print("🎯 LINEAR REGRESSION ANALYSIS - USER CONFIGURATION")
    print("=" * 60)
    
    # Dataset selection
    print("\n1️⃣ SELECT DATASET:")
    print("   1. Housing Dataset (area vs price)")
    print("   2. Advertising Dataset (TV vs sales)")
    
    while True:
        try:
            dataset_choice = int(input("\nEnter your choice (1 or 2): "))
            if dataset_choice in [1, 2]:
                dataset_name = 'Housing' if dataset_choice == 1 else 'advertising'
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Learning method selection
    print("\n2️⃣ SELECT LEARNING METHOD:")
    print("   1. Batch Gradient Descent")
    print("   2. Online (Stochastic) Gradient Descent")
    
    while True:
        try:
            learning_choice = int(input("\nEnter your choice (1 or 2): "))
            if learning_choice in [1, 2]:
                learning_method = 'batch' if learning_choice == 1 else 'online'
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Normalization method selection
    print("\n3️⃣ SELECT NORMALIZATION METHOD:")
    print("   1. Standard Scaling (Z-score normalization)")
    print("   2. Min-Max Scaling (0-1 normalization)")
    
    while True:
        try:
            norm_choice = int(input("\nEnter your choice (1 or 2): "))
            if norm_choice in [1, 2]:
                normalization = 'standard' if norm_choice == 1 else 'minmax'
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Outlier detection method
    print("\n4️⃣ SELECT OUTLIER DETECTION METHOD:")
    print("   1. IQR Method (Interquartile Range)")
    print("   2. Z-score Method")
    print("   3. Modified Z-score Method")
    print("   4. No outlier removal")
    
    while True:
        try:
            outlier_choice = int(input("\nEnter your choice (1, 2, 3, or 4): "))
            if outlier_choice in [1, 2, 3, 4]:
                outlier_methods = ['iqr', 'zscore', 'modified_zscore', 'none']
                outlier_method = outlier_methods[outlier_choice - 1]
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
    
    # Get threshold for outlier detection (if applicable)
    outlier_threshold = None
    if outlier_method != 'none':
        print(f"\n5️⃣ SET OUTLIER DETECTION THRESHOLD:")
        if outlier_method == 'iqr':
            print("   Default IQR multiplier: 1.5 (recommended range: 1.0 - 2.0)")
            default_threshold = 1.5
        elif outlier_method == 'zscore':
            print("   Default Z-score threshold: 3.0 (recommended range: 2.0 - 4.0)")
            default_threshold = 3.0
        elif outlier_method == 'modified_zscore':
            print("   Default Modified Z-score threshold: 3.5 (recommended range: 3.0 - 4.0)")
            default_threshold = 3.5
        
        while True:
            try:
                threshold_input = input(f"\nEnter threshold (press Enter for default {default_threshold}): ").strip()
                if threshold_input == "":
                    outlier_threshold = default_threshold
                    break
                else:
                    outlier_threshold = float(threshold_input)
                    if outlier_threshold > 0:
                        break
                    else:
                        print("❌ Threshold must be positive.")
            except ValueError:
                print("❌ Invalid input. Please enter a valid number.")
    
    return {
        'dataset': dataset_name,
        'learning_method': learning_method,
        'normalization': normalization,
        'outlier_method': outlier_method,
        'outlier_threshold': outlier_threshold
    }

def run_custom_analysis(config):
    """Run analysis based on user configuration"""
    print(f"\n{'='*70}")
    print("RUNNING CUSTOM ANALYSIS")
    print(f"{'='*70}")
    
    print(f"\n📋 CONFIGURATION SUMMARY:")
    print(f"   Dataset: {config['dataset']}")
    print(f"   Learning Method: {config['learning_method'].title()}")
    print(f"   Normalization: {config['normalization'].title()}")
    print(f"   Outlier Detection: {config['outlier_method'].title()}")
    if config['outlier_threshold'] is not None:
        print(f"   Outlier Threshold: {config['outlier_threshold']}")
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer(config['dataset'])
    
    # Load and explore data
    if not analyzer.load_data():
        return None
    
    # Store original data
    analyzer.df_original = analyzer.df.copy()
    
    # Basic data exploration
    analyzer.explore_data()
    
    # Feature importance for housing dataset
    if config['dataset'].lower() == 'housing':
        analyzer.analyze_feature_importance()
    
    # Handle outlier removal
    outliers_removed = 0
    if config['outlier_method'] != 'none':
        print(f"\n{'='*60}")
        print("OUTLIER DETECTION AND REMOVAL")
        print(f"{'='*60}")
        
        # Visualize outliers before removal
        analyzer.visualize_outliers()
        
        # Remove outliers based on selected method and threshold
        outliers_removed = analyzer.remove_outliers(
            method=config['outlier_method'],
            feature_threshold=config['outlier_threshold'],
            target_threshold=config['outlier_threshold']
        )
    
    # Split data
    X_train, X_test, y_train, y_test = analyzer.split_data()
    print(f"\nData split: {len(X_train)} training, {len(X_test)} test samples")
    
    # Train model with selected configuration
    print(f"\n{'='*60}")
    print("MODEL TRAINING")
    print(f"{'='*60}")
    
    # Set hyperparameters based on learning method
    if config['learning_method'] == 'batch':
        learning_rate = 0.01
        max_epochs = 1000
    else:  # online
        learning_rate = 0.001
        max_epochs = 500
    
    # Create and train model
    model = SimpleLinearRegressionFromScratch(
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        normalization=config['normalization']
    )
    
    # Train based on selected method
    if config['learning_method'] == 'batch':
        model.fit_batch(X_train, y_train, verbose=True)
    else:
        model.fit_online(X_train, y_train, verbose=True)
    
    # Evaluate model
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    train_mse = model.mean_squared_error(X_train, y_train)
    test_mse = model.mean_squared_error(X_test, y_test)
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print("-" * 40)
    print(f"Training R² Score:   {train_r2:.4f}")
    print(f"Test R² Score:       {test_r2:.4f}")
    print(f"Training MSE:        {train_mse:.2f}")
    print(f"Test MSE:            {test_mse:.2f}")
    print(f"Epochs to converge:  {len(model.cost_history)}")
    
    # Get normalization statistics
    norm_stats = model.get_normalization_stats()
    print(f"\n🔧 NORMALIZATION PARAMETERS:")
    print("-" * 40)
    if config['normalization'] == 'standard':
        print(f"Method: Standard Scaling (Z-score)")
        print(f"Feature: μ = {norm_stats['X_mean']:.4f}, σ = {norm_stats['X_std']:.4f}")
        print(f"Target:  μ = {norm_stats['y_mean']:.4f}, σ = {norm_stats['y_std']:.4f}")
    else:
        print(f"Method: Min-Max Scaling")
        print(f"Feature: min = {norm_stats['X_min']:.4f}, max = {norm_stats['X_max']:.4f}")
        print(f"Target:  min = {norm_stats['y_min']:.4f}, max = {norm_stats['y_max']:.4f}")
    
    # Visualize results
    visualize_single_model_results(analyzer, model, X_train, X_test, y_train, y_test, config)
    
    # Make predictions on new data
    demonstrate_single_model_predictions(analyzer, model, config)
    
    return {
        'model': model,
        'performance': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse
        },
        'outliers_removed': outliers_removed
    }

def visualize_single_model_results(analyzer, model, X_train, X_test, y_train, y_test, config):
    """Visualize results for single model configuration"""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{analyzer.dataset_name.title()} - {config["learning_method"].title()} Learning with {config["normalization"].title()} Scaling', 
                 fontsize=16, fontweight='bold')
    
    # 1. Actual vs Predicted (Test Set)
    axes[0, 0].scatter(y_test, y_test_pred, alpha=0.7, color='blue', s=60)
    # Perfect prediction line
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'Predictions vs Actual\nR² = {model.score(X_test, y_test):.4f}')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Residual Plot
    residuals = y_test - y_test_pred
    axes[0, 1].scatter(y_test_pred, residuals, alpha=0.7, color='green', s=60)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Training Curve
    axes[1, 0].plot(model.epoch_history, model.cost_history, color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Normalized Cost')
    axes[1, 0].set_title('Training Curve')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # 4. Feature vs Target with Regression Line
    axes[1, 1].scatter(X_test, y_test, alpha=0.6, color='red', label='Actual', s=50)
    
    # Sort for smooth line plotting
    sorted_indices = np.argsort(X_test)
    X_test_sorted = X_test[sorted_indices]
    y_pred_sorted = y_test_pred[sorted_indices]
    
    axes[1, 1].plot(X_test_sorted, y_pred_sorted, color='blue', linewidth=2, label='Predicted')
    axes[1, 1].set_xlabel(analyzer.feature_col)
    axes[1, 1].set_ylabel(analyzer.target_col)
    axes[1, 1].set_title('Regression Line on Test Data')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def demonstrate_single_model_predictions(analyzer, model, config):
    """Demonstrate predictions for single model"""
    print(f"\n{'='*60}")
    print("PREDICTIONS ON NEW DATA")
    print(f"{'='*60}")
    
    if analyzer.dataset_name.lower() == 'housing':
        new_data = np.array([3000, 5000, 7000, 10000, 12000])
        print("🏠 Predicting house prices for new areas (sq ft):")
        unit = "INR"
    else:
        new_data = np.array([50, 100, 150, 200, 250])
        print("📺 Predicting sales for new TV advertising spending:")
        unit = "units"
    
    print(f"\n{analyzer.feature_col:<12} Predicted {analyzer.target_col}")
    print("-" * 35)
    
    for value in new_data:
        prediction = model.predict(np.array([value]))[0]
        print(f"{value:<12.0f} {prediction:<15.2f}")

def main():
    print("🚀 INTERACTIVE LINEAR REGRESSION ANALYSIS")
    print("Choose your own configuration for custom analysis!")
    
    # Get user configuration
    config = get_user_input()
    
    # Run analysis based on user choices
    results = run_custom_analysis(config)
    
    if results:
        print(f"\n{'='*70}")
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*70}")
        
        print(f"\n🎯 SUMMARY OF YOUR ANALYSIS:")
        print(f"   • Dataset: {config['dataset']}")
        print(f"   • Learning Method: {config['learning_method'].title()}")
        print(f"   • Normalization: {config['normalization'].title()}")
        print(f"   • Outlier Detection: {config['outlier_method'].title()}")
        if config['outlier_threshold'] is not None:
            print(f"   • Outliers Removed: {results['outliers_removed']}")
        print(f"   • Final Test R² Score: {results['performance']['test_r2']:.4f}")
        print(f"   • Final Test MSE: {results['performance']['test_mse']:.2f}")
        
        print(f"\n💡 INSIGHTS:")
        if results['performance']['test_r2'] > 0.8:
            print("   • Excellent model performance! 🌟")
        elif results['performance']['test_r2'] > 0.6:
            print("   • Good model performance! 👍")
        elif results['performance']['test_r2'] > 0.4:
            print("   • Moderate model performance. Consider tuning parameters. ⚙️")
        else:
            print("   • Low model performance. Try different configuration. 🔧")
        
        if config['learning_method'] == 'batch':
            print("   • Batch learning provides stable convergence.")
        else:
            print("   • Online learning adapts quickly to new data points.")
        
        if config['normalization'] == 'standard':
            print("   • Standard scaling works well for normally distributed data.")
        else:
            print("   • Min-max scaling bounds features to [0,1] range.")
    
    # Ask if user wants to run another analysis
    print(f"\n🔄 Would you like to run another analysis with different parameters?")
    while True:
        choice = input("Enter 'y' for yes or 'n' for no: ").lower().strip()
        if choice in ['y', 'yes']:
            print("\n" + "="*70)
            main()  # Recursive call for another analysis
            break
        elif choice in ['n', 'no']:
            print("\n👋 Thank you for using the Interactive Linear Regression Analysis!")
            break
        else:
            print("❌ Invalid input. Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()
