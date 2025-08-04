import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Tuple, List
import os

warnings.filterwarnings('ignore')
plt.style.use('default')

class SimpleLinearRegressionFromScratch:
    """Simple Linear Regression implemented from scratch with batch and online learning"""
    
    def __init__(self, learning_rate: float = 0.001, max_epochs: int = 1000, tolerance: float = 1e-6):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        
        # Model parameters
        self.weight = None
        self.bias = None
        
        # Training history
        self.cost_history = []
        self.epoch_history = []
        
        # Data normalization parameters
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
    
    def _normalize_data(self, X: np.ndarray, y: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features and target for better convergence"""
        if fit:
            self.X_mean = np.mean(X)
            self.X_std = np.std(X)
            self.y_mean = np.mean(y)
            self.y_std = np.std(y)
        
        X_norm = (X - self.X_mean) / self.X_std if self.X_std != 0 else X - self.X_mean
        y_norm = (y - self.y_mean) / self.y_std if self.y_std != 0 else y - self.y_mean
        
        return X_norm, y_norm
    
    def _denormalize_predictions(self, y_pred_norm: np.ndarray) -> np.ndarray:
        """Denormalize predictions back to original scale"""
        return y_pred_norm * self.y_std + self.y_mean
    
    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error"""
        return np.mean((y_pred - y_true) ** 2)
    
    def fit_batch(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> None:
        """Train using batch gradient descent"""
        if verbose:
            print("Training with Batch Gradient Descent...")
        
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
        X_norm = (X - self.X_mean) / self.X_std if self.X_std != 0 else X - self.X_mean
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
        self.feature_col = None
        self.target_col = None
        
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
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        if not self.load_data():
            return None
            
        # Explore data
        self.explore_data()
        
        # Feature importance for housing
        if self.dataset_name.lower() == 'housing':
            self.analyze_feature_importance()
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data()
        print(f"\nData split: {len(X_train)} training, {len(X_test)} test samples")
        
        # Train models
        print(f"\n{'='*60}")
        print("MODEL TRAINING AND COMPARISON")
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
        
        return results
    
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

def main():
    print("SIMPLE LINEAR REGRESSION ANALYSIS FROM SCRATCH")    
    datasets = ['Housing', 'advertising']
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"ANALYZING {dataset.upper()} DATASET")
        print(f"{'='*80}")
        
        analyzer = DatasetAnalyzer(dataset)
        results = analyzer.run_complete_analysis()
        all_results[dataset] = results
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL ANALYSIS SUMMARY")
    print(f"{'='*80}")
    
    print("\n📊 COMPARATIVE RESULTS ACROSS DATASETS:")
    print("-" * 80)
    print(f"{'Dataset':<12} {'Model':<8} {'R² Score':<10} {'MSE':<15} {'Status':<15}")
    print("-" * 80)
    
    for dataset, results in all_results.items():
        if results:
            for model_name, result in results.items():
                r2 = result['test_r2']
                mse = result['test_mse']
                status = "Excellent" if r2 > 0.8 else "Good" if r2 > 0.6 else "Fair"
                print(f"{dataset:<12} {model_name:<8} {r2:<10.4f} {mse:<15.2f} {status:<15}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print("• Simple linear regression effectively models linear relationships")
    print("• Batch learning converges faster but requires more memory")
    print("• Online learning is more suitable for streaming data")
    print("• Housing prices show strong correlation with area")
    print("• Model accuracy depends on feature-target correlation strength")
    print("• Both approaches produce comparable results for these datasets")

if __name__ == "__main__":
    main()
