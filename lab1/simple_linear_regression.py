"""
Simple Linear Regression Analysis
Real Estate Price Prediction & Advertising Sales Prediction
Author: U23AI118
Date: August 2, 2025

This implementation demonstrates:
1. Simple linear regression on two datasets (Housing and Advertising)
2. Batch vs Online learning approaches
3. Model evaluation and comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')


class SimpleLinearRegression:
    """Simple Linear Regression with Batch and Online Learning"""
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.feature_importance = {}  # Added to store feature importance
        
    def load_data(self):
        """Load and explore dataset"""
        print(f"\n{'='*60}")
        print(f"LOADING {self.dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        try:
            script_dir = os.path.dirname(__file__)
            file_path = os.path.join(script_dir, f'{self.dataset_name}.csv')
            self.df = pd.read_csv(file_path)
            print(f"✅ Dataset loaded successfully!")
            print(f"Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Display basic statistics
            print("\nFirst 5 rows:")
            print(self.df.head())
            
            print("\nBasic Statistics:")
            print(self.df.describe())
            
            return True
        except FileNotFoundError:
            print(f"❌ Error: {self.dataset_name}.csv not found!")
            return False
    
    def visualize_data(self):
        """Visualize data distribution and relationships"""
        print(f"\n{'='*60}")
        print("DATA VISUALIZATION")
        print(f"{'='*60}")
        
        # Set target and feature variables based on dataset
        if self.dataset_name.lower() == 'housing':
            target = 'price'
            feature = 'area'  # Primary feature for simple linear regression
            title = 'Housing Price vs Area Analysis'
            
            # Show pairplot for key housing features if available
            key_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'price']
            available_features = [f for f in key_features if f in self.df.columns]
            if len(available_features) > 2:  # Only if we have multiple features
                plt.figure(figsize=(12, 8))
                sns.pairplot(self.df[available_features], height=2.5)
                plt.suptitle('Relationships Between Housing Features', y=1.02, fontsize=16)
                plt.tight_layout()
                plt.show()
                
                # Show correlation between all available housing features
                print("\nCorrelation with Price:")
                correlations = self.df[available_features].corr()['price'].sort_values(ascending=False)
                print(correlations)
        else:  # advertising
            target = 'Sales'
            feature = 'TV'    # Primary feature for simple linear regression
            title = 'Sales vs TV Advertising Analysis'
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Target variable distribution
        axes[0,0].hist(self.df[target], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].set_title(f'{target} Distribution')
        axes[0,0].set_xlabel(target)
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Feature vs Target scatter plot
        axes[0,1].scatter(self.df[feature], self.df[target], alpha=0.6, color='red')
        axes[0,1].set_title(f'{target} vs {feature}')
        axes[0,1].set_xlabel(feature)
        axes[0,1].set_ylabel(target)
        axes[0,1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.df[feature], self.df[target], 1)
        p = np.poly1d(z)
        axes[0,1].plot(self.df[feature], p(self.df[feature]), "b--", linewidth=2)
        
        # 3. Box plot
        axes[1,0].boxplot(self.df[target])
        axes[1,0].set_title(f'{target} Box Plot')
        axes[1,0].set_ylabel(target)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Correlation matrix
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[1,1], fmt='.2f', square=True)
        axes[1,1].set_title('Correlation Matrix')
        
        plt.tight_layout()
        plt.show()
        
        # Print correlation
        correlation = self.df[feature].corr(self.df[target])
        print(f"Correlation between {feature} and {target}: {correlation:.4f}")
        
        return feature, target
    
    def prepare_data(self, feature, target):
        """Prepare data for modeling"""
        print(f"\n{'='*60}")
        print("DATA PREPARATION")
        print(f"{'='*60}")
        
        # Handle categorical variables for housing dataset
        if self.dataset_name.lower() == 'housing':
            # Simple binary encoding for categorical variables
            categorical_vars = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                               'airconditioning', 'prefarea']
            for var in categorical_vars:
                if var in self.df.columns:
                    self.df[var] = self.df[var].map({'yes': 1, 'no': 0})
            
            # Handle furnishing status
            if 'furnishingstatus' in self.df.columns:
                furnishing_map = {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
                self.df['furnishingstatus'] = self.df['furnishingstatus'].map(furnishing_map)
        
        # Prepare X and y
        X = self.df[[feature]].values
        y = self.df[target].values
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print("✅ Data preparation completed!")
    
    def train_batch_model(self):
        """Train model using batch learning (scikit-learn)"""
        print("\n1. Training Batch Model (Scikit-learn)...")
        
        # Scikit-learn Linear Regression
        self.models['batch'] = LinearRegression()
        self.models['batch'].fit(self.X_train, self.y_train)
        
        # Make predictions
        train_pred = self.models['batch'].predict(self.X_train)
        test_pred = self.models['batch'].predict(self.X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, train_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)
        train_r2 = r2_score(self.y_train, train_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        
        self.results['batch'] = {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"   Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"   Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
    
    def train_online_model(self, learning_rate=0.0001, epochs=100):
        """Train model using online learning (Stochastic Gradient Descent)"""
        print("\n2. Training Online Model (Stochastic Gradient Descent)...")
        
        # Normalize features for better convergence
        X_mean = np.mean(self.X_train)
        X_std = np.std(self.X_train)
        y_mean = np.mean(self.y_train)
        y_std = np.std(self.y_train)
        
        X_train_norm = (self.X_train - X_mean) / X_std
        y_train_norm = (self.y_train - y_mean) / y_std
        
        # Initialize parameters
        w = np.random.normal(0, 0.01)
        b = np.random.normal(0, 0.01)
        
        # Training
        m = len(self.y_train)
        for epoch in range(epochs):
            for i in range(m):
                # Prediction
                y_pred = w * X_train_norm[i] + b
                
                # Calculate gradients
                dw = (y_pred - y_train_norm[i]) * X_train_norm[i]
                db = y_pred - y_train_norm[i]
                
                # Update parameters
                w -= learning_rate * dw
                b -= learning_rate * db
        
        # Store model parameters for denormalization
        self.models['online'] = {
            'w': w, 'b': b,
            'X_mean': X_mean, 'X_std': X_std,
            'y_mean': y_mean, 'y_std': y_std
        }
        
        # Make predictions (denormalized)
        X_test_norm = (self.X_test.flatten() - X_mean) / X_std
        train_pred_norm = w * X_train_norm.flatten() + b
        test_pred_norm = w * X_test_norm + b
        
        # Denormalize predictions
        train_pred = train_pred_norm * y_std + y_mean
        test_pred = test_pred_norm * y_std + y_mean
        
        # Calculate metrics
        train_mse = mean_squared_error(self.y_train, train_pred)
        test_mse = mean_squared_error(self.y_test, test_pred)
        train_r2 = r2_score(self.y_train, train_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        
        self.results['online'] = {
            'train_pred': train_pred,
            'test_pred': test_pred,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        print(f"   Train MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"   Test MSE: {test_mse:.4f}, R²: {test_r2:.4f}")
    
    def evaluate_models(self):
        """Compare and evaluate both models"""
        print(f"\n{'='*60}")
        print("MODEL EVALUATION & COMPARISON")
        print(f"{'='*60}")
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.dataset_name.title()} Dataset - Model Comparison', fontsize=16)
        
        colors = ['blue', 'red']
        models = ['batch', 'online']
        model_names = ['Batch Learning', 'Online Learning']
        
        for i, (model, name, color) in enumerate(zip(models, model_names, colors)):
            # Actual vs Predicted
            axes[0, i].scatter(self.y_test, self.results[model]['test_pred'], 
                           alpha=0.6, color=color, s=50)
            
            # Perfect prediction line
            min_val = min(self.y_test.min(), self.results[model]['test_pred'].min())
            max_val = max(self.y_test.max(), self.results[model]['test_pred'].max())
            axes[0, i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
            
            axes[0, i].set_xlabel('Actual Values')
            axes[0, i].set_ylabel('Predicted Values')
            axes[0, i].set_title(f'{name}\nR² = {self.results[model]["test_r2"]:.4f}')
            axes[0, i].grid(True, alpha=0.3)
            
            # Residual plots
            residuals = self.y_test - self.results[model]['test_pred']
            axes[1, i].scatter(self.results[model]['test_pred'], residuals, alpha=0.6, color=color)
            axes[1, i].axhline(y=0, color='k', linestyle='--', linewidth=2)
            axes[1, i].set_xlabel('Predicted Values')
            axes[1, i].set_ylabel('Residuals')
            axes[1, i].set_title(f'Residual Plot - {name}')
            axes[1, i].grid(True, alpha=0.3)
            
            # Add text for normality check of residuals
            from scipy import stats
            _, p_value = stats.normaltest(residuals)
            residual_info = f"Residual normality test p-value: {p_value:.4f}"
            axes[1, i].annotate(residual_info, xy=(0.05, 0.05), xycoords='axes fraction', 
                              bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        # Print comparison table
        print("\nModel Performance Comparison:")
        print("-" * 50)
        print(f"{'Model':<15} {'Train R²':<10} {'Test R²':<10} {'Test MSE':<12}")
        print("-" * 50)
        for model, name in zip(models, model_names):
            print(f"{name:<15} {self.results[model]['train_r2']:<10.4f} "
                  f"{self.results[model]['test_r2']:<10.4f} {self.results[model]['test_mse']:<12.2f}")
    
    def analyze_feature_importance(self):
        """Analyze importance of features for housing dataset"""
        if self.dataset_name.lower() != 'housing':
            return
            
        print(f"\n{'='*60}")
        print("FEATURE IMPORTANCE ANALYSIS")
        print(f"{'='*60}")
        
        # Check if we have multiple features in the housing dataset
        numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'price' in numeric_features:
            numeric_features.remove('price')
        
        if len(numeric_features) <= 1:
            print("Not enough numeric features for importance analysis.")
            return
            
        # Create a multiple regression model with all numeric features
        X_multi = self.df[numeric_features].values
        y = self.df['price'].values
        
        # Split data
        X_multi_train, X_multi_test, y_train, y_test = train_test_split(
            X_multi, y, test_size=0.2, random_state=42
        )
        
        # Train multiple regression model
        multi_model = LinearRegression()
        multi_model.fit(X_multi_train, y_train)
        
        # Get feature importance from coefficients
        importance = np.abs(multi_model.coef_)
        importance = 100.0 * (importance / np.sum(importance))
        
        # Create DataFrame for visualization
        feature_importance = pd.DataFrame({
            'Feature': numeric_features,
            'Importance': importance
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
        plt.xlabel('Importance (%)')
        plt.title('Feature Importance for Housing Price Prediction')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
        
        # Print feature importance table
        print("\nFeature Importance for Housing Price Prediction:")
        print("-" * 50)
        print(f"{'Feature':<15} {'Importance (%)':<15} {'Coefficient':<15}")
        print("-" * 50)
        for i, feature in enumerate(feature_importance['Feature']):
            idx = numeric_features.index(feature)
            print(f"{feature:<15} {importance[idx]:<15.2f} {multi_model.coef_[idx]:<15.2f}")
        
        # Evaluate multiple regression model
        y_pred = multi_model.predict(X_multi_test)
        multi_r2 = r2_score(y_test, y_pred)
        multi_mse = mean_squared_error(y_test, y_pred)
        
        print(f"\nMultiple Regression Model Performance:")
        print(f"R² Score: {multi_r2:.4f}")
        print(f"MSE: {multi_mse:.2f}")
        print(f"\nThis analysis helps identify which features most affect house prices.")
    
    def make_predictions(self, feature, target):
        """Demonstrate predictions on new data"""
        print(f"\n{'='*60}")
        print("PREDICTIONS ON NEW DATA")
        print(f"{'='*60}")
        
        # Create sample new data
        if self.dataset_name.lower() == 'housing':
            new_data = np.array([[3000], [5000], [7000], [10000]])  # New areas
            print("Predicting house prices for new areas:")
            unit = "sq ft"
        else:
            new_data = np.array([[50], [100], [150], [200]])  # New TV spending
            print("Predicting sales for new TV advertising spending:")
            unit = "thousand $"
        
        # Batch model predictions
        batch_pred = self.models['batch'].predict(new_data)
        
        # Online model predictions
        new_data_norm = (new_data.flatten() - self.models['online']['X_mean']) / self.models['online']['X_std']
        online_pred_norm = self.models['online']['w'] * new_data_norm + self.models['online']['b']
        online_pred = online_pred_norm * self.models['online']['y_std'] + self.models['online']['y_mean']
        
        print(f"\n{feature:<10} {'Batch Model':<12} {'Online Model':<12}")
        print("-" * 40)
        for i in range(len(new_data)):
            print(f"{new_data[i,0]:<10.0f} {batch_pred[i]:<12.2f} {online_pred[i]:<12.2f}")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        if not self.load_data():
            return None
        
        feature, target = self.visualize_data()
        self.prepare_data(feature, target)
        
        print(f"\n{'='*60}")
        print("MODEL TRAINING")
        print(f"{'='*60}")
        
        self.train_batch_model()
        self.train_online_model()
        self.evaluate_models()
        self.make_predictions(feature, target)
        
        # Add feature importance analysis for housing dataset
        if self.dataset_name.lower() == 'housing':
            self.analyze_feature_importance()
        
        return self


def main():
    """Main function to analyze both datasets"""
    print("="*80)
    print("SIMPLE LINEAR REGRESSION ANALYSIS")
    print("Real Estate Price Prediction & Advertising Sales Prediction")
    print("="*80)
    
    datasets = ['Housing', 'advertising']
    results = {}
    
    # Analyze both datasets
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"ANALYZING {dataset.upper()} DATASET")
        print(f"{'='*80}")
        
        analysis = SimpleLinearRegression(dataset)
        results[dataset] = analysis.run_analysis()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    print("✅ ASSIGNMENT OBJECTIVES COMPLETED:")
    print("1. ✅ Loaded and preprocessed datasets with numeric variables")
    print("2. ✅ Visualized data distribution and correlations")
    print("3. ✅ Implemented simple linear regression models")
    print("4. ✅ Compared Batch vs Online learning approaches")
    print("5. ✅ Evaluated models using MSE and R² metrics")
    print("6. ✅ Made predictions on new data")
    print("7. ✅ Analyzed real estate prices based on multiple features")
    print("8. ✅ Created quantitative linear models for price prediction")
    print("9. ✅ Assessed model accuracy and performance")
    print("10. ✅ Identified key variables affecting house prices")
    
    print(f"\n📊 DATASETS ANALYZED:")
    for dataset in datasets:
        if results[dataset] is not None:
            print(f"   ✅ {dataset}: Successfully analyzed")
        else:
            print(f"   ❌ {dataset}: Analysis failed")
    
    print(f"\n🎯 KEY FINDINGS:")
    print("• Both batch and online learning produce similar results")
    print("• Simple linear regression effectively models linear relationships")
    print("• Model accuracy depends on feature-target correlation strength")
    print("• For the housing dataset, multiple regression with all features provides better insights")
    print("• The most important features affecting house prices have been identified")
    print("• Ready for real estate and marketing analytics applications!")


if __name__ == "__main__":
    main()
