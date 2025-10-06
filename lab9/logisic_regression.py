import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class LogisticRegressionFromScratch:
    """
    Logistic Regression implementation from scratch using gradient descent
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """
        Train the logistic regression model
        
        Parameters:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # Calculate cost (log-likelihood)
            cost = self.compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
                
    def compute_cost(self, y_true, y_pred):
        """Compute logistic regression cost function"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

class EDAAnalyzer:
    """Class for comprehensive Exploratory Data Analysis"""
    
    def __init__(self, df):
        self.df = df
        
    def basic_info(self):
        """Display basic information about the dataset"""
        print("="*80)
        print("BASIC DATASET INFORMATION")
        print("="*80)
        
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Number of samples: {self.df.shape[0]}")
        print(f"Number of features: {self.df.shape[1]}")
        print("\nColumn Names and Data Types:")
        print(self.df.dtypes)
        
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nDataset Info:")
        print(self.df.info())
        
    def missing_values_analysis(self):
        """Analyze missing values in the dataset"""
        print("\n" + "="*80)
        print("MISSING VALUES ANALYSIS")
        print("="*80)
        
        missing_values = self.df.isnull().sum()
        missing_percentage = (missing_values / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing Count': missing_values,
            'Missing Percentage': missing_percentage
        })
        
        print(missing_df[missing_df['Missing Count'] > 0])
        
        if missing_df['Missing Count'].sum() == 0:
            print("No missing values found in the dataset!")
            
    def target_analysis(self):
        """Analyze the target variable distribution"""
        print("\n" + "="*80)
        print("TARGET VARIABLE ANALYSIS")
        print("="*80)
        
        target_col = 'y'
        print(f"Target variable: {target_col}")
        print(f"Target distribution:\n{self.df[target_col].value_counts()}")
        print(f"Target distribution (%):\n{self.df[target_col].value_counts(normalize=True) * 100}")
        
        # Plot target distribution
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        self.df[target_col].value_counts().plot(kind='bar', color=['lightcoral', 'lightblue'])
        plt.title('Target Variable Distribution (Count)')
        plt.xlabel('Target')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        plt.subplot(1, 2, 2)
        self.df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightcoral', 'lightblue'])
        plt.title('Target Variable Distribution (%)')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.show()
        
    def numerical_features_analysis(self):
        """Analyze numerical features"""
        print("\n" + "="*80)
        print("NUMERICAL FEATURES ANALYSIS")
        print("="*80)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"Numerical columns: {numerical_cols}")
        
        print("\nDescriptive Statistics:")
        print(self.df[numerical_cols].describe())
        
        # Plot distributions of numerical features
        if len(numerical_cols) > 0:
            n_cols = min(3, len(numerical_cols))
            n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 5 * n_rows))
            
            for i, col in enumerate(numerical_cols):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.hist(self.df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                
            plt.tight_layout()
            plt.show()
            
    def categorical_features_analysis(self):
        """Analyze categorical features"""
        print("\n" + "="*80)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*80)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        if 'y' in categorical_cols:
            categorical_cols.remove('y')  # Remove target variable
            
        print(f"Categorical columns: {categorical_cols}")
        
        for col in categorical_cols:
            print(f"\n{col} - Unique values: {self.df[col].nunique()}")
            print(self.df[col].value_counts().head(10))
            
        # Plot some key categorical features
        key_categorical = categorical_cols[:6]  # Plot first 6 categorical features
        
        if len(key_categorical) > 0:
            n_cols = 2
            n_rows = (len(key_categorical) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, 5 * n_rows))
            
            for i, col in enumerate(key_categorical):
                plt.subplot(n_rows, n_cols, i + 1)
                value_counts = self.df[col].value_counts()
                
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                    
                value_counts.plot(kind='bar', color='lightgreen')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                
            plt.tight_layout()
            plt.show()
            
    def correlation_analysis(self):
        """Analyze correlations between numerical features"""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) > 1:
            correlation_matrix = self.df[numerical_cols].corr()
            
            print("Correlation Matrix:")
            print(correlation_matrix)
            
            # Plot correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Correlation Matrix of Numerical Features')
            plt.tight_layout()
            plt.show()
            
    def feature_target_relationship(self):
        """Analyze relationship between features and target"""
        print("\n" + "="*80)
        print("FEATURE-TARGET RELATIONSHIP ANALYSIS")
        print("="*80)
        
        target_col = 'y'
        
        # Numerical features vs target
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) > 0:
            plt.figure(figsize=(15, 5 * ((len(numerical_cols) + 2) // 3)))
            
            for i, col in enumerate(numerical_cols):
                plt.subplot((len(numerical_cols) + 2) // 3, 3, i + 1)
                
                for target_val in self.df[target_col].unique():
                    subset = self.df[self.df[target_col] == target_val][col]
                    plt.hist(subset, alpha=0.7, label=f'{target_col}={target_val}', bins=20)
                
                plt.title(f'{col} vs {target_col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.legend()
                
            plt.tight_layout()
            plt.show()

class EvaluationMetrics:
    """Class for computing comprehensive evaluation metrics"""
    
    @staticmethod
    def compute_confusion_matrix(y_true, y_pred):
        """Compute confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    @staticmethod
    def compute_metrics_from_cm(cm):
        """Compute all metrics from confusion matrix"""
        tn, fp, fn, tp = cm.ravel()
        
        # Basic metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Additional metrics
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # Balanced accuracy
        balanced_accuracy = (recall + specificity) / 2
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall (Sensitivity)': recall,
            'Specificity': specificity,
            'F1-Score': f1,
            'Negative Predictive Value': npv,
            'False Positive Rate': fpr,
            'False Negative Rate': fnr,
            'Balanced Accuracy': balanced_accuracy
        }
    
    @staticmethod
    def plot_confusion_matrix(cm, classes=['No', 'Yes']):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_prob):
        """Plot ROC curve and compute AUC"""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return roc_auc

def preprocess_data(df):
    """
    Preprocess the dataset for machine learning
    
    Parameters:
    df: pandas DataFrame
    
    Returns:
    X: Feature matrix
    y: Target vector
    feature_names: List of feature names
    """
    print("\n" + "="*80)
    print("DATA PREPROCESSING")
    print("="*80)
    
    # Make a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Convert target variable to binary (0, 1)
    label_encoder_target = LabelEncoder()
    df_processed['y'] = label_encoder_target.fit_transform(df_processed['y'])
    
    print(f"Target encoding: {dict(zip(label_encoder_target.classes_, label_encoder_target.transform(label_encoder_target.classes_)))}")
    
    # Separate features and target
    X = df_processed.drop('y', axis=1)
    y = df_processed['y'].values
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns to encode: {categorical_cols}")
    print(f"Numerical columns: {numerical_cols}")
    
    # One-hot encoding for categorical variables
    if categorical_cols:
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    else:
        X_encoded = X.copy()
    
    print(f"Shape after encoding: {X_encoded.shape}")
    print(f"Feature names: {list(X_encoded.columns)}")
    
    return X_encoded.values, y, list(X_encoded.columns)

def main():
    """Main function to execute the complete pipeline"""
    print("="*80)
    print("LOGISTIC REGRESSION FROM SCRATCH - BANK MARKETING DATASET")
    print("="*80)
    
    # 1. Load the dataset
    print("\n1. LOADING DATASET...")
    data_path = "Assignment-9/bank-full.csv"
    df = pd.read_csv(data_path, delimiter=';')
    print(f"Dataset loaded successfully! Shape: {df.shape}")
    
    # 2. Exploratory Data Analysis
    print("\n2. PERFORMING EXPLORATORY DATA ANALYSIS...")
    eda = EDAAnalyzer(df)
    
    eda.basic_info()
    eda.missing_values_analysis()
    eda.target_analysis()
    eda.numerical_features_analysis()
    eda.categorical_features_analysis()
    eda.correlation_analysis()
    eda.feature_target_relationship()
    
    # 3. Data Preprocessing
    print("\n3. DATA PREPROCESSING...")
    X, y, feature_names = preprocess_data(df)
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Target distribution: {np.bincount(y)}")
    
    # 4. Train-Test Split (80:20) with random shuffling
    print("\n4. CREATING TRAIN-TEST SPLIT (80:20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training target distribution: {np.bincount(y_train)}")
    print(f"Test target distribution: {np.bincount(y_test)}")
    
    # 5. Feature Scaling
    print("\n5. FEATURE SCALING...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    
    # 6. Train Logistic Regression Model
    print("\n6. TRAINING LOGISTIC REGRESSION MODEL...")
    model = LogisticRegressionFromScratch(learning_rate=0.01, max_iterations=1000)
    model.fit(X_train_scaled, y_train)
    
    # Plot cost function
    plt.figure(figsize=(10, 6))
    plt.plot(model.cost_history)
    plt.title('Cost Function During Training')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # 7. Make Predictions
    print("\n7. MAKING PREDICTIONS...")
    
    # Training predictions
    y_train_pred = model.predict(X_train_scaled)
    y_train_prob = model.predict_proba(X_train_scaled)
    
    # Test predictions
    y_test_pred = model.predict(X_test_scaled)
    y_test_prob = model.predict_proba(X_test_scaled)
    
    print("Predictions completed!")
    
    # 8. Evaluation Metrics
    print("\n8. COMPUTING EVALUATION METRICS...")
    print("\n" + "="*80)
    print("TRAINING SET EVALUATION")
    print("="*80)
    
    # Training metrics
    cm_train = EvaluationMetrics.compute_confusion_matrix(y_train, y_train_pred)
    metrics_train = EvaluationMetrics.compute_metrics_from_cm(cm_train)
    
    print("Confusion Matrix (Training):")
    print(cm_train)
    
    print("\nDetailed Metrics (Training):")
    for metric, value in metrics_train.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot training confusion matrix
    EvaluationMetrics.plot_confusion_matrix(cm_train, classes=['No', 'Yes'])
    
    # Training ROC curve
    auc_train = EvaluationMetrics.plot_roc_curve(y_train, y_train_prob)
    print(f"\nTraining AUC: {auc_train:.4f}")
    
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    # Test metrics
    cm_test = EvaluationMetrics.compute_confusion_matrix(y_test, y_test_pred)
    metrics_test = EvaluationMetrics.compute_metrics_from_cm(cm_test)
    
    print("Confusion Matrix (Test):")
    print(cm_test)
    
    print("\nDetailed Metrics (Test):")
    for metric, value in metrics_test.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot test confusion matrix
    EvaluationMetrics.plot_confusion_matrix(cm_test, classes=['No', 'Yes'])
    
    # Test ROC curve
    auc_test = EvaluationMetrics.plot_roc_curve(y_test, y_test_prob)
    print(f"\nTest AUC: {auc_test:.4f}")
    
    # 9. Model Interpretation
    print("\n9. MODEL INTERPRETATION...")
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE (WEIGHTS)")
    print("="*80)
    
    # Get feature importance (absolute weights)
    feature_importance = np.abs(model.weights)
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("Top 20 Most Important Features:")
    print(feature_importance_df.head(20))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], color='skyblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Absolute Weight (Importance)')
    plt.title('Top 15 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    # 10. Summary Report
    print("\n10. SUMMARY REPORT...")
    print("\n" + "="*80)
    print("FINAL SUMMARY REPORT")
    print("="*80)
    
    print(f"Dataset: Bank Marketing (Portuguese Bank)")
    print(f"Total samples: {df.shape[0]}")
    print(f"Total features: {len(feature_names)} (after preprocessing)")
    print(f"Target classes: {dict(zip(['No', 'Yes'], np.bincount(y)))}")
    print(f"Train-Test split: 80-20")
    
    print(f"\nModel Performance:")
    print(f"Training Accuracy: {metrics_train['Accuracy']:.4f}")
    print(f"Test Accuracy: {metrics_test['Accuracy']:.4f}")
    print(f"Training AUC: {auc_train:.4f}")
    print(f"Test AUC: {auc_test:.4f}")
    
    print(f"\nTest Set Detailed Performance:")
    print(f"Precision: {metrics_test['Precision']:.4f}")
    print(f"Recall: {metrics_test['Recall (Sensitivity)']:.4f}")
    print(f"F1-Score: {metrics_test['F1-Score']:.4f}")
    print(f"Specificity: {metrics_test['Specificity']:.4f}")
    print(f"Balanced Accuracy: {metrics_test['Balanced Accuracy']:.4f}")
    
    overfitting_check = metrics_train['Accuracy'] - metrics_test['Accuracy']
    print(f"\nOverfitting Check:")
    print(f"Training - Test Accuracy Difference: {overfitting_check:.4f}")
    
    if overfitting_check > 0.05:
        print("⚠️  Model might be overfitting (difference > 5%)")
    else:
        print("✅ Model generalization looks good")
    
    print("\n" + "="*80)
    print("LOGISTIC REGRESSION IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
