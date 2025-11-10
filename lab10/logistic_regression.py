import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    """
    Sigmoid activation function implemented from scratch
    """
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

class LogisticRegressionScratch:
    """
    Logistic Regression implementation from scratch using One-vs-Rest approach
    for multi-class classification
    """
    def __init__(self, learning_rate=0.01, max_epochs=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.num_classes = None
        self.cost_history = []

    def fit(self, X, y, num_classes):
        """
        Train the logistic regression model using gradient descent
        """
        n_samples, n_features = X.shape
        self.num_classes = num_classes
        
        # Initialize weights and bias for each class (One-vs-Rest)
        self.weights = np.zeros((num_classes, n_features))
        self.bias = np.zeros(num_classes)
        
        print(f"Training logistic regression for {num_classes} classes...")
        
        # Train one binary classifier for each class
        for class_idx in range(num_classes):
            print(f"Training classifier for class {class_idx}...")
            
            # Create binary labels (current class vs all others)
            y_binary = (y == class_idx).astype(float)
            
            # Initialize weights and bias for this class
            w = np.zeros(n_features)
            b = 0
            
            prev_cost = float('inf')
            
            for epoch in range(self.max_epochs):
                # Forward pass
                z = np.dot(X, w) + b
                predictions = sigmoid(z)
                
                # Compute cost (log-likelihood)
                cost = self._compute_cost(y_binary, predictions)
                
                # Compute gradients
                dw = (1/n_samples) * np.dot(X.T, (predictions - y_binary))
                db = (1/n_samples) * np.sum(predictions - y_binary)
                
                # Update parameters
                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db
                
                # Check for convergence
                if abs(prev_cost - cost) < self.tolerance:
                    print(f"  Converged at epoch {epoch}")
                    break
                prev_cost = cost
                
                # Print progress every 100 epochs
                if epoch % 100 == 0:
                    print(f"  Epoch {epoch}, Cost: {cost:.4f}")
            
            # Store the trained weights and bias
            self.weights[class_idx] = w
            self.bias[class_idx] = b
        
        print("Training completed!")

    def _compute_cost(self, y_true, y_pred):
        """
        Compute the logistic regression cost function
        """
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost

    def predict(self, X):
        """
        Make predictions for multiple samples
        """
        return np.array([self.predict_single(x) for x in X])

    def predict_single(self, x):
        """
        Make prediction for a single sample
        """
        # Compute scores for all classes
        scores = np.dot(self.weights, x) + self.bias
        probabilities = sigmoid(scores)
        
        # Return the class with highest probability
        return np.argmax(probabilities)
    
    def predict_proba(self, X):
        """
        Return prediction probabilities for all classes
        """
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, self.num_classes))
        
        for i, x in enumerate(X):
            scores = np.dot(self.weights, x) + self.bias
            probabilities[i] = sigmoid(scores)
        
        return probabilities

def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)
    """
    correct_predictions = np.sum(y_true == y_pred)
    total_predictions = len(y_true)
    return correct_predictions / total_predictions

def confusion_matrix(y_true, y_pred, num_classes):
    """
    Compute confusion matrix from scratch
    """
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    return matrix

def precision_score(y_true, y_pred, num_classes, average='macro'):
    """
    Calculate precision for multi-class classification
    Precision = TP / (TP + FP)
    """
    if average == 'macro':
        precisions = []
        for class_idx in range(num_classes):
            # True Positives: correctly predicted as this class
            tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
            # False Positives: incorrectly predicted as this class
            fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
            
            if tp + fp == 0:
                precision = 0.0  # No predictions for this class
            else:
                precision = tp / (tp + fp)
            precisions.append(precision)
        
        return np.mean(precisions)
    
    elif average == 'micro':
        # Calculate global TP and FP
        total_tp = np.sum(y_true == y_pred)
        total_fp = np.sum(y_true != y_pred)
        return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

def recall_score(y_true, y_pred, num_classes, average='macro'):
    """
    Calculate recall for multi-class classification
    Recall = TP / (TP + FN)
    """
    if average == 'macro':
        recalls = []
        for class_idx in range(num_classes):
            # True Positives: correctly predicted as this class
            tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
            # False Negatives: should have been predicted as this class but wasn't
            fn = np.sum((y_true == class_idx) & (y_pred != class_idx))
            
            if tp + fn == 0:
                recall = 0.0  # No actual instances of this class
            else:
                recall = tp / (tp + fn)
            recalls.append(recall)
        
        return np.mean(recalls)
    
    elif average == 'micro':
        # For micro-average in multi-class, recall equals accuracy
        return accuracy_score(y_true, y_pred)

def f1_score(y_true, y_pred, num_classes, average='macro'):
    """
    Calculate F1-score for multi-class classification
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    if average == 'macro':
        f1_scores = []
        for class_idx in range(num_classes):
            # Calculate precision and recall for this class
            tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
            fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
            fn = np.sum((y_true == class_idx) & (y_pred != class_idx))
            
            if tp + fp == 0:
                precision = 0.0
            else:
                precision = tp / (tp + fp)
            
            if tp + fn == 0:
                recall = 0.0
            else:
                recall = tp / (tp + fn)
            
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
            
            f1_scores.append(f1)
        
        return np.mean(f1_scores)
    
    elif average == 'micro':
        precision = precision_score(y_true, y_pred, num_classes, average='micro')
        recall = recall_score(y_true, y_pred, num_classes, average='micro')
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

def classification_report(y_true, y_pred, num_classes):
    """
    Generate a comprehensive classification report
    """
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, num_classes, average='macro')
    macro_recall = recall_score(y_true, y_pred, num_classes, average='macro')
    macro_f1 = f1_score(y_true, y_pred, num_classes, average='macro')
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro-averaged Precision: {macro_precision:.4f}")
    print(f"Macro-averaged Recall: {macro_recall:.4f}")
    print(f"Macro-averaged F1-Score: {macro_f1:.4f}")
    
    # Per-class metrics
    print(f"\nPer-class metrics:")
    print(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    
    for class_idx in range(num_classes):
        tp = np.sum((y_true == class_idx) & (y_pred == class_idx))
        fp = np.sum((y_true != class_idx) & (y_pred == class_idx))
        fn = np.sum((y_true == class_idx) & (y_pred != class_idx))
        support = np.sum(y_true == class_idx)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"{class_idx:<8} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }

def array_to_image(pixel_array, title=None, show_plot=True):
    """
    Convert a 784-element array into a 28x28 image and display it
    
    Args:
        pixel_array: 1D numpy array of size 784 containing pixel intensities
        title: Optional title for the plot
        show_plot: Whether to display the plot immediately
    
    Returns:
        img: 2D numpy array (28x28) representing the image
    """
    # Reshape 784-element array to 28x28 image
    img = pixel_array.reshape(28, 28)
    
    if show_plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.axis('off')
        if title:
            plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.show()
    
    return img

def predict_and_show(pixel_array, model, true_label=None):
    """
    Predict the label for a given pixel array and display the image
    
    Args:
        pixel_array: 1D numpy array of size 784 containing pixel intensities (0-255)
        model: Trained LogisticRegressionScratch model
        true_label: Optional true label for comparison
    
    Returns:
        predicted_label: The predicted digit (0-9)
    """
    # Normalize pixel values to [0, 1] range
    normalized_pixels = pixel_array / 255.0
    
    # Make prediction
    predicted_label = model.predict_single(normalized_pixels)
    
    # Get prediction probabilities for all classes
    scores = np.dot(model.weights, normalized_pixels) + model.bias
    probabilities = sigmoid(scores)
    
    # Create title with prediction information
    title = f"Predicted: {predicted_label}"
    if true_label is not None:
        title += f" | True: {true_label}"
        if predicted_label == true_label:
            title += " ✓"
        else:
            title += " ✗"
    
    # Add confidence information
    confidence = probabilities[predicted_label]
    title += f"\nConfidence: {confidence:.3f}"
    
    # Display the image
    array_to_image(pixel_array, title=title)
    
    # Print detailed prediction information
    print(f"Predicted Label: {predicted_label}")
    if true_label is not None:
        print(f"True Label: {true_label}")
        print(f"Correct: {'Yes' if predicted_label == true_label else 'No'}")
    print(f"Confidence: {confidence:.3f}")
    
    # Show top 3 predictions
    top_3_indices = np.argsort(probabilities)[::-1][:3]
    print("\nTop 3 predictions:")
    for i, idx in enumerate(top_3_indices):
        print(f"  {i+1}. Digit {idx}: {probabilities[idx]:.3f}")
    
    return predicted_label

def visualize_predictions(X_test, y_test, model, num_samples=10, random_seed=42):
    """
    Visualize predictions for multiple test samples
    
    Args:
        X_test: Test data (normalized)
        y_test: True test labels
        model: Trained model
        num_samples: Number of samples to visualize
        random_seed: Random seed for reproducible results
    """
    np.random.seed(random_seed)
    
    # Randomly select samples to visualize
    indices = np.random.choice(len(X_test), size=num_samples, replace=False)
    
    print(f"\nVisualizing {num_samples} random test samples:")
    print("="*50)
    
    correct_predictions = 0
    
    for i, idx in enumerate(indices):
        print(f"\nSample {i+1}/{num_samples} (Index: {idx})")
        
        # Convert normalized data back to 0-255 range for visualization
        pixel_data = (X_test[idx] * 255).astype(int)
        true_label = y_test[idx] if y_test is not None else None
        
        predicted_label = predict_and_show(pixel_data, model, true_label)
        
        if true_label is not None and predicted_label == true_label:
            correct_predictions += 1
        
        print("-" * 30)
    
    if y_test is not None:
        sample_accuracy = correct_predictions / num_samples
        print(f"\nSample Accuracy: {correct_predictions}/{num_samples} = {sample_accuracy:.3f}")

def save_test_predictions(test_predictions, filename="test_predictions.csv"):
    """
    Save test predictions to a CSV file
    
    Args:
        test_predictions: Array of predicted labels
        filename: Output filename
    """
    predictions_df = pd.DataFrame({
        'ImageId': range(1, len(test_predictions) + 1),
        'Label': test_predictions
    })
    
    predictions_df.to_csv(filename, index=False)
    print(f"\nTest predictions saved to '{filename}'")
    print(f"Number of predictions: {len(test_predictions)}")
    
    # Show distribution of predicted labels
    unique, counts = np.unique(test_predictions, return_counts=True)
    print("\nPrediction distribution:")
    for digit, count in zip(unique, counts):
        print(f"  Digit {digit}: {count} predictions")

def load_and_preprocess_data():
    """
    Load and preprocess the MNIST dataset
    """
    print("Loading MNIST dataset...")
    
    # Load training data
    train_df = pd.read_csv("MNIST_data/train.csv")
    print(f"Training data shape: {train_df.shape}")
    
    # Load test data
    test_df = pd.read_csv("MNIST_data/test.csv")
    print(f"Test data shape: {test_df.shape}")
    
    # Extract labels and features from training data
    y_train = train_df.iloc[:, 0].values  # First column is the label
    X_train = train_df.iloc[:, 1:].values  # Remaining columns are pixel values
    
    # Test data contains only pixel values (no labels)
    X_test = test_df.values
    
    # Normalize pixel values to [0, 1] range
    X_train = X_train.astype(float) / 255.0
    X_test = X_test.astype(float) / 255.0
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features per sample: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Show class distribution in training data
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTraining data class distribution:")
    for digit, count in zip(unique, counts):
        print(f"  Digit {digit}: {count} samples")
    
    return X_train, y_train, X_test

def main():
    """
    Main function to run the complete MNIST classification pipeline
    """
    print("MNIST Digit Classification using Logistic Regression")
    print("="*60)
    
    # Load and preprocess data
    X_train, y_train, X_test = load_and_preprocess_data()
    
    # Define parameters
    num_classes = 10  # Digits 0-9
    
    # Create and train the model
    print(f"\nInitializing Logistic Regression model...")
    model = LogisticRegressionScratch(learning_rate=0.01, max_epochs=1000, tolerance=1e-6)
    
    print(f"\nTraining model on {X_train.shape[0]} samples...")
    model.fit(X_train, y_train, num_classes)
    
    # Make predictions on training data (to evaluate training performance)
    print(f"\nEvaluating model on training data...")
    y_train_pred = model.predict(X_train)
    
    # Compute and display evaluation metrics
    train_metrics = classification_report(y_train, y_train_pred, num_classes)
    
    # Make predictions on test data
    print(f"\nMaking predictions on test data...")
    test_predictions = model.predict(X_test)
    
    print(f"\nTest predictions completed!")
    print(f"Sample test predictions: {test_predictions[:20]}")
    
    # Save test predictions
    save_test_predictions(test_predictions, "MNIST_data/test_predictions.csv")
    
    # Visualize some test samples with predictions
    print(f"\nVisualizing test predictions...")
    
    # For visualization, we'll use some samples and show predictions
    sample_indices = [0, 100, 500, 1000, 2000]
    print(f"\nShowing predictions for sample indices: {sample_indices}")
    
    for i, idx in enumerate(sample_indices):
        print(f"\n--- Test Sample {idx} ---")
        # Convert normalized data back to 0-255 range for visualization
        pixel_data = (X_test[idx] * 255).astype(int)
        predicted_label = predict_and_show(pixel_data, model)
    
    # Show prediction distribution
    unique_preds, pred_counts = np.unique(test_predictions, return_counts=True)
    print(f"\nTest prediction distribution:")
    for digit, count in zip(unique_preds, pred_counts):
        print(f"  Digit {digit}: {count} predictions ({count/len(test_predictions)*100:.1f}%)")
    
    print(f"\n" + "="*60)
    print("MNIST Classification Complete!")
    print("="*60)
    
    return model, test_predictions, train_metrics

if __name__ == "__main__":
    # Run the complete pipeline
    model, test_predictions, metrics = main()
