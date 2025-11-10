"""
Test script to demonstrate individual functions of the logistic regression implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logistic_regression import (
    LogisticRegressionScratch, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    array_to_image,
    predict_and_show,
    classification_report
)

def test_evaluation_metrics():
    """
    Test the evaluation metrics with simple examples
    """
    print("Testing Evaluation Metrics")
    print("="*40)
    
    # Simple test case
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 1, 1, 0, 2, 2, 1])
    num_classes = 3
    
    print(f"True labels:      {y_true}")
    print(f"Predicted labels: {y_pred}")
    print()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, num_classes)
    recall = recall_score(y_true, y_pred, num_classes)
    f1 = f1_score(y_true, y_pred, num_classes)
    
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    
    # Generate detailed classification report
    classification_report(y_true, y_pred, num_classes)

def test_array_to_image():
    """
    Test the array_to_image function with a sample from the dataset
    """
    print("\nTesting Array to Image Conversion")
    print("="*40)
    
    # Load a small sample from the training data
    train_df = pd.read_csv("MNIST_data/train.csv")
    
    # Get the first sample
    first_sample = train_df.iloc[0]
    true_label = first_sample.iloc[0]  # First column is label
    pixel_array = first_sample.iloc[1:].values  # Remaining columns are pixels
    
    print(f"Sample true label: {true_label}")
    print(f"Pixel array shape: {pixel_array.shape}")
    print(f"Pixel value range: {pixel_array.min()} - {pixel_array.max()}")
    
    # Convert to image and display
    img = array_to_image(pixel_array, title=f"True Label: {true_label}")
    print(f"Image shape after conversion: {img.shape}")

def test_predict_and_show():
    """
    Test the predict_and_show function
    """
    print("\nTesting Prediction and Visualization")
    print("="*40)
    
    # Load training data
    train_df = pd.read_csv("MNIST_data/train.csv")
    
    # Take a small subset for quick training
    subset_size = 1000
    train_subset = train_df.iloc[:subset_size]
    
    y_train = train_subset.iloc[:, 0].values
    X_train = train_subset.iloc[:, 1:].values / 255.0
    
    print(f"Training on {subset_size} samples for demonstration...")
    
    # Train a simple model
    model = LogisticRegressionScratch(learning_rate=0.1, max_epochs=100)
    model.fit(X_train, y_train, 10)
    
    # Test prediction on a few samples
    for i in range(3):
        sample_data = train_df.iloc[i]
        true_label = sample_data.iloc[0]
        pixel_array = sample_data.iloc[1:].values
        
        print(f"\n--- Sample {i+1} ---")
        predicted_label = predict_and_show(pixel_array, model, true_label)

def test_sigmoid_function():
    """
    Test the sigmoid function
    """
    print("\nTesting Sigmoid Function")
    print("="*40)
    
    from logistic_regression import sigmoid
    
    # Test various inputs
    test_values = [-10, -1, 0, 1, 10, 100, -100]
    
    print("Input -> Sigmoid Output")
    for val in test_values:
        result = sigmoid(val)
        print(f"{val:6} -> {result:.6f}")

if __name__ == "__main__":
    print("Individual Function Testing for Logistic Regression")
    print("="*60)
    
    # Test 1: Evaluation metrics
    test_evaluation_metrics()
    
    # Test 2: Sigmoid function
    test_sigmoid_function()
    
    # Test 3: Array to image conversion
    test_array_to_image()
    
    # Test 4: Prediction and visualization (this will take a bit longer)
    test_predict_and_show()
    
    print("\n" + "="*60)
    print("All tests completed!")