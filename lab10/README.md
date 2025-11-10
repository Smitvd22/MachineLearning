# MNIST Digit Classification - Logistic Regression Implementation from Scratch

## Overview
This implementation provides a complete logistic regression classifier for the MNIST digit recognition dataset, implemented entirely from scratch without using any machine learning libraries (like scikit-learn).

## Features Implemented

### 1. Logistic Regression Algorithm (One-vs-Rest)
- **Sigmoid Function**: Implemented from scratch with overflow protection
- **Gradient Descent**: Custom implementation with configurable learning rate and epochs
- **Multi-class Classification**: Using One-vs-Rest approach (10 binary classifiers)
- **Cost Function**: Log-likelihood cost function with proper mathematical formulation
- **Convergence Checking**: Early stopping based on cost function tolerance

### 2. Evaluation Metrics (All from Scratch)
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **Precision**: TP / (TP + FP) - macro and micro averaging
- **Recall**: TP / (TP + FN) - macro and micro averaging  
- **F1-Score**: 2 * (Precision × Recall) / (Precision + Recall)
- **Confusion Matrix**: Complete confusion matrix generation
- **Classification Report**: Comprehensive per-class and overall metrics

### 3. Image Processing Functions
- **Array to Image Conversion**: Converts 784-element array to 28×28 image
- **Visualization**: Display images with matplotlib
- **Prediction with Visualization**: Shows image alongside prediction and confidence

### 4. Data Processing
- **Data Loading**: Handles MNIST CSV format
- **Normalization**: Pixel value normalization (0-255 to 0-1)
- **Class Distribution Analysis**: Shows training data statistics

## File Structure

```
lab10/
├── logistic_regression.py          # Main implementation
├── test_individual_functions.py    # Testing script for individual components  
├── MNIST_data/
│   ├── train.csv                   # Training data (42,000 samples)
│   ├── test.csv                    # Test data (28,000 samples)
│   └── test_predictions.csv        # Generated predictions
└── README.md                       # This documentation
```

## Implementation Details

### Logistic Regression Class
```python
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, max_epochs=1000, tolerance=1e-6)
    def fit(self, X, y, num_classes)        # Train the model
    def predict(self, X)                    # Predict multiple samples
    def predict_single(self, x)             # Predict single sample
    def predict_proba(self, X)              # Get prediction probabilities
```

### Key Functions
```python
# Core mathematical functions
sigmoid(z)                                  # Sigmoid activation function

# Evaluation metrics
accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred, num_classes, average='macro')
recall_score(y_true, y_pred, num_classes, average='macro')
f1_score(y_true, y_pred, num_classes, average='macro')
classification_report(y_true, y_pred, num_classes)

# Image processing
array_to_image(pixel_array, title=None, show_plot=True)
predict_and_show(pixel_array, model, true_label=None)
visualize_predictions(X_test, y_test, model, num_samples=10)
```

## Results

### Training Performance
- **Training Samples**: 42,000
- **Test Samples**: 28,000
- **Features**: 784 (28×28 pixels)
- **Classes**: 10 (digits 0-9)

### Evaluation Metrics on Training Data
- **Overall Accuracy**: 85.03%
- **Macro-averaged Precision**: 85.22%
- **Macro-averaged Recall**: 84.70%
- **Macro-averaged F1-Score**: 84.67%

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.9001    | 0.9487 | 0.9238   | 4132    |
| 1     | 0.8603    | 0.9520 | 0.9038   | 4684    |
| 2     | 0.8826    | 0.8121 | 0.8459   | 4177    |
| 3     | 0.7875    | 0.8460 | 0.8157   | 4351    |
| 4     | 0.8613    | 0.8539 | 0.8576   | 4072    |
| 5     | 0.8850    | 0.6285 | 0.7350   | 3795    |
| 6     | 0.8702    | 0.9272 | 0.8978   | 4137    |
| 7     | 0.8943    | 0.8764 | 0.8852   | 4401    |
| 8     | 0.7620    | 0.8142 | 0.7872   | 4063    |
| 9     | 0.8188    | 0.8116 | 0.8152   | 4188    |

## Usage Examples

### Basic Usage
```python
# Load and preprocess data
X_train, y_train, X_test = load_and_preprocess_data()

# Create and train model
model = LogisticRegressionScratch(learning_rate=0.01, max_epochs=1000)
model.fit(X_train, y_train, num_classes=10)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_true, predictions)
```

### Prediction with Visualization
```python
# Predict and show image for a single sample
pixel_array = X_test[0] * 255  # Convert back to 0-255 range
predicted_label = predict_and_show(pixel_array, model)
```

### Comprehensive Evaluation
```python
# Generate detailed classification report
metrics = classification_report(y_true, y_pred, num_classes)
```

## Mathematical Foundation

### Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
```

### Cost Function (Log-Likelihood)
```
J = -(1/m) * Σ[y*log(h) + (1-y)*log(1-h)]
```

### Gradient Calculation
```
∂J/∂w = (1/m) * X^T * (h - y)
∂J/∂b = (1/m) * Σ(h - y)
```

### Parameter Update
```
w = w - α * ∂J/∂w
b = b - α * ∂J/∂b
```

## Key Features

1. **Complete From-Scratch Implementation**: No use of sklearn or other ML libraries
2. **Robust Error Handling**: Overflow protection in sigmoid function
3. **Comprehensive Metrics**: All major classification metrics implemented
4. **Visualization Support**: Image display and prediction visualization
5. **Multi-class Support**: One-vs-Rest approach for 10-class classification
6. **Progress Monitoring**: Training progress display and convergence checking
7. **Flexible Parameters**: Configurable learning rate, epochs, and tolerance

## Files Generated
- `test_predictions.csv`: Contains predictions for all test samples
- Visualization plots for sample predictions
- Detailed classification reports

This implementation demonstrates a complete understanding of logistic regression mathematics and provides a solid foundation for multi-class image classification tasks.