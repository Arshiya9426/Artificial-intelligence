#Implementation of Perceptron and Backpropagation Neural Network 

import numpy as np

# =====================================================================
# PART A: Single-Layer Perceptron
# =====================================================================
class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=10):
        """
        Initializes the Perceptron classifier.
        
        Arguments:
        input_size -- Number of input features (excluding bias).
        lr         -- Learning rate for weight updates.
        epochs     -- Number of passes over the training dataset.
        """
        self.weights = np.zeros(input_size + 1)  # Includes index 0 for bias
        self.lr = lr
        self.epochs = epochs

    def activate(self, x):
        """Step activation function."""
        return 1 if x >= 0 else 0

    def train(self, X, y):
        """Trains the perceptron using the delta perceptron learning rule."""
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                inputs = np.insert(xi, 0, 1)  # Insert bias unit at index 0
                prediction = self.activate(np.dot(inputs, self.weights))
                
                # Weight update rule: W = W + lr * (target - prediction) * X
                self.weights += self.lr * (target - prediction) * inputs

    def predict(self, X):
        """Predicts the class label for a single input instance."""
        inputs = np.insert(X, 0, 1)  # Insert bias unit at index 0
        return self.activate(np.dot(inputs, self.weights))


# =====================================================================
# PART B: Artificial Neural Network (Backpropagation)
# =====================================================================
def sigmoid(x):
    """Calculates the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Calculates the derivative of the sigmoid function given its output."""
    return x * (1 - x)

def train_backpropagation(X, y, epochs=5000, lr=0.1):
    """
    Trains a 3-layer Neural Network (Input, Hidden, Output) using Backpropagation.
    
    Arguments:
    X      -- Matrix of training inputs.
    y      -- Matrix of expected outputs.
    epochs -- Number of training iterations.
    lr     -- Learning rate factor.
    """
    input_layer_size = X.shape[1]
    hidden_layer_size = 3
    output_layer_size = 1
    
    # Initialize weight matrices with random uniform values
    wh = np.random.uniform(size=(input_layer_size, hidden_layer_size))
    wout = np.random.uniform(size=(hidden_layer_size, output_layer_size))
    
    for _ in range(epochs):
        # --- Forward Pass ---
        hidden_layer_input = np.dot(X, wh)
        hidden_layer_activations = sigmoid(hidden_layer_input)
        
        output_layer_input = np.dot(hidden_layer_activations, wout)
        predicted_output = sigmoid(output_layer_input)
        
        # --- Backpropagation (Error Calculation & Gradient Descent) ---
        # 1. Compute output layer error and delta gradient
        error = y - predicted_output
        d_predicted_output = error * sigmoid_derivative(predicted_output)
        
        # 2. Compute hidden layer error and delta gradient
        hidden_layer_error = d_predicted_output.dot(wout.T)
        d_hidden_layer = hidden_layer_error * sigmoid_derivative(hidden_layer_activations)
        
        # 3. Update weights using computed gradients
        wout += hidden_layer_activations.T.dot(d_predicted_output) * lr
        wh += X.T.dot(d_hidden_layer) * lr
        
    return predicted_output


# =====================================================================
# --- Example Execution ---
# =====================================================================
if __name__ == "__main__":
    # --- Test Case: Perceptron (Linearly Separable Logic Gates) ---
    print("--- Training Perceptron (AND Gate) ---")
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    
    p = Perceptron(input_size=2)
    p.train(X_and, y_and)
    print("Prediction for [1, 1]:", p.predict(np.array([1, 1])))
    print("Prediction for [1, 0]:", p.predict(np.array([1, 0])))

    # --- Test Case: Backpropagation (Non-Linearly Separable Problems) ---
    print("\n--- Training Backpropagation (XOR Gate) ---")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])
    
    outputs = train_backpropagation(X_xor, y_xor, epochs=10000, lr=0.2)
    print("Final raw outputs after training:")
    print(np.round(outputs, 4))
