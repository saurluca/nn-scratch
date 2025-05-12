# Neural Network Implementation from Scratch

This project implements a neural network from scratch using only NumPy, demonstrating the fundamental concepts of deep learning without relying on high-level frameworks like PyTorch or TensorFlow.

## Project Overview

The current implementation focuses on a binary classification task: determining whether a number is positive or negative (greater or less than zero). This serves as a simple demonstration of the neural network's capabilities.

## Implementation Details

### Core Components

1. **Neural Network Architecture**
   - Input layer: 1 neuron (for single number input)
   - Hidden layer: 2 neurons with ReLU activation
   - Output layer: 1 neuron with Sigmoid activation for binary classification

2. **Key Classes**
   - `FeedForwardNeuralNetwork`: Main neural network implementation
   - `LinearLayer`: Implements linear transformations (weights and biases)
   - `ReLU`: Rectified Linear Unit activation function
   - `Sigmoid`: Sigmoid activation function for output layer
   - `BCELoss`: Binary Cross Entropy loss function
   - `SGD`: Stochastic Gradient Descent optimizer

### Training Process

The network is trained using:
- Binary Cross-Entropy Loss
- Stochastic Gradient Descent (SGD) optimization
- Backpropagation for gradient computation

## Dependencies

- NumPy: For numerical computations
- Matplotlib: For plotting training results

## Usage

To run the training:

```bash
python main.py
```

This will:
1. Generate training data (100 samples) and test data (20 samples)
2. Train the model for 10 epochs
3. Evaluate the model on test data
4. Generate plots for loss and accuracy

## Results

The training process generates two plots:
- `results/loss_plot.png`: Shows the training loss over epochs
- `results/accuracy_plot.png`: Shows the accuracy over epochs

## Implementation Notes

- The network uses He initialization for weights
- Forward and backward passes are implemented manually to demonstrate the underlying mathematics
- The implementation includes a modular design with separate classes for different components

## Project Structure

```
.
├── main.py           # Main training script
├── modules.py        # Neural network components
└── results/          # Directory for output plots
    ├── loss_plot.png
    └── accuracy_plot.png
```
