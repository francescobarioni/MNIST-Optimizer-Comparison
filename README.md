# MNIST Optimizer Comparison

## Overview
This project explores and compares the performance of three widely-used stochastic optimization algorithms—**Adam**, **Adagrad**, and **RMSprop**—on the **MNIST image classification task**. The goal is to evaluate their efficiency and effectiveness in terms of convergence speed, loss reduction, and accuracy. 

## Objectives
- **Implement and Compare**: Analyze the behavior of Adam, Adagrad, and RMSprop optimization methods in training a neural network.
- **Hyperparameter Tuning**: Identify the optimal hyperparameters for each optimizer using grid search or random search.
- **Performance Evaluation**: Compare metrics such as training loss, validation loss, accuracy, and test loss.

## Methodology
1. **Dataset**:
   - The MNIST dataset is preprocessed with z-score normalization.
   - It is divided into training, validation, and test sets.
2. **Model Architecture**:
   - A **Multi-Layer Perceptron (MLP)** with 2-3 hidden layers is used for image classification.
3. **Optimization Algorithms**:
   - Implemented Adam, Adagrad, and RMSprop optimizers.
   - Conducted hyperparameter tuning for each optimizer.
4. **Training**:
   - Training is performed with early stopping based on validation performance.
5. **Evaluation**:
   - The trained models are tested, and results are visualized through metrics and plots.

## Features
- **Visualizations**: Training and validation loss, accuracy trends, and test performance for each optimizer.
- **Reproducibility**: All hyperparameter configurations and results are saved for reproducibility.
- **Comprehensive Analysis**: Discusses the strengths and weaknesses of each optimizer.

## Results
- Detailed comparison of convergence rates, generalization performance, and efficiency.
- Graphs and tables summarizing findings for each optimizer.

## Technologies
- **Programming Language**: Python
- **Libraries**: PyTorch, NumPy, Matplotlib, and more.

## Future Work
- Extend the comparison to convolutional neural networks (CNNs).
- Explore additional optimizers like SGD with momentum or Nadam.
- Apply the analysis to larger datasets like CIFAR-10 or ImageNet.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/MNIST-Optimizer-Comparison.git
   cd MNIST-Optimizer-Comparison
