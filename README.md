# MNIST-Optimizer-Comparison
Title: Comparative Analysis of Optimization Methods for Image Classification on MNIST Dataset

Description:
This project focuses on comparing three popular stochastic optimization algorithms—Adam, Adagrad, and RMSprop—on the MNIST image classification task. The goal is to analyze the performance of these optimizers in terms of convergence speed, loss minimization, and accuracy.

The workflow includes:

Dataset Preparation: The MNIST dataset is normalized using z-score and divided into training, validation, and test sets.
Model Architecture: A Multi-Layer Perceptron (MLP) with 2-3 hidden layers is designed for classification.
Hyperparameter Search: Grid search or random search is employed to determine the optimal learning rate and other hyperparameters for each optimizer.
Training and Validation: The model is trained using each optimizer, with early stopping based on validation performance.
Testing and Analysis: The best model for each optimizer is evaluated on the test set, and the results are visualized using plots for training loss, validation loss, accuracy, and test loss.
