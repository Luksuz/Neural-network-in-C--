<!-- Project Title -->
# Neural Network Implementation in C++ 🧠

<!-- Badges (Optional) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- Project Description -->
This repository contains an implementation of a simple neural network in C++. It includes classes for Linear layers, Neurons, Softmax activation, Cross-entropy loss calculation, and batching functionality. This project serves as a practice exercise to deepen understanding of neural network operations and mathematics.

## Overview

The neural network implementation consists of several components:
- **Linear Layer**: Implements a fully connected layer with weights and biases.
- **Neuron**: Represents individual units in the network, capable of forward and backward propagation.
- **Softmax Activation**: Applies the softmax function to output probabilities.
- **Cross-Entropy Loss**: Computes the loss between predicted and true labels.
- **Batching Functionality**: Divides input data into batches for training efficiency.

## Features

- **Modular Design**: Each component (layer, neuron, activation, loss) is encapsulated in its own class.
- **Forward and Backward Propagation**: Implements both forward and backward passes to train the network.
- **Training Loop**: Iterates over epochs and batches to update weights using gradient descent.
- **Usage of Standard Libraries**: Utilizes standard C++ libraries for file I/O, mathematical operations, and data manipulation.

## Installation

### Prerequisites

- C++ Compiler (e.g., g++)
- CMake (for building the project)

**Build and run the main.cpp file**
