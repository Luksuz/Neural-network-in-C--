#include <iostream>
#include "ReLU.h"
#include "Linear.h"

int main() {
    std::vector<double> input = {1.0, -2.0, 3.0, -4.0};

    // Create a Linear layer
    Linear linear(input.size());
    
    // Forward pass through Linear layer
    std::vector<double> z1 = linear.forward(input, 0);

    // Apply ReLU activation
    std::vector<double> output = relu(z1);

    // Print the input vector
    std::cout << "Input: ";
    for (auto val : input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Print the output after ReLU
    std::cout << "Output after ReLU: ";
    for (auto val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}