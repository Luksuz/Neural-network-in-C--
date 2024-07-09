#include "Linear.h"
#include <cstdlib>   // For rand() function
#include <ctime>     // For srand() function

// Constructor to initialize weights with random values
Linear::Linear(int input_size) {
    srand(time(0));  // Seed for random number generation
    for (int i = 0; i < input_size; ++i) {
        weights.push_back(static_cast<double>(rand()) / RAND_MAX); // Random value between 0 and 1
    }
}

std::vector<double> Linear::forward(const std::vector<double>& x, double b) {
    std::vector<double> result;
    for (size_t i = 0; i < x.size(); ++i) {
        result.push_back(x[i] * weights[i] + b);
    }
    return result;
}