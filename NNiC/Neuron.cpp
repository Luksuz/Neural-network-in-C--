#include "Neuron.h"
#include <cstdlib>  
#include <ctime>     
#include <numeric> 

using namespace std;

Neuron::Neuron(int input_size) {
    //srand(time(0));
    weights.resize(input_size);
    for (int i = 0; i < input_size; ++i) {
        weights[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    bias = static_cast<double>(rand()) / RAND_MAX;
}

double Neuron::linear_transform(const vector<double>& x) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); ++i) {
        sum += weights[i] * x[i];
    }
    double linear_prod = sum + bias;
    return linear_prod;
}