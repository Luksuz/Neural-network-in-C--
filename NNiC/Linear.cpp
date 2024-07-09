#include "Linear.h"

using namespace std;

Linear::Linear(int in_features, int n_neurons){
    for (int i = 0; i < n_neurons; ++i) {
        Neuron neuron(in_features);
        neurons.push_back(neuron);
    }
}

vector<vector<double>> Linear::forward(const vector<vector<double>> x) {
    vector<vector<double>> result;
    for (const auto& input : x) {
        vector<double> curr_linear_prod;
        for (Neuron& neuron : neurons) {
            double linear_prod = neuron.linear_transform(input);
            curr_linear_prod.push_back(linear_prod);
        }
        result.push_back(curr_linear_prod);
    }
    return result;
}