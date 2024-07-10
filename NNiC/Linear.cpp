#include "Linear.h"

using namespace std;

Linear::Linear(int in_features, int n_neurons) {
    this->in_features = in_features;
    this->n_neurons = n_neurons;

    for (int i = 0; i < n_neurons; ++i) {
        Neuron neuron(in_features);
        neurons.push_back(neuron);
    }
}

vector<vector<double>> Linear::forward(const vector<vector<double>> x) {
    vector<vector<double>> result;
    for (const auto& input : x) {
        vector<double> curr_linear_prod;
        for (auto& neuron : neurons) {
            double linear_prod = neuron.linear_transform(input);
            curr_linear_prod.push_back(linear_prod);
        }
        result.push_back(curr_linear_prod);
    }
    return result;
}

vector<vector<double>> Linear::backward(const vector<vector<double>> x, const vector<vector<double>> dL_dz) {
    vector<vector<double>> dL_dx(x.size(), vector<double>(in_features, 0.0));

    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < neurons.size(); ++j) {
            neurons[j].backward(x[i], dL_dz[i][j]);
            for (size_t k = 0; k < in_features; ++k) {
                dL_dx[i][k] += neurons[j].weights[k] * dL_dz[i][j];
            }
        }
    }

    return dL_dx;
}