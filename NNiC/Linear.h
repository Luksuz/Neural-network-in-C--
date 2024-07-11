#ifndef Linear_H
#define Linear_H

#include "Neuron.h"
#include <vector>

using namespace std;

class Linear
{
private:
    int in_features;
    int n_neurons;
    vector<Neuron> neurons;
    vector<double> a;

public:
    Linear(int in_features, int n_neurons);
    vector<vector<double>> forward(vector<vector<double>> x);
    vector<vector<double>> backward(const vector<vector<double>> x, const vector<vector<double>> dL_dz);
    void update_weights(double learning_rate);
};

#endif