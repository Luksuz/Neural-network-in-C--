#ifndef Neuron_H
#define Neuron_H

#include <vector>
using namespace std;

class Neuron
{
private:
   

public:
    vector<double> weights;
    double bias;
    vector<double> d_weights;
    double d_bias;
    Neuron(int input_size);
    double linear_transform(const vector<double>& x);
    void backward(const vector<double>& x, double dL_dz);
};

#endif