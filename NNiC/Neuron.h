#ifndef Neuron_H
#define Neuron_H

#include <vector>
using namespace std;

class Neuron
{
private:
    vector<double> weights;
    double bias;

public:
    Neuron(int input_size);
    double linear_transform(const vector<double>& x);
};

#endif