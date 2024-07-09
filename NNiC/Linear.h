#ifndef Linear_H
#define Linear_H

#include <vector>
using namespace std;

class Linear
{
private:
    vector<double> weights;

public:
    Linear(int input_size);
    vector<double> forward(const vector<double>& x, double b);
};

#endif