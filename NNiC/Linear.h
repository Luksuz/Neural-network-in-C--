#ifndef Linear_H
#define Linear_H

#include <vector>
using namespace std;

class Linear
{
private:
    vector<vector<double>> weights;
    vector<double> bias;

public:
    Linear(int input_size, int batch_size);
    vector<double> forward(const vector<vector<double>>& x);
};

#endif