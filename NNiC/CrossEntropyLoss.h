#ifndef CrossEntropyLoss_H
#define CrossEntropyLoss_H

#include <vector>

using namespace std;

class CrossEntropyLoss
{
    public:
        vector<double> compute_loss(vector<vector<double>>);
};

#endif