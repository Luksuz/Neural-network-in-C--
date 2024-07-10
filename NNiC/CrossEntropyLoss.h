#ifndef CrossEntropyLoss_H
#define CrossEntropyLoss_H

#include <vector>

using namespace std;

double crossEntropyLoss(vector<double> softmaxed_z, const vector<double> y_true);

#endif