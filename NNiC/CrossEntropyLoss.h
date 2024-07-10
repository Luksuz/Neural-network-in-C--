#ifndef CrossEntropyLoss_H
#define CrossEntropyLoss_H

#include <vector>

using namespace std;

double crossEntropyLoss(vector<double> softmaxed_z, const vector<double> y_true);
vector<double> crossEntropyLossDeriv(const vector<double> softmaxed_z, const vector<double> y_true);

#endif