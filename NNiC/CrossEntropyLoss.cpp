#include "CrossEntropyLoss.h"
#include <vector>
#include <cmath>

using namespace std;

double crossEntropyLoss(const vector<double> softmaxed_z, const vector<double> y_true) {
    double loss = 0;
    double epsilon = 1e-10;

    for (size_t i = 0; i < softmaxed_z.size(); ++i) {
        loss += -1 * y_true[i] * log(softmaxed_z[i] + epsilon);
    }
    
    return loss;
}

vector<double> crossEntropyLossDeriv(const vector<double> softmaxed_z, const vector<double> y_true) {
    vector<double> deriv(softmaxed_z.size());
    double epsilon = 1e-10;

    for (size_t i = 0; i < softmaxed_z.size(); ++i) {
        deriv[i] = -y_true[i] / (softmaxed_z[i] + epsilon);
    }
    
    return deriv;
}