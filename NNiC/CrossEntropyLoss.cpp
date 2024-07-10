#include "CrossEntropyLoss.h"
#include <vector>

using namespace std;

double crossEntropyLoss(const vector<double> softmaxed_z, const vector<double> y_true) {
    double loss = 0;

    for (size_t i = 0; i < softmaxed_z.size(); ++i) {
        loss += -1 * y_true[i] * log(softmaxed_z[i]);
    }
    
    return loss;
}