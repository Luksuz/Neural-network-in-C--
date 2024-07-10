#include "Softmax.h"
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

vector<VectorXd> softmax(const vector<VectorXd>& x) {
    vector<VectorXd> result;
    for (const auto& vec : x) {
        VectorXd shifted_x = vec.array() - vec.maxCoeff();
        VectorXd exp_x = shifted_x.array().exp();
        double sum_exp_x = exp_x.sum();
        VectorXd softmaxed = exp_x / sum_exp_x;
        result.push_back(softmaxed);
    }
    return result;
}