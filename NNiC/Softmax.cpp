#include <vector>
#include <cmath> // for exp
#include <numeric> // for accumulate
#include <iostream> // for testing

using namespace std;

vector<double> softmax(const vector<double> x) {
    vector<double> shifted_x(x.size());
    double max_val = *max_element(x.begin(), x.end());

    transform(x.begin(), x.end(), shifted_x.begin(), [&](double xi) { return xi - max_val; });

    vector<double> exp_x(shifted_x.size());
    transform(shifted_x.begin(), shifted_x.end(), exp_x.begin(), [](double val) { return exp(val); });

    double sum_exp_x = accumulate(exp_x.begin(), exp_x.end(), 0.0);

    vector<double> softmaxed(exp_x.size());
    transform(exp_x.begin(), exp_x.end(), softmaxed.begin(), [&](double val) { return val / sum_exp_x; });

    return softmaxed;
}
