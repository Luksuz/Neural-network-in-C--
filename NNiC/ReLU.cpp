#include "ReLU.h"

using namespace std;

vector<double> relu(const vector<double>& x){
    vector<double> result;
    for (const auto& val : x){
        result.push_back(val > 0 ? val : 0);
    }
    return result;
}