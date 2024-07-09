#include "Linear.h"
#include <cstdlib>   // For rand() function
#include <ctime>     // For srand() function

using namespace std;

Linear::Linear(int input_size, int batch_size) {
    srand(time(0));
    weights.resize(batch_size);  
    for (int i = 0; i < batch_size; ++i) {
        weights[i].resize(input_size);
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }

    bias.resize(input_size);  
    for (int i = 0; i < batch_size; ++i) {
        bias[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}


vector<double> Linear::forward(const vector<vector<double>>& x) {
    vector<double> result;
    for (int i = 0; i < x.size(); ++i){
        double sum = 0.0;
        for (int j = 0; j < x[i].size(); ++j){
            sum += weights[i][j] * x[i][j];
        }
        result.push_back(sum + bias[i]);
    }
    return result;
}