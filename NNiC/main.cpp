#include <iostream>
#include "ReLU.h"
#include "Linear.h"

using namespace std;

vector<vector<vector<double>>> create_batch(vector<vector<double>>& x, int batch_size) {
    vector<vector<vector<double>>> batched_x;
    for (size_t i = 0; i < x.size(); i += batch_size) {
        vector<vector<double>> batch;
        for (size_t j = i; j < i + batch_size && j < x.size(); ++j) {
            batch.push_back(x[j]);
        }
        batched_x.push_back(batch);
    }
    return batched_x;
}

int main() {
    vector<vector<double>> input_arr = {{0.3, 0.0, 0.22, 0.1}, {1.0, 0.4, 0.4, 0.98}, {0.91, 0.02, 0.1, 0.44}};
    vector<vector<vector<double>>> batched_x = create_batch(input_arr, 2);

    cout << "Size of the input array: " << input_arr.size() << endl;
    cout << "Number of batches: " << batched_x.size() << endl;

    Linear linearLayer(4, 5);

    for (const auto& batch : batched_x) {
        vector<vector<double>> z1 = linearLayer.forward(batch);
        for (const auto& z : z1) {
            vector<double> a1 = relu(z);
            cout << "Output after ReLU: ";
            for (auto val : a1) {
                cout << val << " ";
            }
            cout << endl;
        }
    }

    return 0;
}