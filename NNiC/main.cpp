#include <iostream>
#include "ReLU.h"
#include "Linear.h"
#include "Softmax.h"
#include "CrossEntropyLoss.h"


using namespace std;

vector<pair<vector<vector<double>>, vector<vector<double>>>> create_batch(const vector<vector<double>>& x, const vector<vector<double>>& y, int batch_size) {
    vector<pair<vector<vector<double>>, vector<vector<double>>>> batched_data;
    for (size_t i = 0; i < x.size(); i += batch_size) {
        vector<vector<double>> batch_x;
        vector<vector<double>> batch_y;
        for (size_t j = i; j < i + batch_size && j < x.size(); ++j) {
            batch_x.push_back(x[j]);
            batch_y.push_back(y[j]);
        }
        batched_data.push_back({batch_x, batch_y});
    }
    return batched_data;
}

int main() {
    vector<vector<double>> input_arr = {{0.3, 0.0, 0.22, 0.1}, {1.0, 0.4, 0.4, 0.98}, {0.91, 0.02, 0.1, 0.44}};
    vector<vector<double>> y_true = {{0, 0, 1, 0}, {0, 0, 1, 0}, {0, 0, 1, 0}};

    int batch_size = 2;

    auto batched_data = create_batch(input_arr, y_true, batch_size);

    cout << "Size of the input array: " << input_arr.size() << endl;
    cout << "Number of batches: " << batched_data.size() << endl;

    Linear linearLayer(4, 4);

    for (const auto& batch_data : batched_data) {
        vector<vector<double>> z1 = linearLayer.forward(batch_data.first);

        vector<vector<double>> a1;
        for (const auto& z : z1) {
            vector<double> a = relu(z);
            a1.push_back(a);
        }

        vector<vector<double>> softmaxed;
        for (const auto& a : a1) {
            vector<double> out = softmax(a);
            softmaxed.push_back(out);
        }

        vector<double> total_loss;
        for (int i = 0; i < softmaxed.size(); ++i) {
            double loss = crossEntropyLoss(softmaxed[i], batch_data.second[i]);
            total_loss.push_back(loss);
        }

        for (double loss : total_loss) {
            cout << "Batch loss: " << loss << endl;
        }
        cout << "Softmax result: ";
            for (vector<double> vec : softmaxed) {
                for(int i = 0; i < vec.size(); ++i){
                    if(i == 3){
                        vec[i] = 0.925;
                    }else{
                    vec[i] = 0.025;
                    }
                }
            }
    }
   
    cout << endl;

    for (const auto& batch_data : batched_data) {
        cout << "Batch X: ";
        for (const auto& x : batch_data.first) {
            for (const auto& val : x) {
                cout << val << " ";
            }
            cout << endl;
        }
        cout << "Batch Y: ";
        for (const auto& y : batch_data.second) {
            for (const auto& val : y) {
                cout << val << " ";
            }
            cout << endl;
        }
    }

    return 0;
}