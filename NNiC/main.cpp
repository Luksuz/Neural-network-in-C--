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

    // Hyperparameters
    int batch_size = 2;
    double lr = 0.02;
    int epochs = 10;

    auto batched_data = create_batch(input_arr, y_true, batch_size);

    Linear linearLayer(400, 4);

    for(int i = 0; i < epochs; ++i){
        cout << "Epoch " << i << ": " << endl;

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

        for(vector<double> s: softmaxed){
            for(double val: s){
                cout << val << endl;
            }
        }

        vector<double> total_loss;
        for (int i = 0; i < softmaxed.size(); ++i) {
            double loss = crossEntropyLoss(softmaxed[i], batch_data.second[i]);
            total_loss.push_back(loss);
        }

        vector<vector<double>> dL_dz;
        for (int i = 0; i < softmaxed.size(); ++i) {
            vector<double> deriv = crossEntropyLossDeriv(softmaxed[i], batch_data.second[i]);
            dL_dz.push_back(deriv);
        }

        vector<vector<double>> dL_dx = linearLayer.backward(batch_data.first, dL_dz);
        linearLayer.update_weights(lr);

        for (double loss : total_loss) {
            cout << "Batch loss: " << loss << endl;
        }
    }

    }

    return 0;
}

double train(vector<vector<double>> X,  vector<vector<double>> y, vector<int> layers, int batch_size, double lr){
    auto batched_data = create_batch(X, y, batch_size);

    vector<Linear> layers_arr;

    for(int i = 0; i < layers.size()-1; ++i){
        int in_features = layers[i];
        int out_features = layers[i+1];
        Linear linearLayer(in_features, out_features);
    }

    vector<vector<double>> prev_input;
    for(auto batch: batched_data){
        for(Linear linearLayer: layers_arr){
            if(!prev_input.empty()){
                vector<vector<double>> z = linearLayer.forward(prev_input);
            }else{
                vector<vector<double>> z = linearLayer.forward(batch.first);
            }
            vector<vector<double>> z = linearLayer.forward(batch.first);
            vector<vector<double>> a1;
            for (const auto z : z) {
                vector<double> a = relu(z);
                a1.push_back(a);
            }
            prev_input = a1;
        }
        
        vector<vector<double>> softmaxed;
        for (const auto& a : prev_input) {
            vector<double> out = softmax(a);
            softmaxed.push_back(out);
        }
    }
}