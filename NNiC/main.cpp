#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath> 
#include <algorithm> 
#include <numeric> 
#include "ReLU.h"
#include "Linear.h"
#include "Softmax.h"
#include "CrossEntropyLoss.h"

using namespace std;

vector<pair<vector<vector<double>>, vector<vector<double>>>> create_batch(const vector<vector<double>> &x, const vector<vector<double>> &y, int batch_size) {
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

double train(vector<vector<double>> X, vector<vector<double>> y, vector<int> layers, int batch_size, double lr, int epochs, vector<Linear>& layers_arr) {
    auto batched_data = create_batch(X, y, batch_size);

    for (int i = 0; i < layers.size() - 1; ++i) {
        int in_features = layers[i];
        int out_features = layers[i + 1];
        Linear linearLayer(in_features, out_features);
        layers_arr.push_back(linearLayer);
    }

    double loss;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        for (const auto &batch : batched_data) {
            vector<vector<double>> prev_input = batch.first;

            vector<vector<vector<double>>> activations;
            activations.push_back(prev_input);

            for (auto &linearLayer : layers_arr) {
                vector<vector<double>> z = linearLayer.forward(prev_input);
                vector<vector<double>> a1;
                for (const auto &z_val : z) {
                    vector<double> a = relu(z_val);
                    a1.push_back(a);
                }
                activations.push_back(a1);
                prev_input = a1;
            }

            vector<vector<double>> softmaxed;
            for (const auto &a : prev_input) {
                softmaxed.push_back(softmax(a));
            }

            double batch_loss = 0.0;
            for (size_t i = 0; i < softmaxed.size(); ++i) {
                batch_loss += crossEntropyLoss(softmaxed[i], batch.second[i]);
            }
            epoch_loss += batch_loss;

            vector<vector<double>> dL_dz;
            for (size_t i = 0; i < softmaxed.size(); ++i) {
                dL_dz.push_back(crossEntropyLossDeriv(softmaxed[i], batch.second[i]));
            }

            for (int i = layers_arr.size() - 1; i >= 0; --i) {
                dL_dz = layers_arr[i].backward(activations[i], dL_dz);
                layers_arr[i].update_weights(lr);
            }
        }
        cout << "Epoch " << epoch + 1 << " loss: " << epoch_loss / batched_data.size() << endl;
        loss = epoch_loss;
    }

    return loss / X.size();
}

vector<vector<double>> readCSV(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line;

    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string value;
        while (getline(ss, value, ',')) {
            row.push_back(stod(value));
        }
        data.push_back(row);
    }

    return data;
}

pair<double, double> test(const vector<vector<double>>& X_test, const vector<vector<double>>& y_true, vector<Linear>& layers_arr) {
    double total_loss = 0.0;
    int correct_predictions = 0;

    for (size_t i = 0; i < X_test.size(); ++i) {
        vector<vector<double>> prev_input = {X_test[i]};

        // Forward pass
        for (auto &linearLayer : layers_arr) {
            vector<vector<double>> z = linearLayer.forward(prev_input);
            vector<vector<double>> a1;
            for (const auto &z_val : z) {
                vector<double> a = relu(z_val);
                a1.push_back(a);
            }
            prev_input = a1;
        }

        vector<double> softmaxed = softmax(prev_input[0]);
        total_loss += crossEntropyLoss(softmaxed, y_true[i]);

        int predicted_class = distance(softmaxed.begin(), max_element(softmaxed.begin(), softmaxed.end()));
        int true_class = distance(y_true[i].begin(), max_element(y_true[i].begin(), y_true[i].end()));

        cout << "Predicted class: " << predicted_class << ", True class: " << true_class << endl;

        if (predicted_class == true_class) {
            ++correct_predictions;
        }
    }

    double average_loss = total_loss / X_test.size();
    double accuracy = static_cast<double>(correct_predictions) / X_test.size() * 100;

    return {average_loss, accuracy};
}

int main() {
    vector<vector<double>> input_arr = readCSV("./data/X_train.csv");
    vector<vector<double>> y_true = readCSV("./data/y_train.csv");

    vector<vector<double>> X_test = readCSV("./data/X_test.csv");
    vector<vector<double>> y_test = readCSV("./data/y_test.csv");

    vector<int> layers = {4, 30, 3};

    int batch_size = 32;
    double learning_rate = 0.00001;
    int epochs = 1012;

    vector<Linear> layers_arr;

    double avg_loss = train(input_arr, y_true, layers, batch_size, learning_rate, epochs, layers_arr);

    cout << "Average Loss: " << avg_loss << endl;

    auto [test_loss, test_accuracy] = test(X_test, y_test, layers_arr);

    cout << "Test Loss: " << test_loss << endl;
    cout << "Test Accuracy: " << test_accuracy << "%" << endl;

    return 0;
}