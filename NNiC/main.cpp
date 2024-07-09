#include <iostream>
#include "ReLU.h"
#include "Linear.h"

using namespace std;

int main() {
    vector<vector<double>> input_arr = {{1.0, -2.0, 3.0, -4.0}, {2.0, 3.0, 1.0, 0}, {3.2, 1.1, -1.0, -3.0}};

    Linear linear(input_arr[0].size(), input_arr.size());
    
    vector<double> z1 = linear.forward(input_arr);
    vector<double> a1 = relu(z1);

    cout << "Input: ";
    for (auto input : input_arr) {
        for (auto val : input) {
        cout << val << " ";
        }
    }
    cout << endl;

    cout << "Output after ReLU: ";
    for (auto val : a1) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}