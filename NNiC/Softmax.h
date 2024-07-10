#ifndef Softmax_H
#define Softmax_H
#include <Eigen/Dense>
#include <vector>

using namespace Eigen;
using namespace std;

vector<VectorXd> softmax(vector<VectorXd>& logits);

#endif