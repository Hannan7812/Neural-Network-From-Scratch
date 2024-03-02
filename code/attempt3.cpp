#include "classes.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>

using namespace std;
using namespace Eigen;

double extract_float(string word){
    int to_divide = 1;
    double ans = 0;
    bool decimal_passed = false;
    for (int i = 0; i < word.length(); i++) {
        if (word[i] == '.') {
            decimal_passed = true;
        } else {
            ans *= 10; // Move the decimal place one position to the left
            ans += int(word[i] - '0'); // Add the digit
            if (decimal_passed) {
                to_divide *= 10; // Increment divisor only after decimal point
            }
        }
    }
    ans /= to_divide;
    return ans;
}


int main(){
    DenseLayer* l1 = new DenseLayer(784,16);
    DenseLayer* l2 = new DenseLayer(16,10);
    DenseLayer* l3 = new DenseLayer(10,10);

    vector<DenseLayer*> v = {l1, l2, l3};

    Network* network = new Network(v);

    vector<vector<double>> dataset;
    ifstream file;
    file.open("archive/mnist_train.csv");
    string line, word;
    while (getline(file, line)) {
        vector<double> row; 
        stringstream s(line); 
  
        while (getline(s, word, ',')) {
  
            row.push_back(extract_float(word)); 
        }
        dataset.push_back(row);
    }
    file.close();

    vector<vector<double>> test_dataset;

    file.open("archive/mnist_test.csv");
    while (getline(file, line)) {
        vector<double> row; 
        stringstream s(line); 
  
        while (getline(s, word, ',')) {
  
            row.push_back(extract_float(word)); 
        }
        test_dataset.push_back(row);
    }
    file.close();

    network->train(dataset, 50, 10, 0.01);

    int correct = 0, wrong = 0, total = 0;
    for (auto x : test_dataset){
        double label = x.back();
        x.pop_back();
        vector<double> features = x;
        int predicted = network->predict(features, 10, label);
        cout << predicted << '\n';
        if (predicted == label){
            correct++;
        }
        else{
            wrong++;
        }
        total++;
    }

    cout << "Out of " << total << ", " << correct << " were predicted correct and " << wrong << " were predicted wrong";
    
}