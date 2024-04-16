#include "classes.h"
#include <iostream>

using namespace Eigen;

Layer::Layer(int input_size, int output_size){
    inputs = MatrixXd(input_size, 1); 
        outputs = MatrixXd(output_size, 1);
    for (int i = 0; i < input_size; i++){
        inputs(i,0) = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    }
    for (int i = 0; i < output_size; i++){
        outputs(i,0) = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    }
}


DenseLayer::DenseLayer(int input_size, int output_size) : Layer::Layer(input_size, output_size){
    weights = MatrixXd(output_size, input_size);
    biases =  MatrixXd(output_size, 1);
    for (int i = 0; i < output_size; i++){
        biases(i, 0) = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        for (int j = 0; j < input_size; j++){
            weights(i, j) = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        }
    }
}

MatrixXd DenseLayer::forward_propagate(MatrixXd input_to_layer){
    inputs = input_to_layer;
    outputs = weights * inputs + biases;
    outputs = outputs.unaryExpr(&tanh);
    return outputs;
}

MatrixXd DenseLayer::apply_activation(MatrixXd* apply_on){
    return apply_on->unaryExpr(&tanh);
}

MatrixXd DenseLayer::backward_propagate(MatrixXd output_gradient, double learning_rate){
    MatrixXd weights_gradient = output_gradient * inputs.transpose();
    weights -= learning_rate * weights_gradient;
    biases -= learning_rate * output_gradient;
    MatrixXd to_return = (weights.transpose()) * output_gradient;
    return to_return;
}

double DenseLayer::tanh(double x){
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

Network::Network(std::vector<DenseLayer*> arr){
    network = arr;
}

std::vector<double> one_hot_encode(int label, int size){
    std::vector<double> to_return;
    for (int i = 0; i < size; i++){
        if (i == label){
            to_return.push_back(1);
        }
        else{
            to_return.push_back(0);
        }
    }
    return to_return;
}

void Network::train(std::vector<std::vector<double>> training_data, int epochs=10, int num_classes=10, double learning_rate=0.01){
    for (int i = 0; i < epochs; i++){
        MatrixXd del_error;
        double mean_error = 0;
        double count = 0;
        for (auto data_point : training_data){
            double label_i = data_point.back();
            data_point.pop_back();
            std::vector<double> label = one_hot_encode(label_i, num_classes);
            MatrixXd e_label = Map<MatrixXd>(label.data(), label.size(), 1);
            MatrixXd input = Map<MatrixXd>(data_point.data(), data_point.size(), 1);
            //std::cout << "Input is\n" << input << '\n';
            for (int j = 0; j < network.size(); j++){
                input = network[j]->forward_propagate(input);
            }
            //std::cout << "The final layer is\n" << network[network.size() - 1]->outputs << '\n';

            del_error = (2 * (network[network.size() - 1]->outputs - e_label)) / label.size();
            //std::cout << "Error matrix:\n" << del_error << '\n';

            mean_error += mse(network[network.size() - 1]->outputs, e_label);

            for (int j = network.size() - 1; j >= 0 ; j--){
                del_error = network[j]->backward_propagate(del_error, learning_rate);
                //cout << "del_error\n" << del_error << '\n';
            }
            //cout << "Mean error:\n" << mean_error << '\n';
            count++;
            
        }
        mean_error /= training_data.size();
        
        std::cout << "Epoch " << i << " done. Error: " << mean_error << '\n';
    }   
}

int Network::predict(std::vector<double> data_point, int num_classes, int label_i){
    std::vector<double> label = one_hot_encode(label_i, num_classes);
    MatrixXd e_label = Map<MatrixXd>(label.data(), label.size(), 1);
    MatrixXd input = Map<MatrixXd>(data_point.data(), data_point.size(), 1);
    for (int j = 0; j < network.size(); j++){
        input = network[j]->forward_propagate(input);
    }
    int to_return_i = 0;
    double dum = 0;
    //std::cout << network[network.size() - 1]->outputs << '\n';
    for (int i = 0; i < num_classes; i++){
        if (network[network.size() - 1]->outputs.coeff(i, 0) > dum){               
            dum = network[network.size() - 1]->outputs.coeff(i, 0);
            to_return_i = i;
        }
    }
    return to_return_i;
}

double Network::mse(MatrixXd predicted, MatrixXd actual){
    return ((actual - predicted).array().square()).sum() / actual.size();
}

double DenseLayer::sigmoid(double x){
    return 1 / (1 + exp(-x));
}

DenseLayer* Network::get_addr(int idx){
    return network[idx]; 
}