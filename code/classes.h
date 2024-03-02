#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>

class Layer{
    public:
    Eigen::MatrixXd inputs;
    Eigen::MatrixXd outputs;
    Layer(int input_size, int output_size);
    Layer();
};

class DenseLayer : public Layer{
    public:
    Eigen::MatrixXd weights;
    Eigen::MatrixXd biases;

    Eigen::MatrixXd forward_propagate(Eigen::MatrixXd input_to_layer);
    Eigen::MatrixXd apply_activation(Eigen::MatrixXd* apply_on);
    Eigen::MatrixXd backward_propagate(Eigen::MatrixXd output_gradient, double learning_rate);

    DenseLayer(int input_size, int output_size);

    static double tanh(double x);
};

class Network{
    public:
    std::vector<DenseLayer*> network;

    Network(std::vector<DenseLayer*> arr);

    void train(std::vector<std::vector<double>> training_data, int epochs, int num_classes, double learning_rate);
    int predict(std::vector<double> data_point, int num_classes, int label_i);

    double mse(Eigen::MatrixXd predicted, Eigen::MatrixXd actual);
    DenseLayer* get_addr(int idx);
};