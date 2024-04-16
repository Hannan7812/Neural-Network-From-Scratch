#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>

/**
 * @brief Represents a generic layer in a neural network.
 */
class Layer{
    public:
    Eigen::MatrixXd inputs; /**< The input data to the layer. */
    Eigen::MatrixXd outputs; /**< The output data from the layer. */

    /**
     * @brief Constructs a Layer object with the given input and output sizes.
     * @param input_size The size of the input data.
     * @param output_size The size of the output data.
     */
    Layer(int input_size, int output_size);

    /**
     * @brief Default constructor for the Layer class.
     */
    Layer();
};

/**
 * @brief Represents a dense layer in a neural network.
 */
class DenseLayer : public Layer{
    public:
    Eigen::MatrixXd weights; /**< The weights of the dense layer. */
    Eigen::MatrixXd biases; /**< The biases of the dense layer. */

    /**
     * @brief Forward propagates the input data through the dense layer.
     * @param input_to_layer The input data to the layer.
     * @return The output data from the layer.
     */
    Eigen::MatrixXd forward_propagate(Eigen::MatrixXd input_to_layer);

    /**
     * @brief Applies the activation function to the given matrix.
     * @param apply_on The matrix to apply the activation function on.
     * @return The matrix with the activation function applied.
     */
    Eigen::MatrixXd apply_activation(Eigen::MatrixXd* apply_on);

    /**
     * @brief Backward propagates the output gradient through the dense layer.
     * @param output_gradient The gradient of the output data.
     * @param learning_rate The learning rate for the backward propagation.
     * @return The gradient of the input data.
     */
    Eigen::MatrixXd backward_propagate(Eigen::MatrixXd output_gradient, double learning_rate);

    /**
     * @brief Constructs a DenseLayer object with the given input and output sizes.
     * @param input_size The size of the input data.
     * @param output_size The size of the output data.
     */
    DenseLayer(int input_size, int output_size);

    /**
     * @brief Computes the hyperbolic tangent of the given value.
     * @param x The input value.
     * @return The hyperbolic tangent of the input value.
     */
    static double tanh(double x);

    /**
     * @brief Computes the sigmoid function of the given value.
     * @param x The input value.
     * @return The sigmoid function of the input value.
     */
    static double sigmoid(double x);
};

/**
 * @brief Represents a neural network.
 */
class Network{
    public:
    std::vector<DenseLayer*> network; /**< The layers of the neural network. */

    /**
     * @brief Constructs a Network object with the given layers.
     * @param arr The layers of the neural network.
     */
    Network(std::vector<DenseLayer*> arr);

    /**
     * @brief Trains the neural network using the given training data.
     * @param training_data The training data.
     * @param epochs The number of training epochs.
     * @param num_classes The number of classes in the data.
     * @param learning_rate The learning rate for training.
     */
    void train(std::vector<std::vector<double>> training_data, int epochs, int num_classes, double learning_rate);

    /**
     * @brief Predicts the label of a data point using the trained neural network.
     * @param data_point The data point to predict the label for.
     * @param num_classes The number of classes in the data.
     * @param label_i The index of the label in the output vector.
     * @return The predicted label.
     */
    int predict(std::vector<double> data_point, int num_classes, int label_i);

    /**
     * @brief Computes the mean squared error between the predicted and actual values.
     * @param predicted The predicted values.
     * @param actual The actual values.
     * @return The mean squared error.
     */
    double mse(Eigen::MatrixXd predicted, Eigen::MatrixXd actual);

    /**
     * @brief Gets the address of the dense layer at the specified index.
     * @param idx The index of the dense layer.
     * @return The address of the dense layer.
     */
    DenseLayer* get_addr(int idx);
};