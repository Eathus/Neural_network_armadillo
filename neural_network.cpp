#define _USE_MATH_DEFINES
#include "neural_network.h"
#include <cmath>
#include "data_pack.h"
#include <iomanip>

namespace nn {
	double f_rand(const double& f_min, const double& f_max) {	//generate random double from f_min to f_max
		const auto f = static_cast<double>(rand()) / RAND_MAX;
		return f_min + f * (f_max - f_min);
	}

	void shuffle(std::vector<std::string>& vs) {
		for (int i = 0; i < vs.size(); i++) {
			std::swap(vs[i], vs[rand() % vs.size()]);
		}
	}

	double xavier(const int& layer_from_size) {	//function used to initialize initial weights.
		double val = sqrt(1.0 / static_cast<double>(layer_from_size));
		return val;
	}
	double cost_out_derivative(const double& out, const double& target)
		//derivative of the cost with respect to the out-value for the neurons in the last layer
	{
		const double val = out - target;
		return val;
	}
	arma::dmat sigmoid(const arma::dmat input) { //sigmoid function for out value
		arma::dmat result = 1.0 / (1.0 + arma::exp(-input));
		return result;
	}
	arma::dmat sigmoid_prime(const arma::dmat input) { //sigmoid function for out value
		arma::dmat result = sigmoid(input) % (1 - sigmoid(input));
		return result;
	}
	Network::Network(std::vector<int> layer_sizes, double learning_rate, double lambda)
		:layer_sizes_{ layer_sizes }, learning_rate_{ learning_rate }, lambda_{ lambda }
	{
		for (int i = 1; i < layer_sizes_.size(); i++) {
			//double xavier_val = xavier(layer_sizes[i - 1]);
			biases_.push_back(arma::zeros(layer_sizes_[i]));
			bias_grads_.push_back(arma::zeros(layer_sizes_[i]));
			weights_.push_back(arma::dmat(layer_sizes_[i], layer_sizes_[i - 1]));	//might become an issue
			weight_grads_.push_back(arma::zeros(layer_sizes_[i], layer_sizes_[i - 1]));
		}
	}

	void Network::set_weights() {
		for (int i = 0; i < weights_.size(); i++) {
			for (double& d : weights_[i])
				d = f_rand(-xavier(layer_sizes_[i]), xavier(layer_sizes_[i]));
		}
	}
	void Network::set_weights(std::ifstream& weights_save) {
		for (arma::dmat& dma : weights_) {
			for (double& d : dma)
				weights_save >> d;
		}
	}
	void Network::set_biases(std::ifstream& biases_save) {
		for (arma::vec& ve : biases_) {
			for (double& d : ve)
				biases_save >> d;
		}
	}

	void Network::apply_grads(int batch_size, int data_set_size, double lambda, double learning_rate) {
		for (int i = 0; i < weights_.size(); i++) {
			weights_[i] = (1 - learning_rate * lambda / static_cast<double>(data_set_size))* weights_[i] - (learning_rate / static_cast<double>(batch_size))* weight_grads_[i];
			weight_grads_[i].zeros();
		}
		for (int i = 0; i < biases_.size(); i++) {
			biases_[i] -= bias_grads_[i] / static_cast<double>(batch_size);
			bias_grads_[i].zeros();
		}
	}
	void Network::feed_forward(const arma::dmat& input_activations) {
		std::vector<arma::dmat> inputs{ input_activations };
		std::vector<arma::dmat> activations{ input_activations };
		for (int i = 1; i < layer_sizes_.size(); i++) {
			arma::dmat input = weights_[i - 1] * activations[i - 1] + biases_[i - 1];
			inputs.push_back(input);
			activations.push_back(sigmoid(input));
		}
		inputs_ = inputs;
		activations_ = activations;
	}

	void Network::backprop(const arma::dmat& input_activations, const arma::vec& target) {
		//feedforward
		feed_forward(input_activations);

		//calculate output error (delta)
		std::vector<arma::dmat> deltas{ (activations_[activations_.size() - 1] - target) };

		//backward pass/backproprpgate
		for (int i = weights_.size() - 1; i > 0; i--) {
			arma::dmat delta = (weights_[i].t() * deltas[0]) % sigmoid_prime(inputs_[i]);
			deltas.emplace(deltas.begin(), delta);
		}

		//gradient descent

		for (int i = weights_.size() - 1; i >= 0; i--) {
			arma::dmat bias_grad = deltas[i];
			arma::dmat weight_grad = deltas[i] * activations_[i].t();
			weight_grads_[i] += weight_grad;
			bias_grads_[i] += bias_grad;
		}
	}
	bool Network::correct_guess(const arma::dmat& target) {
		double guess_value = 0;
		int guess;
		for (int i = 0; i < target.size(); i++) {
			if (activations_[activations_.size() - 1][i] > guess_value) {
				guess_value = activations_[activations_.size() - 1][i];
				guess = i;
			}
		}
		return target[guess] == 1;
	}
	double Network::evaluate(std::ifstream& test_file, const int& sample_size) { //evaluate network accuracy
		int correct = 0;
		std::vector<std::string> string_images;
		for (int i = 0; test_file && i < sample_size; i++) {
			std::string line;
			std::getline(test_file, line);
			string_images.push_back(line);
		}
		for (int i = 0; i < sample_size; i++) {
			dp::Mnist_dpack image = string_images[i];
			feed_forward(image.get_input());
			if (correct_guess(image.get_target())) correct++; //keep track of correct guesses
		}

		double accuracy = 100 * static_cast<double>(correct) / sample_size; //calculate accuracy
		return accuracy;
	}
	void Network::mini_batch_train(std::ifstream& practice_file, const std::string& weight_file_name, const std::string& bias_file_name,
		const std::string& test_file_name, const int& data_set_size, const int& epochs, const int& batch_size, const bool& test)
	{
		if (data_set_size % batch_size != 0)
			throw std::exception("cannot parson batches of this batch size with a data set of this size");
		srand(time(NULL));
		double max_accuracy = 0;
		std::vector<std::string> string_images;
		for (int i = 0; practice_file && i < data_set_size; i++) {
			std::string line;
			std::getline(practice_file, line);
			string_images.push_back(line);
		}
		for (int i = 0; i < epochs; i++) {
			shuffle(string_images);
			for (int j = 0; j < data_set_size; j++) {
				dp::Mnist_dpack image = string_images[j];
				backprop(image.get_input(), image.get_target());

				if ((j + 1) % batch_size == 0) apply_grads(batch_size, data_set_size, lambda_, learning_rate_);
			}
			if (test) {
				std::ifstream test_file{ test_file_name };
				double accuracy = evaluate(test_file, 10000);
				std::cout << "Epoch " << i << ":\t" << accuracy << "%\n";
				if (max_accuracy < accuracy) {
					save_current_weights(weight_file_name);
					save_current_biases(bias_file_name);
					max_accuracy = accuracy;
				}
			}
		}
	}
	void Network::save_current_weights(const std::string& save_file_name) { //save weights to save file
		std::ofstream save_file;
		save_file.open(save_file_name, std::ios_base::trunc); //reset previous irrelevant weights
		for (const arma::dmat& dma : weights_) {
			for (const double& d : dma)
				save_file << std::defaultfloat << std::setprecision(std::numeric_limits<double>::digits10)
				<< d << '\n';
		}
	}
	void Network::save_current_biases(const std::string& save_file_name) { //save biases to save file
		std::ofstream save_file;
		save_file.open(save_file_name, std::ios_base::trunc); //reset previous irrelevant biases
		for (const arma::vec& ve : biases_) {
			for (const double& d : ve)
				save_file << std::defaultfloat << std::setprecision(std::numeric_limits<double>::digits10)
				<< d << '\n';
		}
	}
	int Network::weights_size() {
		int size = 0;
		for (arma::dmat dma : weights_) size += dma.size();
		return size;
	}
	int Network::biases_size() {
		int size = 0;
		for (arma::vec ve : biases_) size += ve.size();
		return size;
	}
}