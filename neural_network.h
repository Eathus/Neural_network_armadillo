#pragma once
#include <armadillo>
#include <array>

namespace nn {
	double xavier(const int& layer_from_size);
	double cost_out_derivative(const double& out, const double& target);

	class Network {
		double learning_rate_, lambda_;
		std::vector<int> layer_sizes_;
		std::vector<arma::dmat> weights_;
		std::vector<arma::vec> biases_;
		std::vector<arma::dmat> weight_grads_;
		std::vector<arma::vec> bias_grads_;
		std::vector<arma::dmat> inputs_;
		std::vector<arma::dmat> activations_;
		void apply_grads(int batch_size, int data_set_size, double lambda, double learning_rate);
		void feed_forward(const arma::dmat& input_activations);
	public:
		Network(std::vector<int> layer_sizes, double learning_rate_, double lambda_);
		void backprop(const arma::dmat& input_activations, const arma::vec& target);
		void mini_batch_train(std::ifstream& practice_file, const std::string& weight_file_name, const std::string& bias_file_name,
			const std::string& test_file_name, const int& data_set_size, const int& epochs, const int& batch_size, const bool& test);
		double evaluate(std::ifstream& test_file, const int& sample_size);
		bool correct_guess(const arma::dmat& target);
		void save_current_weights(const std::string& save_file_name);
		void save_current_biases(const std::string& save_file_name);
		void set_weights();
		void set_weights(std::ifstream& weights_save);
		void set_biases(std::ifstream& biases_save);
		int weights_size();
		int biases_size();
	};
}