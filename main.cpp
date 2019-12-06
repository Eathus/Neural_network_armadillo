#include "neural_network.h"
#include <string>
#include <iostream>
#include <armadillo>

int lines_in_file(const std::string& file_name) {
	int lines = 0;
	std::ifstream ifs{ file_name };
	if (!ifs) throw std::exception("file cannot be found");
	for (std::string s; ifs; std::getline(ifs, s))
		if (!s.empty()) lines++;
	return lines;
}

int main()
{
	const std::vector<int> layers{ 784, 30, 10 };
	constexpr double learning_rate = 0.5;
	constexpr double lambda = 5;
	constexpr int epochs = 30;
	constexpr int file_sample_train = 50000;
	constexpr int file_sample_test = 10000;
	constexpr  int adjust_interval = 10;
	const std::string practice_file_name = "mnist_train.csv";
	const std::string test_file_name = "mnist_test.csv";
	const std::string validation_file_name = "mnist_validation.csv";
	const std::string weight_save_file_name = "weights.txt";
	const std::string biases_save_file_name = "biases.txt";

	std::ifstream practice_file{ practice_file_name };
	std::ifstream test_file{ test_file_name };
	std::ifstream validation_file{ validation_file_name };
	std::ifstream i_weights_save{ weight_save_file_name };
	std::ifstream i_biases_save{ biases_save_file_name };

	nn::Network network{ layers, learning_rate, lambda };
	if (lines_in_file("weights.txt") == network.weights_size() && lines_in_file("biases.txt") == network.biases_size()) {
		network.set_weights(i_weights_save);
		network.set_biases(i_biases_save);
	}
	else network.set_weights();

	//std::ofstream weights_save{ weight_save_file_name };
	//std::ofstream biases_save{ biases_save_file_name };

	//network.stochastic_train(practice_file, file_sample_train, epochs, true);
	network.mini_batch_train(practice_file, weight_save_file_name, biases_save_file_name, test_file_name, file_sample_train, epochs, adjust_interval, true);
	//network.stochastic_train(practice_file, file_sample_train, epochs, true);
	//network.save_current_weights(weight_save_file_name);
	//network.save_current_biases(biases_save_file_name);

	//practice_file.close();
	//practice_file.open(practice_file_name);
	//
	std::cout << "Accuracy over " << file_sample_test << " images:\t" << network.evaluate(validation_file, file_sample_test) << "%\n";
	system("pause");
	return 0;
}