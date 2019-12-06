#include  "data_pack.h"
#include <string>
#include <utility>

namespace dp {
	Data_pack::Data_pack()
		:input_{ arma::vec() }
	{}
	Mnist_dpack::Mnist_dpack()
	{
		input_ = arma::zeros(784);
	}
	Data_pack::Data_pack(arma::vec input)
		: input_{ std::move(input) }
	{}
	Data_pack::Data_pack(arma::vec input, arma::vec target)
		: input_{ input }, target_{ target }
	{}
	void Mnist_dpack::set_target(const int& target)
	{
		target_ = arma::zeros(10);
		target_[target] = 1;
	}
	void Mnist_dpack::set_input(const arma::vec& input)
	{
		if (input.size() != 784)
			throw std::exception("image in incorrect file format for mnist");
		input_ = input;
	}
	Mnist_dpack::Mnist_dpack(std::string image) {
		std::vector<double> input; //all data converted to relevant format
		std::string value; //data in string format
		int target = std::stoi(std::string(1, image[0]));
		image.erase(image.begin(), image.begin() + 2);
		//convert image string to relevant grey scale and target doubles and store them to be returned
		for (const char& ch : image) {
			switch (ch) {
			case '0': case '1':
			case '2': case '3':
			case '4': case '5':
			case '6': case '7':
			case '8': case '9':
			case '.':
				value += ch;
				break;
			default:
				input.push_back(std::stod(value) / 255.0);
				value.clear();
				break;
			}
		}
		input.push_back(std::stod(value) / 255.0); //store last piece of data
		set_target(target);
		input_ = input;
	}
	std::istream& operator>>(std::istream& is, Mnist_dpack& dp) {
		std::string line;
		std::getline(is, line);
		dp = line;
		return is;
	}
}