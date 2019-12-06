#pragma once
#include "neural_network.h"
#include <vector>
#include <iostream>
#include <string>

namespace dp {
	class Data_pack
	{
	protected:
		Data_pack();
		Data_pack(arma::vec input);
		Data_pack(arma::vec input, arma::vec target);
		virtual const arma::vec& get_input() const { return input_; }
		virtual const arma::vec& get_target() const { return target_; }
		arma::vec input_;
		arma::vec target_;
	};
	class Mnist_dpack : Data_pack
	{
	public:
		Mnist_dpack(std::string image);
		Mnist_dpack();
		const arma::vec& get_input() const override { return input_; }
		const arma::vec& get_target() const override { return target_; }
		void set_target(const int& target);
		void set_input(const arma::vec& input);
	};
	std::istream& operator>>(std::istream& is, Mnist_dpack& dp);
}