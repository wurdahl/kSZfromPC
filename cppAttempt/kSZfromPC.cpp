#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <vector>
#include <valarray>

std::valarray<double> getRadialUnitVecs(std::valarray<double> r, std::valarray<double> theta, std::valarray<double> phi) {
	cos(phi)*sin(theta);
}

int main() {

	std::valarray<double> in = { 1.0,2.0,3.0 };

	std::valarray<double> result = getRadialUnitVecs(in, in, in);
	std::cout << result[1] << std::endl;

	return 0;
}