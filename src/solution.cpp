#include "../include/solution.h"
#include "../include/constants.h"
#include <fstream>

std::pair<int, const bool> _solution_::atTime(const double& t, const _input_& input) const {
	/* 
	* Znajduje indeks wartosci t w wektorze T metoda bisekcji. 
	* Ostatnie kilka krokow odbywa sie liniowo, zeby nie bawic sie corner case'y
	* Zwraca pare <indeks, NodeValue>
	*/
	const double& dt = input.dt;
    int begin = 0;
    int end = T.size()-1;

	const int linearSearchRegion = 5;
	while (end - begin > linearSearchRegion) {
		const int mid = (begin + end) / 2;
		if ( abs(t - T(mid)) < 1e-10 ) return std::make_pair(mid, NODE_VALUE);
		if (t > T(mid))		begin = mid;
		else				end   = mid;
	}

	for (int i = begin; i <= end; i++) {
		if ( abs(T(i) 	   - t) < 1e-10 ) return std::make_pair(i, NODE_VALUE);
		if ( abs(T(i)+dt/2 - t) < 1e-10 ) return std::make_pair(i, INTERMEDIATE_VALUE);
	}

	throw std::runtime_error("atTime: index not found");
}

dataJoint _solution_::getDynamicValues(const int index, const _input_& input) const {
	const int Nvars = alpha.rows();
	dataJoint data = dataJoint(Nvars);
	data.t       = T(index);
	data.alpha   = alpha.col(index);
	data.dalpha  = dalpha.col(index);
	data.d2alpha = d2alpha.col(index);
	data.lambda  = lambda.col(index);
	data.pjoint  = pjoint.col(index);

	return data;
}

void _solution_::print() const {
	IOFormat exportFmt(FullPrecision, 0, " ", "\n", "", "", "", "");
	std::ofstream outFile;
	outFile.open("../output/results.txt");

	if (outFile.fail() )
		throw std::runtime_error("nie udalo sie otworzyc pliku.");

	const int N = alpha.cols();
	const int n = alpha.rows();
	MatrixXd sol(2*n+1, N);

	sol.row(0) = T;
	sol.block(1, 0, n, N) = d2alpha;
	sol.block(1+n, 0, n, N) = alpha;

	outFile << sol;
	outFile.close();
}