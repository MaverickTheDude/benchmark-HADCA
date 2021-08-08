#include "../include/solution.h"
#include <fstream>

int _solution_::atTime(const double& t) {
	/* 
	* Znajduje indeks wartosci t w wektorze T metoda bisekcji. 
	* Ostatnie kilka krokow odbywa sie liniowo, zeby nie bawic sie corner case'y
    * to do: automatyczny test
	*/
    int begin = 0;
    int end = T.size()-1;

	const int linearSearchRegion = 5;
	while (end - begin > linearSearchRegion) {
		const int mid = (begin + end) / 2;
		if ( abs(t - T(mid)) < 1e-10 ) return mid;
		if (t > T(mid))		begin = mid;
		else				end   = mid;
	}

	for (int i = begin; i <= end; i++)
		if ( abs(t - T(i)) < 1e-10 ) return i;

	throw std::runtime_error("inex not found");
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
	sol.block(1, 0, n, N) = pjoint;
	sol.block(1+n, 0, n, N) = alpha;

	outFile << sol;
	outFile.close();
}