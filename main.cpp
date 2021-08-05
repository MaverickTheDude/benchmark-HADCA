#include "include/input.h"
#include "include/utils.h"
#include "include/assembly.h"
#include "Eigen/Dense"

#include <fstream>
#include <iostream>

using namespace Eigen;
using std::cout;
using std::endl;


int main(int argc, char* argv[]) {
    // const int Nbodies = argv[1];
    const int Nbodies = 4;
    _input_ input = _input_(Nbodies);

    MatrixXd sol = RK_solver(input);


	// DRUKOWANIE WYNIKOW (pamietac o logTotalEnergy() w RK_solver())
/*
    IOFormat exportFmt(FullPrecision, 0, " ", "\n", "", "", "", "");
	std::ofstream outFile;
	outFile.open("../results.txt");

	if (outFile.fail() ) {
		std::cerr << "nie udalo sie otworzyc pliku.";
		return 2;
	}

	outFile << sol.format(exportFmt) << endl;

	outFile.close();
*/
    cout << "done" << endl;
    return 0;
}