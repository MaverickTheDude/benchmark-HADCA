#include <iostream>
#include <iomanip> // std::setprecision
#include <string>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <omp.h>
#include <chrono>

using namespace std::literals::chrono_literals;
using std::cout;
using std::endl;
const std::string getJobId(int, int);
struct times {double t, wt;};
times timeTask(int, int, int);


int main() {
	const std::vector<int> threads {1, 2, 4, 8};
	const std::vector<int> bodies  {32, 64, 128, 256};
	const int Nth = threads.size();
	const int Nbodies = bodies.size();
	int mean = 3;

	std::ofstream outFile;
	outFile.open("../output/times.txt");
	if (outFile.fail() ) {
		std::cerr << "nie udalo sie otworzyc pliku.";
		throw std::runtime_error("timing.cpp: nie udalo sie otworzyc pliku");
	}
	outFile << "Bodies:";
	for (int i = 0; i < Nbodies; i++) outFile << "\t\t" << bodies[i];
	outFile << "\nthreads -------------------------------------- (mean = " << mean << ")\n";

	for (int i = 0; i < Nth; i++) {
		outFile << threads[i] << " | \t";
		for (int j = 0; j < Nbodies; j++) {
			times T = timeTask(threads[i], bodies[j], mean);
			outFile << std::fixed /* << std::setprecision(6) << T.t << " / "  */
								  << std::setprecision(6) << T.wt << "\t";
		}
		outFile << endl;
	}

	outFile.close();
	return 0;
}


times timeTask(int threads, int bodies, int mean) {
	double* tab  = new double[mean];
	double* wtab = new double[mean];
	clock_t start, end;
	double wt1, wt2;
	double t = 0.0, wt = 0.0;
	const std::string job = getJobId(threads, bodies);

	for (int i = 0; i < mean; i++) {
		start = clock();
// auto start = std::chrono::high_resolution_clock::now();
		wt1 = omp_get_wtime();
		system(job.c_str());
		wt2 = omp_get_wtime();
		end = clock();
		wtab[i] = wt2 - wt1;
		tab[i] = double(end - start) / double(CLOCKS_PER_SEC);
// auto end = std::chrono::high_resolution_clock::now();
// std::chrono::duration<double> duration = end - start;
// tab[i] = duration.count();
	}

	cout << "czasy dla zadania: Nthreads = " << threads << " Nbodies = " << bodies << '\n';
	for (int i = 0; i < mean; i++)	
		cout << std::fixed << std::setprecision(6) << tab[i] << '\t';
	cout << "\t (CPU time)\n";
	for (int i = 0; i < mean; i++)
		cout << std::fixed << std::setprecision(6) << wtab[i] << '\t';
	cout << "\t (wall time)";
	cout << '\n' << endl;

	// oblicz średnią wszystkich czasów:
	for (int i = 0; i < mean; i++){
		t += tab[i];
		wt+= wtab[i];
	}
	t = t / (double) mean;
	wt = wt / (double) mean;
	times out = {t, wt};

	delete [] tab;
	delete [] wtab;
	return out;
}

const std::string getJobId(int threads, int bodies) {
	const std::string path = "~/code/examplecp/builddir/main";
	const std::string threads_c = std::to_string(threads);
	const std::string bodies_c  = std::to_string(bodies);
	return path + " " + threads_c + " " + bodies_c;
}
