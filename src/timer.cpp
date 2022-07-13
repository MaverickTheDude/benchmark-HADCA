#include "../include/timer.h"
#include "fstream"
#include <iostream>

// global timing variables:
long long int Timer_Fwd_Leaves = 0;
long long int Timer_Fwd_Asm = 0;
long long int Timer_Fwd_Disasm = 0;
long long int Timer_Fwd_Force_Leaves = 0;
long long int Timer_Fwd_Force_Asm = 0;
long long int Timer_Fwd_Force_Disasm = 0;
long long int Timer_Fwd_Acc_Leaves = 0;
long long int Timer_Fwd_Acc_Asm = 0;
long long int Timer_Fwd_Acc_Disasm = 0;
long long int Timer_Fwd_Cumsum = 0;
long long int Timer_Fwd_Proj = 0;

long long int Timer_Adj_Leaves = 0;
long long int Timer_Adj_Asm = 0;
long long int Timer_Adj_Disasm = 0;
long long int Timer_Adj_Proj = 0;
long long int Timer_Adj_Interp = 0;
long long int Timer_Adj_bc = 0;


long long int Timer_Fwd_Global = 0;
long long int Timer_Adj_Global = 0;

long long int tmpTimerMain = 0;
long long int tmpTimerRHS = 0;

procTimer::procTimer(long long int& counter) : m_counter(counter), m_Stopped(false) {
    m_StartTimepoint = std::chrono::high_resolution_clock::now();
}

procTimer::~procTimer() {
    if (!m_Stopped)
        Stop();
}

void procTimer::Start() {
    if (m_Stopped) {
        m_StartTimepoint = std::chrono::high_resolution_clock::now();
        m_Stopped = false;
    }
}

void procTimer::Stop()
{
    auto endTimepoint = std::chrono::high_resolution_clock::now();

    long long start = std::chrono::time_point_cast<std::chrono::milliseconds>(m_StartTimepoint).time_since_epoch().count();
    long long end   = std::chrono::time_point_cast<std::chrono::milliseconds>(endTimepoint).time_since_epoch().count();
    m_counter += end - start;

    m_Stopped = true;
}

void printDetailedTimes(int Nbodies, int Nthreads) {
	std::ofstream outFile;
    const std::string filepath = "../output/TimesDetailed.txt";
	outFile.open(filepath);
    
    std::string header = "Bodies: " + std::to_string(Nbodies) + " Threads: " + std::to_string(Nthreads) + '\n';

	if (outFile.fail() )
		throw std::runtime_error("printDetailedTimes(...): nie udalo sie otworzyc pliku.");

    long long int Fwd_subset = Timer_Fwd_Force_Leaves + Timer_Fwd_Force_Asm + Timer_Fwd_Force_Disasm
                             + Timer_Fwd_Acc_Leaves + Timer_Fwd_Acc_Asm + Timer_Fwd_Acc_Disasm;

    outFile << header
            <<  "Fwd_Global"        << '\t' << (double)Timer_Fwd_Global / 1000         << '\n'
            <<  "Fwd_Leaves"        << '\t' << (double)Timer_Fwd_Leaves / 1000         << '\n'
            <<  "Fwd_Asm"           << '\t' << (double)Timer_Fwd_Asm / 1000            << '\n'
            <<  "Fwd_Disasm"        << '\t' << (double)Timer_Fwd_Disasm / 1000         << '\n'
            // <<  "Fwd_ForceLeaves"  << '\t' << (double)Timer_Fwd_Force_Leaves / 1000   << '\n'
            // <<  "Fwd_ForceAsm"     << '\t' << (double)Timer_Fwd_Force_Asm / 1000      << '\n'
            // <<  "Fwd_ForceDisasm"  << '\t' << (double)Timer_Fwd_Force_Disasm / 1000   << '\n'
            // <<  "Fwd_AccLeaves"    << '\t' << (double)Timer_Fwd_Acc_Leaves / 1000     << '\n'
            // <<  "Fwd_AccAsm"       << '\t' << (double)Timer_Fwd_Acc_Asm / 1000        << '\n'
            // <<  "Fwd_AccDisasm"    << '\t' << (double)Timer_Fwd_Acc_Disasm / 1000     << '\n'
            <<  "Fwd_Proj"          << '\t' << (double)Timer_Fwd_Proj / 1000           << '\n'
            <<  "Fwd_Subset"        << '\t' << (double)Fwd_subset / 1000               << '\n' // Force + Acc phases combined
            <<  "Fwd_Cumsum"        << '\t' << (double)Timer_Fwd_Cumsum / 1000         << '\n'
            <<  "Adj_Global"        << '\t' << (double)Timer_Adj_Global / 1000         << '\n'
            <<  "Adj_Leaves"        << '\t' << (double)Timer_Adj_Leaves / 1000         << '\n'
            <<  "Adj_Asm"           << '\t' << (double)Timer_Adj_Asm / 1000            << '\n'
            <<  "Adj_Disasm"        << '\t' << (double)Timer_Adj_Disasm / 1000         << '\n'
            <<  "Adj_Proj"          << '\t' << (double)Timer_Adj_Proj / 1000           << '\n'
            <<  "Adj_Interp"        << '\t' << (double)Timer_Adj_Interp / 1000         << '\n'
            <<  "Adj_BoundaryCond"  << '\t' << (double)Timer_Adj_bc / 1000             << '\n';

	outFile.close();
    std::cout << "printed detailed timings to " + filepath << std::endl;
}
