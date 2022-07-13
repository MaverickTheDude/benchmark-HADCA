#pragma once
#include <chrono>
using namespace std::literals::chrono_literals;

extern long long int Timer_Fwd_Leaves;
extern long long int Timer_Fwd_Asm;
extern long long int Timer_Fwd_Disasm;
extern long long int Timer_Fwd_Force_Leaves;
extern long long int Timer_Fwd_Force_Asm;
extern long long int Timer_Fwd_Force_Disasm;
extern long long int Timer_Fwd_Acc_Leaves;
extern long long int Timer_Fwd_Acc_Asm;
extern long long int Timer_Fwd_Acc_Disasm;
extern long long int Timer_Fwd_Cumsum;
extern long long int Timer_Fwd_Proj;
extern long long int Timer_Adj_Leaves;
extern long long int Timer_Adj_Asm;
extern long long int Timer_Adj_Disasm;
extern long long int Timer_Adj_Proj;
extern long long int Timer_Adj_Interp;
extern long long int Timer_Fwd_Global;
extern long long int Timer_Adj_Global;
extern long long int Timer_Adj_bc;

extern long long int tmpTimerMain;
extern long long int tmpTimerRHS;

void printDetailedTimes(int, int);

class procTimer {
public:
    procTimer() = delete;
    procTimer(long long int& counter);
    ~procTimer();
    void Stop();
    void Start();

private:
    long long int& m_counter;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_StartTimepoint;
    bool m_Stopped;
};