#include <iostream>
#include <stdlib.h>
#include <vector>
#include <omp.h>
#include <ctime>

using namespace std;

#define ARRSIZE 20000000

int main(){

	int *A, *B, *C, *D;
	
	A = (int *)malloc(ARRSIZE*sizeof(int));
	B = (int *)malloc(ARRSIZE*sizeof(int));
	C = (int *)malloc(ARRSIZE*sizeof(int));
	D = (int *)malloc(ARRSIZE*sizeof(int));
	
	for(int i = 0;i < ARRSIZE;i ++){
		A[i] = (rand()%29301);
		B[i] = (rand()%24341);
	}
	cout << "\nFilled";
	//Serial addition
	clock_t timer = clock();
	for(int i = 0;i < ARRSIZE;i ++){
		
		C[i] = (A[i] + B[i]);
	}
	cout << "\n Serial addition requires " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds";

	cout << "\nParallel";
	//Parallel Addition
	timer = clock();
	#pragma omp parallel for default(none) shared(A, B, D)
	for(int i = 0;i < ARRSIZE;i ++){
		D[i] = (A[i] + B[i]);
	}
	cout << "\n Parallel addition requires " << (float)(clock() - timer) / CLOCKS_PER_SEC << " seconds";
	cout << endl;
	return 0;
}
