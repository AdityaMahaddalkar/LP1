#include <iostream>
#include <stdlib.h>
#include <ctime>

#define AXIS 30000
using namespace std;

int main(){
	int **mat, *vec;
	int *result_serial, *result_parallel;
	
	vec = (int *)malloc(AXIS*sizeof(int));
	result_serial = (int *)malloc(AXIS*sizeof(int));
	result_parallel = (int *)malloc(AXIS*sizeof(int));
	mat = (int **)malloc(AXIS*sizeof(int *));
	for(int i = 0;i < AXIS;i ++){
		mat[i] = (int *)malloc(AXIS*sizeof(int));
	}
	
	//Fill vector and matrix
	for(int i = 0;i < AXIS;i ++){
		vec[i] = rand()%23943;
		for(int j = 0;j < AXIS;j ++){
			mat[i][j] = rand()%28393;
		}
	}
	
	//Serial multiplication
	clock_t timer = clock();
	for(int i = 0;i < AXIS;i ++){
		for(int j = 0;j < AXIS;j ++){
			result_serial[i] += mat[i][j] * vec[j];
		}
	}
	cout << "\n Time for serial: " << (float)(clock()-timer) / CLOCKS_PER_SEC;
	
	
	//Parallel multiplication
	timer = clock();
	#pragma omp parallel shared(mat, vec, result_parallel)
	{
		#pragma omp for
		for(int i = 0;i < AXIS;i ++){
			for(int j = 0;j < AXIS;j ++){
				result_parallel[i] += mat[i][j] * vec[j];
			}
		}
	}
	cout << "\n Time for parallel: " << (float)(clock()-timer) / CLOCKS_PER_SEC;
	cout << endl;
	
	//Test
	for(int i = 0;i < AXIS;i ++){
		if(result_serial[i] != result_parallel[i]){
			cout << "Incorrect result";
		}
	}
	return 0;
	
}
