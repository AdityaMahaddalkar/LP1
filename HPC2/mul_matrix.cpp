#include <iostream>
#include <stdlib.h>
#include <ctime>

#define AXIS 500
using namespace std;

int main(){
	int **mat, **mat2;
	int **result_serial, **result_parallel;
	
	mat2 = (int **)malloc(AXIS*sizeof(int *));
	result_serial = (int **)malloc(AXIS*sizeof(int *));
	result_parallel = (int **)malloc(AXIS*sizeof(int *));
	mat = (int **)malloc(AXIS*sizeof(int *));
	
	for(int i = 0;i < AXIS;i ++){
		mat[i] = (int *)malloc(AXIS*sizeof(int));
		mat2[i] = (int *)malloc(AXIS*sizeof(int));
		result_serial[i] = (int *)malloc(AXIS*sizeof(int));
		result_parallel[i] = (int *)malloc(AXIS*sizeof(int));
	}
	
	//Fill vector and matrix
	for(int i = 0;i < AXIS;i ++){
		for(int j = 0;j < AXIS;j ++){
			mat[i][j] = rand()%28393;
			mat2[i][j] = rand()%40523;
			result_serial[i][j] = 0;
			result_parallel[i][j] = 0;
		}
	}
	
	//Serial multiplication
	clock_t timer = clock();
	for(int i = 0;i < AXIS;i ++){
		for(int j = 0;j < AXIS;j ++){
			for(int k = 0;k < AXIS;k ++){
				result_serial[i][j] += mat[i][k] * mat2[k][j];
			}
		}
	}
	cout << "\n Time for serial: " << (float)(clock()-timer) / CLOCKS_PER_SEC;
	
	
	//Parallel multiplication
	timer = clock();
	#pragma omp parallel shared(mat, mat2, result_parallel) num_threads(40)
	{
		#pragma omp for schedule(static)
		for(int i = 0;i < AXIS;i ++){
			#pragma omp parallel for
			for(int j = 0;j < AXIS;j ++){
				for(int k = 0;k < AXIS;k ++){
					result_parallel[i][j] += mat[i][k] * mat2[k][j];
				}
			}
		}
	}
	cout << "\n Time for parallel: " << (float)(clock()-timer) / CLOCKS_PER_SEC;
	cout << endl;
	
	//Test
	for(int i = 0;i < AXIS;i ++){
		for(int j = 0;j < AXIS;j ++){
			if(result_serial[i][j] != result_parallel[i][j]){
				cout << "\nIncorrect result\n";
				cout << result_serial[i][j] << " " << result_parallel[i][j];
				break;
			}
		}
	}
	return 0;
	
}
