#include <omp.h>
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <stdlib.h>
#include <algorithm>

#define SIZE 1000

using namespace std;

void bubblesort(int *a){
	
	for(int i = 0;i < SIZE;i ++){
		bool stop = true;
		
		for(int j = 0;j < SIZE-1;j ++){
			if( a[j] > a[j+1] ){
				swap(a[j], a[j+1]);
				stop = false;
			}
		}
		
		if(stop)
			break;
	}
}


void parallel_bubblesort(int *a){
	
	for(int i = 0;i < SIZE;i ++){
		
		
		int first = i%2;
		
		#pragma omp parallel for default(none), shared(a, first)
		
		for(int j = first;j < SIZE - 1;j += 2){
			if(a[j] > a[j+1]){
				swap(a[j], a[j+1]);
			}
		}
	}
}

int main()
{
	int *a, *b;
	a = (int *)malloc(SIZE*sizeof(int));
	b = (int *)malloc(SIZE*sizeof(int));
	
	for(int i = 0;i < SIZE;i ++){
		a[i] = rand()%324392;
		b[i] = a[i];
	}	
	
	//Sequential bubble sort
	clock_t timer = clock();
	bubblesort(a);
	cout << "\nTime for serial bubble sort: " << (float)(clock() - timer) / CLOCKS_PER_SEC;
	
	//Parallel bubble sort
	timer = clock();
	parallel_bubblesort(b);
	cout << "\nTime for parallel bubble sort: " << (float)(clock() - timer) / CLOCKS_PER_SEC;
	
	//Verification
	for(int i = 0;i < SIZE;i ++){
		if(a[i] != b[i]){
			cout << "\nIncorrect result";
			break;
		}
	}
}
