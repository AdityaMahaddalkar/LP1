#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;
void mergesort(int a[], int i, int j);
void merge(int a[], int i1, int j1, int i2, int j2);
void parallel_mergesort(int b[], int i, int j);

#define SIZE 20000

int main(){
	int *a, *b;
	a = (int *)malloc(SIZE*sizeof(int));
	b = (int *)malloc(SIZE*sizeof(int));
	for(int i = 0;i < SIZE;i ++){
		a[i] = rand() % 182384;
		b[i] = a[i];
	}
	
	clock_t timer = clock();
	mergesort(a, 0, SIZE-1);
	cout << "\n Serial sorting time : " << (float)(clock()-timer) / CLOCKS_PER_SEC;
	
	timer = clock();
	parallel_mergesort(b, 0, SIZE-1);
	cout << "\n Parallel sorting time : " << (float)(clock()-timer) / CLOCKS_PER_SEC;
	
	//Test case
	/*
	cout << "\nSorted array: \n";
	
	for(int i = 0;i < SIZE;i ++){
		cout << a[i] << endl;
	}
	*/
	
	//Validation
	for(int i = 0;i < SIZE;i ++){
		if(a[i] != b[i]){
			cout << "Error";
			break;
		}
	}
}

void mergesort(int a[], int i, int j){
	
	int mid;
	
	if(i < j)
	{
		mid = (i+j)/2;
		mergesort(a, i, mid);
		mergesort(a, mid+1, j);
		merge(a, i, mid, mid+1, j);
	}
}

void parallel_mergesort(int b[], int i, int j){
	
	int mid;
	
	if(i < j){
		
		mid = (i+j)/2;
		
		#pragma omp parallel sections
		{
			#pragma omp section
			{
				parallel_mergesort(b, i, mid);
			}
			
			#pragma omp section
			{
				parallel_mergesort(b, mid+1, j);
			}
		}
		
		merge(b, i, mid, mid+1, j);
	}
}

void merge(int a[], int i1, int j1, int i2, int j2){
	int temp[SIZE];
	int i, j, k;
	i = i1;
	j = i2;
	k = 0;
	
	while(i<=j1 && j<=j2){
		if(a[i]<a[j])
			temp[k++] = a[i++];
		else
			temp[k++] = a[j++];
	}
	
	while(i<=j1)
		temp[k++] = a[i++];
	while(j<=j2)
		temp[k++] = a[j++];
	
	for(i=i1, j=0;i<=j2;i ++, j ++){
		a[i] = temp[j];
	}
}

