#include <bits/stdc++.h>

using namespace std;

int rleCPU(int *in, int n, int *symbolsOut, int *countsOut){

	if (n == 0)
		return 0;

	int outIndex = 0;
	int symbol = in[0];
	int count = 1;

	for(int i = 1;i < n;i ++){

		if(in[i] != symbol) {
			symbolsOut[outIndex] = symbol;
			countsOut[outIndex] = count;
			outIndex++;

			symbol = in[i];
			count = 1;
		}
		else{
			++count;
		}
	}

	symbolsOut[outIndex] = symbol;
	countsOut[outIndex] = count;
	outIndex++;

	return outIndex;
}

int main(){

	int N = 10000000;
	int *in = (int*)malloc(N*sizeof(int));
	int *symbolsOut = (int*)malloc(N*sizeof(int));
	int *countsOut = (int*)malloc(N*sizeof(int));
	int currNo = rand()%30 + 1980;
	for(int i = 0;i < N;i ++){
		if(i % ( rand() % 50 + 23 ) == 0)
			currNo = rand()%30 + 1980;
		in[i] = currNo;
	}

	rleCPU(in, N, symbolsOut, countsOut);

	for(int i = 0;i < 1000;i++){
		cout << "(" << symbolsOut[i] << ", " << countsOut[i] << ")";
	}

	return 0;


}