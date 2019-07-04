#include <bits/stdc++.h>
#include <chrono>
using namespace std;

int binary_search(vector<int> a, int low, int high, int element){
	if(low <= high){
		int mid = floor((high + low)/2);

		if(a[mid] == element){
			return mid;
		}
		else if(a[mid] > element){
			return binary_search(a, low, mid-1, element);
		}
		else{
			return binary_search(a, mid+1, high, element);
		}
	}
	return -1;
}

int main(){
    
  int k = thread::hardware_concurrency();
  cout << "Number of processors present: " << k << endl;
  vector<int> a(100000);
  generate(a.begin(), a.end(), rand);

  int n = a.size();
  
  auto start = chrono::high_resolution_clock::now();
  sort(a.begin(), a.end());
  auto finish = chrono::high_resolution_clock::now();

  int element = a[38428];
  
  chrono::duration<double> elapsed = finish - start;
  cout << "Processing time for sorting: " << elapsed.count() << endl;
 
  start = chrono::high_resolution_clock::now();
  cout << "For serial result= " << binary_search(a, 0, a.size()-1, element) << endl;
  finish = chrono::high_resolution_clock::now();
  elapsed = finish - start;
  cout << "Processing time for serial binary search: " << elapsed.count() << endl;

  start = chrono::high_resolution_clock::now();
#pragma omp for
  for(int i = 0;i < k;i ++){
    if(binary_search(a, i*n/k, (i+1)*n/k - 1, element) != -1){
      cout << "Processing time for parallel search:" << binary_search(a, i*n/k, (i+1)*n/k-1, element) << endl;
    }
  }
  finish = chrono::high_resolution_clock::now();
  elapsed = finish - start;
  cout << " Processing time for parallel binary search: " << elapsed.count() << endl;

  return 0;
}
