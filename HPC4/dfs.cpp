#include <bits/stdc++.h>
#include "tree.hpp"
#include <chrono>

using namespace std;

void serialDfs(node *root){
  if(root != NULL){
    serialDfs(root->left);
    //cout << root->data << endl;
    serialDfs(root->right);
  }
}

void parallelDfs(node *root){
	if(root != NULL){
#pragma omp parallel sections
		{
#pragma omp section
			{
				parallelDfs(root->left);
			}
#pragma omp section
			{
				//cout << root->data << endl;
			}
#pragma omp section
			{
				parallelDfs(root->right);
			}
		}
	}
}
int main(){
#define clock_now chrono::high_resolution_clock::now
  tree t;
  auto start = chrono::high_resolution_clock::now();
  for(int i = 0;i < 1000000;i ++){
    t.addNode(rand());
  }
  auto finish = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>(finish - start);
  cout << "For adding: " << duration.count() << endl;

  start = clock_now();
  serialDfs(t.returnRoot());
  finish = clock_now();
  duration = chrono::duration_cast<chrono::microseconds>(finish - start);
  cout << "For serial processing  " << duration.count() << endl;
  
  start = clock_now();
  parallelDfs(t.returnRoot());
  finish = clock_now();
  duration = chrono::duration_cast<chrono::microseconds>(finish - start);
  cout << "For parallel processing: " << duration.count() << endl;
  
}
