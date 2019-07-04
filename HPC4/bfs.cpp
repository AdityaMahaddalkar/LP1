#include <bits/stdc++.h>
#include "tree.hpp"
#include <chrono>

using namespace std;

#define clock_now chrono::high_resolution_clock::now
#define duration chrono::duration_cast<chrono::microseconds>

void bfs(node *root){
	queue<node*> q;
	q.push(root);
	node *c;
	while(!q.empty()){

		c = q.front();
		q.pop();
		if(c->left != NULL){
			q.push(c->left);
		}
		if(c->right != NULL){
			q.push(c->right);
		}
	}
}

void parallelBfs(node *root){
	queue<node*> q;
	q.push(root);
	node *c;

#pragma omp parallel sections
	{
		while(!q.empty()){
		c = q.front();
		q.pop();
#pragma omp section
		{
			if(c->left != NULL){
				q.push(c->left);
			}
		}
#pragma omp section
		{
			if(c->right != NULL){
				q.push(c->right);
			}
		}
		}
	}
}

int main(){
	tree t;
	auto start = clock_now();
	for(int i = 0;i < 100000;i ++){
		t.addNode(rand());
	}
	auto stop = clock_now();
	auto time = duration(stop - start);
	cout << "For filling the tree: " << time.count() << " microseconds" << endl;

	start = clock_now();
	bfs(t.returnRoot());
	stop = clock_now();
	time = duration(stop - start);
	cout << "For serial bfs: " << time.count() << " microseconds" << endl;

	start = clock_now();
	parallelBfs(t.returnRoot());
	stop = clock_now();
	time = duration(stop - start);
	cout << " Parallel bfs: " << time.count() << " microseconds" << endl;

	return 0;
}
