#include <bits/stdc++.h>
using namespace std;
class node{

public:
  int data;
  node *left, *right;
  
  node(){
    left = NULL;
    right = NULL;
  }
  node(int d){
    data = d;
    left = NULL;
    right = NULL;
  }
};

class tree{
  node *root;
public:
  tree(){
    root = NULL;
  }

  void addNode(int d){
    if(root == NULL){
      root = new node(d);
    }
    else{
      node *c = root;
      while(c != NULL){
	if(c->data > d){
	  if(c->left == NULL){
	    c->left = new node(d);
	    break;
	  }
	  else{
	    c = c->left;
	  }
	}
	else{
	  if(c->right == NULL){
	    c->right = new node(d);
	    break;
	  }
	  else{
	    c = c->right;
	  }
	}
      }
    }
  }

  node* returnRoot(){
    return root;
  }
};
    
