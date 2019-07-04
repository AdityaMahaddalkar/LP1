#include <bits/stdc++.h>

class node{
  int data;
  node *left, *right;
public:
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
      while(c->left != NULL || c->right != NULL){
	if(c->data > d){
	  if(c->left == NULL){
	    c->left = new node(d);
	  }
	  else{
	    c = c->left;
	  }
	}
	else{
	  if(c->right == NULL){
	    c->right = new node(d);
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
    
