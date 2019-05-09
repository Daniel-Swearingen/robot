#ifndef ANN_H
#define ANN_H
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
using namespace std;

class ann {
public:
	ann(){};
	int init_ann(ifstream*,ifstream*,ifstream*,ifstream*);
	void back_propagation(ifstream*,ifstream*,int,long double);
	void classify(ifstream*);
	vector<long double> fileToLDVector(ifstream*,int);
	vector<int> fileToIntVector(ifstream*,int);
	void printStructure();
	void printWeights();
	void printFirstNodeWeights();
	void printEncoding();

private:
	class Node {
	public:
		void set_a(long double);
		long double get_a();
		void addWeight(long double);
		void setWeight(long double,int);
		long double get_weight(int);

		void printWeights();
	private:
		long double _a;
		vector<long double> _weights;
	};
	int eucDist();
	long double calc_a(long double);
	void stepTwoThree();
	void printAccuracy();
	vector<int> _classifiedVals;
	vector<int> _actualVals;
	vector<vector<Node*> > _nodeNet;
	vector<vector<long double> > _encoding;
};

#endif

/*












*/