#include <cstdlib>
#include "ann.h"
using namespace std;


int main(int argc, char* argv[]) {
	if (argc != 10) {
		cout << "YOU SUCK LOL, WRONG # OF ARGS!" << endl;
		return -1;
	}

	ifstream train_input,train_output,test_input,test_output,
		structure,weights,encoding;
	double alpha;
	int k;

	//Open files
	train_input.open(argv[1]);
	if(!train_input.is_open()) {
		cout << "YOU SUCK LOL, FAILED TO OPEN " << argv[1] << endl;
		return -1;
	}
	train_output.open(argv[2]);
	if(!train_output.is_open()) {
		cout << "YOU SUCK LOL, FAILED TO OPEN " << argv[2] << endl;
		return -1;
	}
	test_input.open(argv[3]);
	if(!test_input.is_open()) {
		cout << "YOU SUCK LOL, FAILED TO OPEN " << argv[3] << endl;
		return -1;
	}
	test_output.open(argv[4]);
	if(!test_output.is_open()) {
		cout << "YOU SUCK LOL, FAILED TO OPEN " << argv[4] << endl;
		return -1;
	}
	structure.open(argv[5]);
	if(!structure.is_open()) {
		cout << "YOU SUCK LOL, FAILED TO OPEN " << argv[5] << endl;
		return -1;
	}
	weights.open(argv[6]);
	if(!weights.is_open()) {
		cout << "YOU SUCK LOL, FAILED TO OPEN " << argv[6] << endl;
		return -1;
	}
	encoding.open(argv[7]);
	if(!encoding.is_open()) {
		cout << "YOU SUCK LOL, FAILED TO OPEN " << argv[7] << endl;
		return -1;
	}

	//read alpha and k
	alpha = atof(argv[8]);
	k = atoi(argv[9]);

	ann network = ann();
	//init ann with structure and weights
	if (network.init_ann(&structure,&weights,&encoding,&test_output) == -1) {
		cout << "Failed to init ann." << endl;
		return -1;
	}
	// network.printWeights();
	// network.printStructure();
	network.back_propagation(&train_input,&train_output,k,alpha);
	network.printFirstNodeWeights();
	network.classify(&test_input);

	// network.printStructure();


	train_input.close();
	train_output.close();
	test_input.close();
	test_output.close();
	structure.close();
	weights.close();
	encoding.close();
	return 0;
}
