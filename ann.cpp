#include "ann.h"


int ann::init_ann(ifstream* structure, ifstream* weights, ifstream* encoding, ifstream* test_output) {
	Node* node;
	int pos = 0;
	string s,w,e;
	string sub;
	vector<Node*> tempStructure;
	vector<long double> tempEnc;


	_actualVals = fileToIntVector(test_output,1);
	while(getline(*encoding,e)) {
		while(pos != -1) {
			pos = e.find(" ");
			sub = e.substr(0,pos);
			e = e.substr(pos+1,e.length());
			if (!encoding->eof())
				tempEnc.push_back(stold(sub));
		}
		_encoding.push_back(tempEnc);
		pos = 0;
		tempEnc.clear();
	}

	//create structure and add initial weights.
	while(getline(*structure,s)){
		
		for(int i = 0; i < stoi(s); ++i) {
			node = new Node();
			node->addWeight(0.01);
			if(!structure->eof()) {
				getline(*weights,w);
				while(pos != -1) {
					pos = w.find(" ");
					sub = w.substr(0,pos);
					w = w.substr(pos+1,w.length());
					// cout << pos << ":" << w.length() << endl;
					// cout << sub << endl;
					if ((pos+1 >! w.length() && pos != 0) || sub.length()>2)  {
						node->addWeight(stold(sub));
					}
				}
				// cout << endl;
				pos = 0;
			}
			tempStructure.push_back(node);
		}
		_nodeNet.push_back(tempStructure);
		tempStructure.clear();
	}
	return 0;
}

void ann::classify(ifstream* test_input) {
	vector<long double> a;
	uint numInputs = _nodeNet[0].size();
	a = fileToLDVector(test_input,numInputs);

	for(uint i = 0; i < a.size(); i=(i+numInputs)) {
		//step One
		for (uint j = 0; j < numInputs; ++j){
			_nodeNet[0][j]->set_a(a[i+j]);
		}
		stepTwoThree();
		_classifiedVals.push_back(eucDist());
	}
	for(vector<int>::iterator it = _classifiedVals.begin(); it != _classifiedVals.end(); ++it) {
		cout << *it << endl;
	}
	printAccuracy();
}

void ann::back_propagation(ifstream* train_input, ifstream* train_output,int iterations, long double alpha) {
	string input;
	const int outputLayer = _nodeNet.size()-1;
	long double sum = 0;
	vector<long double> in,out;
	vector<vector<long double> > delta;
	vector<long double> subDelta;
	int y;
	int numNodes = 0;
	const uint xInputs = _nodeNet[0].size();
	const uint yOutputs = _nodeNet[outputLayer].size();

	for(uint i = 0; i < _nodeNet.size(); ++i) {
		vector<long double> subDelta;
		numNodes += _nodeNet[i].size();

		delta.push_back(subDelta);
	}

	in = fileToLDVector(train_input,xInputs);
	out = fileToLDVector(train_output,1);

	for(int m = 0; m < iterations; ++m) {
		for(uint n = 0; n < in.size(); n = (n+xInputs)){
			//step One
			for (uint i = 0; i < xInputs; ++i){
				_nodeNet[0][i]->set_a(in[n+i]);
				// cout << "Network[0][" << n+i << "]: " << in[n+i] << endl;
			}
			if(n == 0) {
				y = out[0];
			} else {
				y = out[n/xInputs];
			} 
			// cout << endl << "Step Two/Three" << endl << "------------------" << endl << endl;
			//step two/three
			stepTwoThree();

			// cout << endl << "Step Four" << endl << "------------------" << endl << endl;

			//step four
			for(uint i = 0; i < yOutputs; ++i) {
				long double a = _nodeNet[outputLayer][i]->get_a();

				delta[outputLayer].push_back((a*(1-a)*(_encoding[y][i]-a)));
				// cout << "a[" << i << "] = " << a << endl;
				// cout << "delta[" << outputLayer << "][" << i << "] = " << (a*(1-a)*(_encoding[y][i]-a)) << endl;
			}

			// cout << endl << "Step Five/Six" << endl << "------------------" << endl << endl;
			//step five/six
			sum = 0;

			for(int i = 1; i < outputLayer; ++i) {
				for(uint j = 0; j < _nodeNet[outputLayer-i].size(); ++j) {
					long double a = _nodeNet[outputLayer-i][j]->get_a();
					for(uint k = 0; k < _nodeNet[outputLayer-i+1].size(); ++k) {
						//cout << "k: " << k << endl;
						sum += delta[outputLayer-i+1][k]*_nodeNet[outputLayer-i][j]->get_weight(k+1);
					}
					// cout << "delta[" << i << "][" << j << "] ="  <<  a*(1-a)*sum << endl;
					delta[outputLayer-i].push_back(a*(1-a)*sum);
					sum = 0;
				}
			}		

			// cout << endl << "Step Seven" << endl << "------------------" << endl << endl;
			//step seven
			long double a,d;
			for(int i = 0; i < outputLayer; ++i) {
				for (uint j = 0; j < _nodeNet[i].size(); ++j) {
					a = _nodeNet[i][j]->get_a();
					for (uint k = 0; k < _nodeNet[i+1].size(); ++k) {
						d = _nodeNet[i][j]->get_weight(k+1)+alpha*a*delta[i+1][k];
						_nodeNet[i][j]->setWeight(d,k+1);
						// cout << "W[" << i << "][" << j << "][" << k << "] = " << _nodeNet[i][j]->get_weight(k+1) << endl;
					}
				}
			}

			for (uint i = 1; i < _nodeNet.size(); ++i){
				for (uint j = 0; j < _nodeNet[i].size(); ++j){
					long double w = _nodeNet[i][j]->get_weight(0)+alpha*delta[i][j];
					// cout << "W[" << i << "][0] = " << w << endl;
					_nodeNet[i][j]->setWeight(w,0);
				}
			}
			for (uint i = 0; i < delta.size(); ++i) {
				delta[i].clear();
			}
		}
	}

	// cout << endl << endl;
}

void ann::printAccuracy() {
	long double accuracy;
	int success = 0;
	uint length = _actualVals.size();

	for (uint i = 0; i < length; ++i) {
		if(_actualVals[i]== _classifiedVals[i]){
			++success;
		}
	}
	accuracy = (success)/double(length);
	cout << showpoint << fixed << setprecision(12) << accuracy << endl;
}

void ann::stepTwoThree() {
	long double sum = 0;
	

	for (uint layer = 1; layer < _nodeNet.size(); ++layer) {
		for (uint i = 0; i < _nodeNet[layer].size(); ++i) {
			sum += _nodeNet[layer][i]->get_weight(0);
			for(uint j = 0; j < _nodeNet[layer-1].size(); ++j) {
				sum += _nodeNet[layer-1][j]->get_weight(i+1)*_nodeNet[layer-1][j]->get_a();
			}
			_nodeNet[layer][i]->set_a(calc_a(sum));
			// cout << "Network[" << layer << "][";
			// cout  << i << "]: " << _nodeNet[layer][i]->get_a() << endl;
			sum = 0;
		}
	}
}

int ann::eucDist(){
	uint outerLayer = _nodeNet.size()-1;
	long double v = 0;
	long double sq;
	long double lowest=100;
	long double a[_nodeNet[outerLayer].size()];
	int lowestDigitVal;

	for(uint i = 0; i < _nodeNet[outerLayer].size(); ++i) {
		a[i] = _nodeNet[outerLayer][i]->get_a();
	}
	for(int i = 0; i < 10; ++i) {
		for(uint j = 0; j < _nodeNet[outerLayer].size(); ++j) {
			v += pow((_encoding[i][j]-a[j]),2);
		}
		sq = sqrt(v);
		v = 0;
		if(sq < lowest){
			lowest = sq;
			lowestDigitVal = i;
		}

	}
	return lowestDigitVal;
}
//(_nodeNet[layer2][i-nodesTillNextLayer]->get_weight()+alpha*a*delta[j]),j)

vector<long double> ann::fileToLDVector(ifstream* f,int dataPerLine) {
	string s,sub;
	int i,pos;
	vector<long double> r;

	while(getline(*f,s)) {
		i = 0;
		while(pos != -1) {
			pos = s.find(" ");
			sub = s.substr(0,pos);
			s = s.substr(pos+1,s.length());
			if (i < dataPerLine){
				r.push_back(stold(sub));
				++i;
			}
		}
		pos = 0;
	}
	return r;
}

vector<int> ann::fileToIntVector(ifstream* f,int dataPerLine) {
	string s,sub;
	int i,pos;
	vector<int> r;
	while(getline(*f,s)) {
		i =0;
		while(pos != -1) {
			pos = s.find(" ");
			sub = s.substr(0,pos);
			s = s.substr(pos+1,s.length());
			if (i < dataPerLine){
				r.push_back(stoi(sub));
				++i;
			}
		}
		pos = 0;
	}
	return r;
}

long double ann::calc_a(long double sum) {
	long double res;
	res = (1/(1+exp(-1*sum)));
	return res;
}

void ann::printStructure() {
	for(uint i = 0; i < _nodeNet.size(); ++i) {
		cout << _nodeNet[i].size();
		cout << endl;
	}
	cout << endl;
}

void ann::printWeights() {
	for(uint i = 0; i < _nodeNet.size(); ++i) {
		for(uint j = 0; j < _nodeNet[i].size(); ++j) {
			_nodeNet[i][j]->printWeights();
			cout << endl;
		}

	}
	cout << endl;
}

void ann::printFirstNodeWeights() {
	for (uint i = 0; i < _nodeNet[1].size(); ++i) {
		cout << showpoint << fixed << setprecision(12) << _nodeNet[0][0]->get_weight(i+1) << " ";	
	}
	cout << endl;
}

void ann::printEncoding() {
	for (uint i = 0; i < _encoding.size(); ++i) {
		for (uint j = 0; j < _encoding[i].size(); ++j) {
			cout << _encoding[i][j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

//Node functions
void ann::Node::set_a(long double a) {
	_a = a;
}

long double ann::Node::get_a() {
	return _a;
}

void ann::Node::addWeight(long double weight) {
	_weights.push_back(weight);
}

void ann::Node::setWeight(long double weight,int index) {
	_weights[index] = weight;
}

long double ann::Node::get_weight(int index) {
	return _weights[index];
}

void ann::Node::printWeights() {
	for(vector<long double>::iterator it = _weights.begin(); it != _weights.end(); ++it) {
		cout << showpoint << fixed << setprecision(12) << *it << " ";
	}
}