#include "cocycle.h"
#include <vector>

#include <torch/extension.h>
namespace py = pybind11;


void Cocycle::insert(int x){
	cochain.push_back(x);
}

// add x to cocycle
// IMPORTANT: this function assumes that cocycles are sorted!
void Cocycle::add(const Cocycle &x){
	// quick check to see if there is anything to do
	if (x.cochain.size() == 0) {return;}
	if (cochain.size() == 0) {cochain = x.cochain; return;}

	// now we know there is something non-trivial to do
	std::vector<int> tmp;
	size_t i1 = 0;
	size_t i2 = 0;
	do {
		size_t v1 = cochain[i1];
		size_t v2 = x.cochain[i2];
		if (v1 == v2) {
			// F2 means sum is 0
			i1++;
			i2++;
		} else if (v1 < v2) {
			tmp.push_back(v1);
			i1++;
		} else { // v2 < v1
			tmp.push_back(v2);
			i2++;
		}
	} while (i1 < cochain.size() && i2 < x.cochain.size());
	// run through rest of entries and dump in
	// only one of the loops will actually do anything
	while (i1 < cochain.size()) {
		tmp.push_back(cochain[i1]);
		i1++;
	}
	while (i2 < x.cochain.size()) {
		tmp.push_back(x.cochain[i2]);
		i2++;
	}
	cochain = tmp;
	return;
}

// take dot product with cocycle
// IMPORTANT: this function assumes that cocycles are sorted!
int  Cocycle::dot(const Cocycle &x) const{
	// inner product
	// quick check to see if anything to be done
	if (cochain.size() == 0 || x.cochain.size() == 0) return 0;
	// loop over indices to compute size of intersection
	size_t i1 = 0;
	size_t i2 = 0;
	size_t intersection = 0;
	do {
		auto v1 = cochain[i1];
		auto v2 = x.cochain[i2];
		if (v1 == v2) {
			i1++;
			i2++;
			intersection++;
		} else if (v1 < v2) {
			i1++;
		} else { // v2 < v1
			i2++;
		}
	} while (i1 < cochain.size() && i2 < x.cochain.size());
	// std::set<int> tmp;
	// std::set_intersection(x.cochain.begin(), x.cochain.end(),  cochain.begin(),cochain.end(),std::inserter(tmp,tmp.begin()));
	// return tmp.size()%2;
	return intersection % 2;
}

int Cocycle::dim() const{
	return (cochain.size()==0) ? 0 : cochain.size()-1;
}

void Cocycle::print(){
	py::print(index, " : ", cochain);
}
