#include "cocycle.h"
#include <vector>

#include <torch/extension.h>
namespace py = pybind11;


void Cocycle::insert(size_t x){
	cochain.push_back(x);
}


void Cocycle::add(const Cocycle &x){
	// std::set<int> tmp;
	// std::set_symmetric_difference(x.cochain.begin(), x.cochain.end(),cochain.begin(),cochain.end(), std::inserter(tmp,tmp.begin()));
	// cochain = tmp;
}


int  Cocycle::dot(const Cocycle &x) const{
	// inner product
	// std::set<int> tmp;
	// std::set_intersection(x.cochain.begin(), x.cochain.end(),  cochain.begin(),cochain.end(),std::inserter(tmp,tmp.begin()));
	// return tmp.size()%2;
	return 0;
}

int Cocycle::dim() const{
	return (cochain.size()==0) ? 0 : cochain.size()-1;
}

void Cocycle::print(){
	py::print(index, " : ", cochain);
}
