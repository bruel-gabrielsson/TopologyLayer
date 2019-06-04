#include "cocycle.h"
#include "sparsevec.h"
#include <vector>

#include <torch/extension.h>
namespace py = pybind11;


void Cocycle::insert(int x){
	cochain.insert(x);
}

// add x to cocycle
// IMPORTANT: this function assumes that cocycles are sorted!
void Cocycle::add(const Cocycle &x){
	cochain.add(x.cochain);
	return;
}

// take dot product with cocycle
// IMPORTANT: this function assumes that cocycles are sorted!
int  Cocycle::dot(const Cocycle &x) const{
	return cochain.dot(x.cochain);
}

int Cocycle::dim() const{
	return (cochain.nzinds.size()==0) ? 0 : cochain.nzinds.size()-1;
}

void Cocycle::print(){
	py::print(index, " : ");
	cochain.print();
}
