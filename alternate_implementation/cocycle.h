#ifndef _COCYCLE_H
#define _COCYCLE_H



class Cocycle{
	public:
		int index;
		std::set<int> cochain;

		// we should never have this
		Cocycle() : index(-1){}
		
		// initializations
		Cocycle(int x) : index(x) {cochain.insert(x); }
		Cocycle(int x, std::set<int> y) :  index(x) , cochain(y) {}

		// for debug purposes
		void insert(int x);
	
		// add two cocycles over Z_2	
		void add(const Cocycle &x);

		// dot product of two cocycles
		int multiply(const Cocycle &x) const;

		// dimension - number of nonzero entries -1
		int dim() const;
		
		// debug function
		void print();

};


// -------------------------------------------------------
// Implementation - should be moved to cpp file 
// but its quite short at this point
// -------------------------------------------------------



void Cocycle::insert(int x){
	cochain.insert(x);
}
		

void Cocycle::add(const Cocycle &x){
	std::set<int> tmp;
	std::set_symmetric_difference(x.cochain.begin(), x.cochain.end(),cochain.begin(),cochain.end(), std::inserter(tmp,tmp.begin()));
	cochain = tmp;
}


int  Cocycle::multiply(const Cocycle &x) const{
	std::set<int> tmp;
	std::set_intersection(x.cochain.begin(), x.cochain.end(),  cochain.begin(),cochain.end(),std::inserter(tmp,tmp.begin()));
	return tmp.size()%2;
}

int Cocycle::dim() const{
	return (cochain.size()==0) ? 0 : cochain.size()-1; 
}

void Cocycle::print(){
	std::cout<<index<<" |  ";
	for(auto  s: cochain){
 		std::cout<<s<<"  ";
	}	
	std::cout<<std::endl;
}




#endif

