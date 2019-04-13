#include<iostream>
#include "cohomology.hpp"




int main(){
	std::cout<<"hello"<<std::endl;

	Cohomology C;

	std::set<int> s;

	std::vector<std::set<int>> cc;
	// test
	s.insert(0);
	cc.push_back(s);
	
	s.clear();
	s.insert(1);
	cc.push_back(s);
	

	s.clear();
	s.insert(2);
	cc.push_back(s);


	s.clear();
	s.insert(1);	
	s.insert(2);
	cc.push_back(s);

	s.clear();
	s.insert(0);	
	s.insert(2);
	cc.push_back(s);

	s.clear();
	s.insert(1);	
	s.insert(0);
	cc.push_back(s);

	s.clear();
	s.insert(0);	
	s.insert(1);	
	s.insert(2);
	cc.push_back(s);

	C.init(cc);
	//
	// check
	//
	
	
	
	std::vector<double> f = {1.0 , 2.0 ,3.0 ,4.0 ,5.0 , 6.0 ,7.0 };
	C.compute(f);
	C.printBars();
}








