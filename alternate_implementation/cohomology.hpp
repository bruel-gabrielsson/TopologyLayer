

// this is for testing purposes - we will never build things one by one
void Cohomology::addSimplex(const std::vector<int> &x){
	std::set<int> tmp;
	for(auto v : x){
		tmp.insert(v);
	}
	complex.push_back(tmp);
}



// call this after complex is internally put in somehow
void Cohomology::initComplex(){
	int maxdim  = 0;
	int indx = 0;
	std::map<std::set<int>,int> reverse_map;
//	std::cout<<"Complex size= "<<complex.size()<<std::endl;
	for(auto s : complex){
		reverse_map[s] = indx++;
		maxdim = (maxdim < s.size()-1) ? s.size()-1 : maxdim;
	}
	
	// this may be overkill
	Z.reserve(indx);

	// this probably not
	bdr.reserve(indx);
	function_map.reserve(complex.size());
	function_map = std::vector<int>(complex.size(),-1);
	

	full_function.reserve(complex.size());
	full_function = std::vector<std::pair<double,int>>(complex.size(),std::pair<double,int>(0.0,0));	
	

	// initialize barcode
	for (int i = 0 ; i<=maxdim; ++i){
		persistence_diagram[i];
	}

	//initialize boundary
	for (auto s: complex){
		std::set<int> tmp;
		std::set<int> s_copy;
		if(s.size()>1){
			for(int v :  s){
				s_copy = s;
				s_copy.erase(v);
				tmp.insert(reverse_map[s_copy]);
			}
				
		}

		bdr.emplace_back(Cocycle(reverse_map[s],tmp));
	}
		
}


void Cohomology::step(int i){
	bool flag = false;
	auto pivot = Z.rbegin();
	for(auto x  = Z.rbegin(); x != Z.rend();  ++x){
		if(x->multiply(bdr[i])){
			if(flag==false){
				pivot = x;
				flag=true;	
			}
			else{
				x->add(*pivot); 
			}
		}
	}

	if(flag){
		// add assertion that it exists
		partial_diagram[pivot->index].close(bdr[i].index);
		// stupid translation from reverse to iterator
		Z.erase(std::next(pivot).base());
	}
	else{
		partial_diagram[i] = Interval(i);
		Z.emplace_back(Cocycle(i));
	}
}



// 
//   Output functions
//

std::map<int,std::vector<std::vector<double>>> Cohomology::returnBars(){
			std::map<int,std::vector<std::vector<double>>> tmp;

			for(auto x : persistence_diagram){
				std::vector<std::vector<double>> dimbars;
				for(auto y : x.second){
					std::vector<double> rr;
					rr.push_back(y.birth);
					rr.push_back(y.death);
					dimbars.push_back(rr);
				}
				tmp[x.first] = dimbars;
			}
			return tmp;
		}

// take 2
pybind11::dict Cohomology::barcode(){
	pybind11::dict  bc;
			
	for(auto x : persistence_diagram){
		std::vector<std::vector<double>> dimbars;
		for(auto y : x.second){
			std::vector<double> rr;
			rr.push_back(y.birth);
			rr.push_back(y.death);
			dimbars.push_back(rr);
		}
		bc["x.first"] = dimbars; 
	}
	return bc;
}


void Cohomology::printBars(){	
	for(auto x : persistence_diagram){
		std::cout<<x.first<<" = "<<std::endl;
		for(auto i : x.second){
			std::cout<<i.birth<<"  "<<i.death<<std::endl;
		}
	}
}




// the input here will be changed toa tensor
// this is again a testing version
// it assumes the function given is on the full complex 
void Cohomology::computeFiltration(const std::vector<double> &f){
	// check that vector is of the correct size
	// get index sort
	std::vector<int> idx(f.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in x
    	std::sort(idx.begin(), idx.end(),
		[&f](int i1, int i2) {return f[i1] < f[i2];});
	
			
	// computation step
	for(auto x : idx ) {
		step(x);
	}
	
	// fill in barcode - this will be changed to a tensor
	for(auto it = partial_diagram.begin(); it!=partial_diagram.end(); ++it){
		int  bindx = it->first;
		auto I = it->second;
		if(I.death_index==-1){
			persistence_diagram[bdr[bindx].dim()].emplace_back(Interval(I.birth_index, I.death_index, f[I.birth_index],-1));
		}
		else{
			persistence_diagram[bdr[bindx].dim()].emplace_back(Interval(I.birth_index, I.death_index, f[I.birth_index],f[I.death_index]));
		}

	}


}

//
//
// Input files
//
//

void Cohomology::readInOFF(std::string filename){
	std::ifstream input(filename.c_str());
	std::string buf;
	std::getline(input,buf); // first line
	int vertex_size, edge_size, face_size;
	input>>vertex_size>>face_size>>edge_size;
	std::getline(input,buf); 
	// we dont care about the geometry
	// but we assume vertices come first
	for(int i=0;i<vertex_size;++i){
		std::getline(input,buf); // first linei
		std::vector<int> tmp = {i};
		addSimplex(tmp);
	}
	
	int dim, v;

	//insert top dimensional simplices
	for(int i = 0; i<face_size;++i){
		input>>dim;
	//	std::cout<<dim<<std::endl;
		std::vector<int> simplex;
		for(int j=0;j<dim;++j){
			input>>v;
	//		std::cout<<v<<std::endl;
			simplex.push_back(v);	
		}
		addSimplex(simplex);
		// here i deal with two cases only dim = 3 or 4
		if(dim==3){
			// insert all edges 
			addSimplex(std::vector<int>(simplex.begin(),simplex.end()-1));
			addSimplex(std::vector<int>(simplex.begin()+1,simplex.end()));
			simplex.erase(simplex.begin()+1);
			addSimplex(simplex);
	
		}
		if(dim==4){
			// insert all triangles
			std::vector<int> e1 = {simplex[0],simplex[1]};
			std::vector<int> e2 = {simplex[0],simplex[2]};
			std::vector<int> e3 = {simplex[0],simplex[3]};
			std::vector<int> e4 = {simplex[1],simplex[2]};
			std::vector<int> e5 = {simplex[1],simplex[3]};
			std::vector<int> e6 = {simplex[2],simplex[3]};


			std::vector<int> t1 = {simplex[0],simplex[1],simplex[2]};
			std::vector<int> t2 = {simplex[0],simplex[1],simplex[3]};
			std::vector<int> t3 = {simplex[0],simplex[2],simplex[3]};
			std::vector<int> t4 = {simplex[1],simplex[2],simplex[3]};
			
			addSimplex(e1);
			addSimplex(e2);
			addSimplex(e3);
			addSimplex(e4);
			addSimplex(e5);
			addSimplex(e6);
			
			addSimplex(t1);
			addSimplex(t2);
			addSimplex(t3);
			addSimplex(t4);


		}
	}
	
	std::unique(complex.begin(),complex.end());


	input.close();	
}

// stable sorting - there is one trick we need to take care of - 
// since we do the extension - we need to make sure the 
// returned order is valid	
std::vector<int> Cohomology::sortedOrder(){
	std::vector<int> idx(full_function.size());
	std::iota(idx.begin(), idx.end(), 0);

	 // sort indexes based on comparing values in x - take into account dimension
    	std::sort(idx.begin(), idx.end(), [this](int i1, int i2) {return (full_function[i1].first==full_function[i2].first) ? full_function[i1].second < full_function[i2].second : full_function[i1].first < full_function[i2].first;});
	
	return idx;
}


// reworked extension function
// input is on vertices
void Cohomology::extend(const std::vector<double> &f ){
	// move this to initialization
	const size_t N(complex.size());
	for(size_t i = 0 ; i<N;++i){
		int element = *std::max_element(complex[i].begin(),complex[i].end(),[&f](int i1, int i2){return f[i1]<f[i2];});
	//	std::cout<<"extending function to "<<i<<"  "<<element <<std::endl;
		full_function[i] = std::pair<double,int>(f[element], complex[i].size()-1);	
		function_map[i] = element;
	}

}





// 
//
//
// debug functions
//
//
//
//
void Cohomology::printFunction(){
	std::cout<<"Size = "<<full_function.size()<<std::endl;
	for (auto x  : full_function){
		std::cout<<x.first<<"  "<<x.second<<std::endl;
	}
}

void Cohomology::printComplex(){
	for(auto s : complex){
		for(auto x : s){	
			std::cout<<x<<", ";
		}
		std::cout<<std::endl;
	}
}


void Cohomology::printComplexOrder(const std::vector<int> &I){
	for(auto i : I){
		for(auto x : complex[i]){	
			std::cout<<x<<", ";
		}
		std::cout<<std::endl;
	}
}






/*

	// TODO check this is correct
		std::vector<double> extend(const std::vector<double> &x){
			const size_t N(complex.size());
			std::vector<double> fullfunc(N);
			for(size_t i = 0 ; i<N;++i){
				int element = *std::max_element(complex[i].begin(),complex[i].end(),[&x](int i1, int i2){return x[i1]<x[i2];});
				fullfunc[i]  = x[element];
				function_map[i] = element;
			}
			return fullfunc;
		}
		

*/
