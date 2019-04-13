#include<iostream>
#include<set>
#include<vector>
#include<map>
#include<algorithm>
#include<numeric>

class Cocycle{
	public:
		int index;
		std::set<int> cochain;

		// we should never have this
		Cocycle() : index(-1){}
		
		Cocycle(int x) : index(x) {cochain.insert(x); }
		Cocycle(int x, std::set<int> y) :  index(x) , cochain(y) {}

		void insert(int x){
			cochain.insert(x);
		}
		

		void add(const Cocycle &x)
		{
			std::set<int> tmp;
			std::set_symmetric_difference(x.cochain.begin(), x.cochain.end(),cochain.begin(),cochain.end(), std::inserter(tmp,tmp.begin()));
			cochain = tmp;
		}


		int multiply(const Cocycle &x) const{
			std::set<int> tmp;
			std::set_intersection(x.cochain.begin(), x.cochain.end(),  cochain.begin(),cochain.end(),std::inserter(tmp,tmp.begin()));
			return tmp.size()%2;
		}

		int dim() const{
			return cochain.size()-1;
		}
		
		void print(){
		       std::cout<<index<<" |  ";
		       for(auto  s: cochain){
		       		std::cout<<s<<"  ";
		       }	
		       std::cout<<std::endl;
		}

};

class Interval{
	public:
		double birth;
		double death;
		int birth_index;
		int death_index;

		Interval(){}


		Interval(int bindex) : birth_index(bindex), death_index(-1){}
		Interval(int bindex, double btime) : birth_index(bindex), birth(btime){}

		Interval(int bindex, int dindex, double btime, double dtime) : birth_index(bindex), death_index(dindex), birth(btime), death(dtime){}

		
		void close(int dindex){
			death_index = dindex;
		}
		
		void close(int dindex, double dtime){
			death_index = dindex;
			dtime = dtime;
		}
};



class Cohomology{
	public:
		
		typedef std::map<int,Interval> Barcode;

		std::vector<Cocycle> Z;
		std::vector<Cocycle> bdr;
	
		Barcode partial_diagram;

		std::map<int,std::vector<Interval>> persistence_diagram;
		

		// for testing urposes we will make as input a simplicial complex - a vector of sets
		// if we make this an initializer - we can make bdr const
		void init(std::vector<std::set<int> > &complex){
			int maxdim  = 0;
			int indx = 0;
			std::map<std::set<int>,int> reverse_map;

			for(auto s : complex){
				reverse_map[s] = indx++;
				maxdim = (maxdim < s.size()-1) ? s.size()-1 : maxdim;
			}
			// this may be overkill
			Z.reserve(indx);

			// this probably not
			bdr.reserve(indx);


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

		// the input here will be changed toa tensor
		void compute(const std::vector<double> &f){
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

               	void step(int i){
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


		void printBars(){
			for(auto x : persistence_diagram){
				std::cout<<x.first<<" = "<<std::endl;
				for(auto i : x.second){
					std::cout<<i.birth<<"  "<<i.death<<std::endl;
				}
			}
		}
};
