#ifndef __COHOMOLOGY__H
#define __COHOMOLOGY__H

#include<iostream>
#include<fstream>
#include<set>
#include<vector>
#include<map>
#include<algorithm>
#include<numeric>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cocycle.h"
#include "interval.h"



class Cohomology{
	public:

		typedef std::map<int,Interval> Barcode;

		std::vector<Cocycle> Z;
		std::vector<Cocycle> bdr;

		Barcode partial_diagram;

		std::map<int,std::vector<Interval>> persistence_diagram;

		std::vector<int> function_map;
		std::vector<std::pair<double,int>> full_function;


		// this is mainly for debugging purposes
		std::vector<std::set<int>> complex;


		// this is for testing purposes - we will never build things one by one
		void addSimplex(const std::vector<int> &x);


		// for testing purposes we will make as input a simplicial complex - a vector of sets
		// if we make this an initializer - we can make bdr consta

		void initComplex();

		// the input here will be changed toa tensor
		// this is again a testing version
		void computeFiltration(const std::vector<double> &f);


		// this function should be made private
               	void step(int i);


		// output functions

		// -------------------------------------------------------
		// output functions
		//
		// return tensor?
		std::map<int,std::vector<std::vector<double>>> returnBars();

		// take 2
		pybind11::dict barcode();


		void printBars();


		// debug functions
		void printComplex();
		void printComplexOrder(const std::vector<int>&);
		void printFunction();



		//main compute
		std::vector<std::vector<std::pair<int,int>>> computeDiagram(std::vector<double> &f,int maxdim){
			std::vector<std::vector<std::pair<int,int>>> bc(maxdim,std::vector<std::pair<int,int>>());
			// determine when simplices appear in filtration order
			extend(f);
			// find sorted order of filtration.  filtration_order is permutation for filtration order
			auto filtration_order = sortedOrder();

			for(int i : filtration_order ){
			//	std::cout<<i<<"  "<< full_function[i].first << "  "<<full_function[i].second<<std::endl;
				// take inner product, add or reduce active cocycle matrix as necesary
				step(i);
			}

			// put together barcode
			// this is the part where the good index scheme would pay off
			for(auto it = partial_diagram.begin(); it!=partial_diagram.end(); ++it){
				int  bindx = it->first;
				auto I = it->second;
				if(I.death_index==-1){
				//	std::cout<<full_function[bindx].second<<"  "<<full_function[I.birth_index].first<<"  "<<I.death_index<<"  | "<<function_map[I.birth_index]<<" -1 "<<std::endl;
					bc[full_function[bindx].second].push_back(std::pair<int,int>(function_map[I.birth_index], -1));
				}
				else
				//	std::cout<<full_function[bindx].second<<"  "<<full_function[I.birth_index].first<<"  "<<full_function[I.death_index].first<<"  | "<<function_map[I.birth_index]<<"  "<<function_map[I.death_index]<<std::endl;
bc[full_function[bindx].second].push_back(std::pair<int,int>(function_map[I.birth_index], function_map[I.death_index]));

			}
			return bc;
		}

		// lets do OFF file first
		void readInOFF(std::string filename);

		// preprocessing functions - to be made private
		void extend(const std::vector<double> &f);
		std::vector<int> sortedOrder();


		// void computeNew(const std::vector<double> &f){
		//
		// 	std::vector<std::pair<double,int> >  simplicial_function = extend(f);
		//
		//
		// 	// check that vector is of the correct size
		// 	// get index sort
		// 	std::vector<int> idx =  sortedOrder(simplicial_function);
		// 	// computation step
		// 	for(auto x : idx ) {
		// 		step(x);
		// 	}
		//
		// 	// fill in barcode - this will be changed to a tensor
		// 	// here we also give vertex indicies
		// 	for(auto it = partial_diagram.begin(); it!=partial_diagram.end(); ++it){
		// 		int  bindx = it->first;
		// 		auto I = it->second;
		// 		if(I.death_index==-1){
		// 			persistence_diagram[bdr[bindx].dim()].emplace_back(Interval(function_map[I.birth_index], I.death_index, f[I.birth_index],-1));
		//
		// 		}
		// 		else{
		// 			persistence_diagram[bdr[bindx].dim()].emplace_back(Interval(function_map[I.birth_index], function_map[I.death_index], f[I.birth_index],f[I.death_index]));
		// 		}
		//
		// 	}
		//
		//
		// }

		// // TODO check this is correct
		// std::vector<double> extend(const std::vector<double> &x){
		// 	const size_t N(complex.size());
		// 	std::vector<double> fullfunc(N);
		// 	for(size_t i = 0 ; i<N;++i){
		// 		int element = *std::max_element(complex[i].begin(),complex[i].end(),[&x](int i1, int i2){return x[i1]<x[i2];});
		// 		fullfunc[i]  = x[element];
		// 		function_map[i] = element;
		// 	}
		// 	return fullfunc;
		// }



		// stable sorting - there is one trick we need to take care of -
		// since we do the extension - we need to make sure the
		// returned order is valid
		std::vector<int> sortedOrder(const std::vector<std::pair<double,int> > &f) const{
			std::vector<int> idx(f.size());
		     	std::iota(idx.begin(), idx.end(), 0);

	        	// sort indexes based on comparing values in x - take into account dimension
    			std::sort(idx.begin(), idx.end(),
				[&f](int i1, int i2) {return (f[i1].first==f[i2].first) ? f[i1].second < f[i2].second : f[i1].first < f[i2].first;});

			return idx;
		}

		// // reworked extension function
		// std::vector<std::pair<double,int> > extend(const std::vector<double> &f ){
		// 	const size_t N(complex.size());
		// 	std::vector<std::<double,int>> fullfunc(N);
		// 	for(size_t i = 0 ; i<N;++i){
		// 		fullfunc[i]  =  std::pair<double,int>(x[*std::max_element(complex[i].begin(),complex[i].end(),[&x](int i1, int i2){return x[i1]<x[i2];})], bdr[i].dim());
		// 	}
		// 	return fullfunc;
		//
		// }

		// data structure for simplex -> vertex
		int returnMap(int i){
			return function_map[i];
		}




		// // lets do OFF file first
		// void readInOFF2D(std::string filename){
		// 	std::ifstream input(filename.c_str());
		// 	std::string buf;
		// 	std::getline(input,buf); // first line
		// 	int vertex_size, edge_size, face_size;
		// 	input>>vertex_size>>edge_size>>face_size;
		// 	// we dont care about the geometry
		// 	for(int i=0;i<vertex_size;++i){
		// 		std::getline(input,buf); // first line
		// 	}
		// 	int dim, v;
		// 	//insert top dimensional simplices
		// 	for(int i = 0; i<face_size;++i){
		// 		input>>dim;
		// 		std::set<int> simplex;
		// 		for(int j=0;j<dim;++j){
		// 			input>>v;
		// 			simplex.insert(v);
		// 		}
		// 		complex.push_back(simplex);
		//
		// 	}
		//
		// 	input.close();
		// }

		// void initFull(){
		// 	int maxdim  = 0;
		// 	int indx = 0;
		// 	std::map<std::set<int>,int> reverse_map;
		//
		// 	for(auto s : complex){
		// 		reverse_map[s] = indx++;
		// 		maxdim = (maxdim < s.size()-1) ? s.size()-1 : maxdim;
		// 	}
		// 	// this may be overkill
		// 	Z.reserve(indx);
		//
		// 	// this probably not
		// 	bdr.reserve(indx);
		// 	function_map.reserve(complex.size());
		//
		// 	// initialize barcode
		// 	for (int i = 0 ; i<=maxdim; ++i){
		// 		persistence_diagram[i];
		// 	}
		//
		// 	//initialize boundary
		// 	for (auto s: complex){
		// 		std::set<int> tmp;
		// 		std::set<int> s_copy;
		// 		if(s.size()>1){
		// 			for(int v :  s){
		// 				s_copy = s;
		// 				s_copy.erase(v);
		// 				tmp.insert(reverse_map[s_copy]);
		// 			}
		//
		// 		}
		//
		// 		bdr.emplace_back(Cocycle(reverse_map[s],tmp));
		// 	}
		//
		// }
	};


#include "cohomology.hpp"
#endif
