#ifndef _COCYCLE_H
#define _COCYCLE_H

#include <vector>
#include <cstddef>
#include "sparsevec.h"


class Cocycle{
	public:
		// birth index
		size_t index;

		// non-zero entries
		// IMPORTANT: this is assumed to always be sorted!
		SparseF2Vec<int> cochain;

		// we should never have this
		Cocycle() : index(-1){}

		// initializations
		Cocycle(size_t x) : index(x) , cochain((int) x) {}
		Cocycle(size_t x, std::vector<int> y) :  index(x) , cochain(y) {}

		// for debug purposes
		void insert(int x);

		// add two cocycles over Z_2
		void add(const Cocycle &x);

		// dot product of two cocycles
		int dot(const Cocycle &x) const;

		// dimension - number of nonzero entries -1
		int dim() const;

		// debug function
		void print();

};



#endif
