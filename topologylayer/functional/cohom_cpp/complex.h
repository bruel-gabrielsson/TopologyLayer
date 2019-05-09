#ifndef _COMPLEX_H
#define _COMPLEX_H
/*
complex.h

class to hold simplicial complex

must be able to:
* generate boundary matrix (F2)
* add simplices
* sort by filtration order
*/
#include <vector>
#include "cocycle.h"

// TODO: template over filtration type
// TODO: different types for different complexes/ filtrations
class SimplicialComplex{
  public:

    // complex is stored as list of cells
    std::vector<std::vector<size_t>> cells;
    // maybe hold filtration in class
    // std::vector<float> filtration;

    // discrete time filtration on cells
    std::vector<size_t> filtration;

    // holds the permutation on cells that gives filtration order
    std::vector<size_t> filtration_perm;

    // holds map to critical cell
    std::vector<size_t> function_map;
    // holds filtration information
    std::vector<std::pair<double, int>> full_function;

    // for backpropagation lookup
    // holds the critical simplex for each cell
    // for levelset, will be vector of vectors with 1 entry each
    // for rips, will be vector of vectors with 2 entries each
    // for cech, will be vector of vectors with variable entries
    std::vector<std::vector<size_t>> backprop_lookup;

    // hold boundary matrix of complex
    std::vector<Cocycle> bdr;

    // appends simplex to complex
    void append(std::vector<size_t> &x);

    // prints the list of simplices
    void printComplex();

    // prints filtration
    void printFiltration();

    // prints boundary matrix
    void printBoundary();

    // pre-allocate vectors once cell list has been completed.
    void initialize();

    // extend filtration on 0-cells to filtration on all cells
    // TODO: figure out how to use template with PyBind...
    // template <typename T>
    // void extend(std::vector<T> &f);
    void extend(std::vector<double> &f);

    // fill in filtration order
    void sortedOrder();

    // extend a filtration on vertices to whole complex
    // template <typename T>
    // std::vector<T> extend_lower_star(std::vector<T>);

    // extend a filtration on edges to whole complex
    // template <typename T>
    // std::vector<T> extend_flag(std::vector<T>);

    //--------------------------------
    // (Primoz) Note: Rather than extend lower and flag
    //  why about if we have an extend which takes
    //  some input (maybe passed by const reference)
    //-------------------------------
    //  template <typename T>
    //  void extend(std::vector<T>&);
    //   ------------------------------------
    // ==========================
    //   Some more thoughts:
    //    - it doesnt need to be float - this is the type that will index
    //   template<typename T>
    //   std::vector<T> filtration;
    //
    //  extend should initialize a lookup table which specifies what points we care about for a given point.
    //   for lower star filtrations - reverse lookup from simplex to 1 vertex - but for other things
    //   we may need multiple vertices  -  eg. rips - 2 vertices or cech - k vertices.
    //
    //   std::vector<std::vector<size_t>> backprop_lookup;


};

#endif
