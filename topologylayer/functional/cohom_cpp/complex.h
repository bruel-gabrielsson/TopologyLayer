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
#include <torch/extension.h>
#include <vector>
#include <cstddef>
#include "cocycle.h"

// TODO: template over filtration type
// TODO: different types for different complexes/ filtrations
class SimplicialComplex{
  public:

    // complex is stored as list of cells
    std::vector<std::vector<int>> cells;
    // number of cells in each dimension
    std::vector<int> ncells;

    // holds the permutation on cells that gives filtration order
    // j = filtration_perm[i] means cell j is in the i'th location in filtration.
    std::vector<size_t> filtration_perm;

    // holds inverse permutation on cells for filtration order
    // i = inv_filtraiton_perm[j] means cell j is in i'th locaiton in filtration.
    std::vector<size_t> inv_filtration_perm;

    // holds map to critical cell
    // function_map[j] = [...] face that causes cell j to appear
    std::vector<std::vector<int>> function_map;

    // holds filtration information
    std::vector<std::pair<float, int>> full_function;

    /*
    for backpropagation lookup
    holds the critical simplex for each birth and death
    for levelset, will be vector of vectors with 1 entry each
    for rips, will be vector of vectors with 2 entries each
    for cech, will be vector of vectors with variable entries
    backprop_lookup[k][j][0] is critical simplex of birth of bar j in dim k
    backprop_lookup[k][j][1] is critical simplex of death of bar j in dim k
    */
    std::vector<std::vector<std::vector<int>>> backprop_lookup;

    // hold boundary matrix of complex
    std::vector<Cocycle> bdr;

    // appends simplex to complex
    void append(std::vector<int> &x);

    // prints the list of simplices
    void printComplex();

    // prints filtration
    void printFiltration();

    // prints function map
    void printFunctionMap();

    // prints boundary matrix
    void printBoundary();

    // print number of cells in each dim
    void printDims();

    // get number of pairs for homology in dimension dim
    int numPairs(int dim);

    // print critical indices
    void printCritInds();

    // pre-allocate vectors once cell list has been completed.
    void initialize();

    // extend filtration on 0-cells to filtration on all cells
    // template <typename T>
    // void extend(std::vector<T> &f);
    void extend(torch::Tensor f);

    // extend filtration on 1-cells to filtration on all cells
    void extend_flag(torch::Tensor f);

    // fill in filtration order
    void sortedOrder();

    // get dimension of cell j
    size_t dim(size_t j);

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
