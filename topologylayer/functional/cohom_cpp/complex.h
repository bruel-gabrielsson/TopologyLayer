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

class SimplicialComplex{
  public:

    // complex is stored as list of cells
    std::vector<std::vector<int>> cells;
    // maybe hold filtration in class
    // std::vector<float> filtration;

    // appends simplex to complex
    void append(std::vector<int> &x);

    // prints the list of simplices
    void print();

    // extend a filtration on vertices to whole complex
    // template <typename T>
    // std::vector<T> extend_lower_star(std::vector<T>);

    // extend a filtration on edges to whole complex
    // template <typename T>
    // std::vector<T> extend_flag(std::vector<T>);

};

#endif
