import cohomopt as chopt

p  = chopt.Cohomology()

p.addSimplex([0])
p.addSimplex([1])
p.addSimplex([2])

p.addSimplex([0,2])
p.addSimplex([1,2])
p.addSimplex([0,1])

p.addSimplex([0,1,2])
print "printC"
p.printComplex()
p.printComplexOrder([0,3, 5,2])
p
p.init()

# the compute function takes a list of function values
# right now its for the simplicial complex
# but everything is in place for a vertex function
p.compute([0,1,2,3,4,5,6])

# while we work out the return type
p.printBars()

# barcode should be 0 inf, 1 4, 2 3, 5 6
