from __future__ import print_function
import dionysus as d
import numpy as np
import csv

from collections import defaultdict

# Returns enhanced diagram from a filtration
# dgm - a dictionary indexed by dimension
# Note:
#	if death time is infinite - then the second pairing is meaningless
# 	each dictionarry is a list of 4-tuples
#		(birth time, death time, birth vertex, death vertex)
# Input: m is the output of homology_persistence(f),
#        f is a filtration
#        Tbl is a dictionary from simplex to vertex (depends on the function)
def returndgm(m,f,Tbl):
	dgm = {}
	for i in range(len(m)-1):
		dgm[i] = []
    # Tbl is a dictionary from simplex to vertex (depends on the function) / from simplex to two points (three points cause includes value)
    # b_value_i, d_value_i, b_e_b (or -1), b_e_d (or -1), d_e_b (or -1), d_e_d (or -1)
	#point_index_to_points_indices = defaultdict(lambda: [])
	for i in range(len(m)-1):
		if m.pair(i) < i: continue      # skip negative simplices
		dim = int(f[i].dimension())
		pair = m.pair(i) # now an edge
		if pair != m.unpaired and f[pair].data-f[i].data > 0:
			b_a, d_a = Tbl[i], Tbl[pair] # FROM TABLE!
			dgm[dim].append( [f[i].data, f[pair].data, b_a[0], b_a[1], d_a[0], d_a[1]] )
		elif m.pair(i) == m.unpaired:
			b_a = Tbl[i]
			dgm[dim].append([f[i].data, np.inf, b_a[0], b_a[1], -1, -1])
	return dgm #, point_index_to_points_indices

# currently the fastest update procedure
# takes in a filtration F and a vector fnew of new
# function values defined on the vertices
# TODO: check if we update Tbl is it faster
#       than creating it from scratch each time
''' OBS: Don't think it's working for homology classes of dimension 2 '''
def computePersistence(F):
	# update filtration
	# TODO: F and fnew doesn't have to have the same number of decimals!!!!!
	Tbl = {} # From simplex to attaching edge [a,b,value]
	#round_decimals = 10
	# SETTING IT TO s.data changes the float, must be somthing with the filtration structure
	F.sort() # so we know we are visting lover dimension before higher dimensions
	for s in F:
		index_s = F.index(s)
		if s.dimension() == 0:
			Tbl[index_s] = [-1, -1, 0]
		elif s.dimension() == 1:
			max_edge = [s[0], s[1], s.data]
			Tbl[index_s] = max_edge
		elif s.dimension() == 2: # s is now a triangle
			edges = [F.index(b) for b in s.boundary()]
			edges_data = [F[e].data for e in edges]
			edge_index = np.argmax(np.array([edges_data]))
			attaching = [F[edges[edge_index]][0], F[edges[edge_index]][1], edges_data[edge_index]]
			Tbl[index_s] = attaching
		elif s.dimension() == 3: # s is now a tetrahedron
			edges_triples = [Tbl[F.index(b)] for b in s.boundary()]
			edges_data = [e[2] for e in edges_triples]
			edge_index = np.argmax(np.array([edges_data]))
			attaching = edges_triples[edge_index]
			Tbl[index_s] = attaching
		else:
			print("WARNING-topologicalutils: something is wrong. Too high dimensional")
			assert(False)

	m = d.homology_persistence(F) # compute persistence
	dgms = returndgm(m,F,Tbl) # compute diagrams
	return dgms, Tbl  # return
