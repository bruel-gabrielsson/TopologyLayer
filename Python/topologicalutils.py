from __future__ import print_function
import dionysus as d
import numpy as np
import csv

# Get boundary for a certain filtration value <= max_f_value, a basis for Dk+1(Ck+1) in Ck
# return (tot_num_simplicies, num_basis_vectors)
def boundary_basis(filtration, max_f_value, dimension):
	f = filtration
	f.sort()
	R = d.homology_persistence(f) # compute persistence
	# dgms = returndgm(m,F,Tbl)
	basis = []
	tot_num_simplicies = len(R)
	#print("boundary_basis")
	for i in range(tot_num_simplicies): # iterate over columns
		if R.pair(i) < i: # This is a boundary column
			birth = f[i].data
			dim = int(f[i].dimension())
			col = R[i]
			if dim == (dimension+1) and birth <= max_f_value:
				assert(len(col) > 0)
				chain = np.zeros(tot_num_simplicies)
				inds = [l.index for l in col]
				chain[inds] = 1
				basis.append(chain)
	basis = np.array(basis).T
	#print basis.shape
	return basis

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
	dgm[0] = []
	dgm[1] = []
	dgm[2] = []
	#dgm[3] = []

	for i in range(len(m)):
		if m.pair(i) < i: continue      # skip negative simplices
		dim = int(f[i].dimension())
		if m.pair(i) != m.unpaired and f[m.pair(i)].data-f[i].data > 0:
			dgm[dim].append([f[i].data, f[m.pair(i)].data, Tbl[f[i]], Tbl[f[m.pair(i)]]] )
		elif m.pair(i) == m.unpaired :
			dgm[dim].append([f[i].data, np.inf, Tbl[f[i]], np.inf] )
	return dgm


# currently the fastest update procedure
# takes in a filtration F and a vector fnew of new
# function values defined on the vertices
# TODO: check if we update Tbl is it faster
#       than creating it from scratch each time
def computePersistence(F,fnew):
	# update filtration
	# TODO: F and fnew doesn't have to have the same number of decimals!!!!!
	Tbl = {}
	# SETTING IT TO s.data changes the float, must be somthing with the filtration structure
	for s in F:
		if s.dimension() == 0:
			s.data = fnew[0,s[0]]
			Tbl[s] = s[0]
		elif s.dimension() == 1:
			max_value = np.max(np.array([ fnew[0,s[0]],fnew[0,s[1]] ]))
			s.data = max_value.astype(np.float64)
			if fnew[0,s[0]] == max_value:
				Tbl[s] = s[0]
			elif fnew[0,s[1]] == max_value:
				Tbl[s] = s[1]
			else:
				print("o", fnew[0,s[0]], fnew[0,s[1]], max_value)
				print(fnew)
				print("WARNING-topologicalutils: something is wrong")
				assert(False)
		elif s.dimension() == 2:
			max_value = np.max(np.array([ fnew[0,s[0]],fnew[0,s[1]],fnew[0,s[2]] ]))
			s.data = max_value.astype(np.float64)
			#print fnew[0,s[0]],fnew[0,s[1]],fnew[0,s[2]], "d", s.data,max_value
			if fnew[0,s[0]] == max_value:
				Tbl[s] = s[0]
			elif fnew[0,s[1]] == max_value:
				Tbl[s] = s[1]
			elif fnew[0,s[2]] == max_value:
				Tbl[s] = s[2]
			else:
				print("WARNING-topologicalutils: something is wrong")
				assert(False)
		elif s.dimension() == 3:
			max_value = np.max(np.array([ fnew[0,s[0]],fnew[0,s[1]],fnew[0,s[2]],fnew[0,s[3]] ]))
			s.data = max_value.astype(np.float64)
			#print fnew[0,s[0]],fnew[0,s[1]],fnew[0,s[2]], "d", s.data,max_value
			if fnew[0,s[0]] == max_value:
				Tbl[s] = s[0]
			elif fnew[0,s[1]] == max_value:
				Tbl[s] = s[1]
			elif fnew[0,s[2]] == max_value:
				Tbl[s] = s[2]
			elif fnew[0,s[3]] == max_value:
				Tbl[s] = s[3]
			else:
				print("WARNING-topologicalutils: something is wrong")
				assert(False)
		else:
			print("WARNING-topologicalutils: something is wrong. Too high dimensional")
			assert(False)

	F.sort() # sort filtration
	m = d.homology_persistence(F) # compute persistence
	dgms = returndgm(m,F,Tbl)
	return dgms,Tbl  # retur
