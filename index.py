import numpy as np
import argparse
import time
import heapq

data = np.loadtxt("data10K10.txt", delimiter=" ", dtype=float)
queries = np.loadtxt("queries10.txt", delimiter=" ", dtype=float)

parser = argparse.ArgumentParser()

parser.add_argument("--pivots", "-p", type = int, default=10, 
	help="How many pivot points to use (heuristic)")
parser.add_argument("--epsilon", "-e", type = float, default=0.2, 
	help="How far away to search for points in the range queries")
parser.add_argument("--knn", "-k", type = int, default=5, 
	help="The number k of nearest neighbors to find")

args = parser.parse_args()

# A bit less nice than the Zote code, as we don't do "real" count of dist,
# and instead delegate that to functions
def dist(x, y):
	return np.linalg.norm(np.subtract(x, y))

seed = data[0]
dists0 = np.apply_along_axis(dist, 1, data, seed)
pivot_inds = [dists0.argmax()]

dists = np.apply_along_axis(dist, 1, data, data[pivot_inds[-1]]).reshape(-1, 1)

while len(pivot_inds) < args.pivots:
	pivot_inds.append(np.sum(dists, axis=1, keepdims=True).argmax())

	new_dists = np.apply_along_axis(dist, 1, data, data[pivot_inds[-1]])
	dists = np.column_stack((dists, new_dists))

def naive_range_query(query, eps):
	dists = np.apply_along_axis(dist, 1, data, query)
	res = []
	i = 0
	for d in dists:
		if d <= eps:
			res.append(i)
		i = i + 1
	return res, len(dists)

def pivot_range_query(query, eps):
	q_dists = [dist(query, data[pivot_ind,:]) for pivot_ind in pivot_inds]
	comps = len(q_dists)
	res = []
	# What beautiful nesting (:
	for line in range(len(data)):
		if all(abs(dists[line, i] - q_dist) <= eps for (i, q_dist) in enumerate(q_dists)):
			comps += 1
			if dist(query, data[line, :]) < eps:
				res.append(line)

	return res, comps

def test_range_queries():
	# No actual sanity checks, as compared to zote
	naive_comps = 0
	start = time.time()
	for query in range(len(queries)):
		_, cmp = naive_range_query(queries[query,:], args.epsilon)
		naive_comps += cmp
	naive_time = time.time() - start

	pivot_comps = 0
	start = time.time()
	for query in range(len(queries)):
		_, cmp = pivot_range_query(queries[query,:], args.epsilon)
		pivot_comps += cmp
	pivot_time = time.time() - start
	
	print(f"average distance comp per range query (Naive) = {naive_comps/len(queries)}")
	print(f"average distance comp per range query (Pivot) = {pivot_comps/len(queries)}")

	print(f"Total time Naive = {naive_time}")
	print(f"Total time Pivot = {pivot_time}\n")

test_range_queries()

def naive_knn_query(query, k):
	dists = np.apply_along_axis(dist, 1, data, query)
	sorted_dist_inds = np.argsort(dists)
	smallest_inds = sorted_dist_inds[:5]
	smallest_dists = dists[smallest_inds]
	return smallest_inds, smallest_dists, len(dists)

def pivot_knn_query(query, k):
	q_dists = [dist(query, data[pivot_ind,:]) for pivot_ind in pivot_inds]
	comps = len(q_dists)

	heap = []
	for o_ind in range(k):
		# Negative to conv to max-heap
		heapq.heappush(heap, (-dist(query, data[o_ind, :]), o_ind))
		comps += 1
	eps = -heap[0][0]
	
	for line in range(k, len(data)):
		if all(abs(dists[line, i] - q_dist) <= eps for (i, q_dist) in enumerate(q_dists)):
			d = dist(query, data[line, :])
			comps += 1
			if d  < eps:
				heapq.heappop(heap)
				heapq.heappush(heap, (-d, line))
				eps = -heap[0][0]

	return heap, comps

def test_knn_queries():
	naive_comps = 0
	start = time.time()
	for query in range(len(queries)):
		_, _, cmp = naive_knn_query(queries[query,:], args.knn)
		naive_comps += cmp
	naive_time = time.time() - start

	pivot_comps = 0
	start = time.time()
	for query in range(len(queries)):
		_, cmp = pivot_knn_query(queries[query,:], args.knn)
		pivot_comps += cmp
	pivot_time = time.time() - start

	print(f"average distance comp per knn query (Naive) = {naive_comps/len(queries)}")
	print(f"average distance comp per knn query (Pivot) = {pivot_comps/len(queries)}")

	print(f"Total time Naive = {naive_time}")
	print(f"Total time Pivot = {pivot_time}")
	
test_knn_queries()
