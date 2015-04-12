from __future__ import print_function
import numpy as np
import sys
import glob
import random
import itertools
import math
import json
import util

import matplotlib
# Force matplotlib not to use X11 backend, which produces exception when run
# over SSH.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def distance(a, b):
  return np.sum((a - b)**2)

def compute_distances(mutpairs):
  N = mutpairs.shape[0]
  distances = np.zeros((N, N))
  combos = itertools.combinations(range(N), 2)
  for a, b in combos:
    dist = distance(mutpairs[a], mutpairs[b])
    distances[a,b] = distances[b,a] = dist
  return distances

def assign_medioids(N, medioids, distances):
  assignments = {}

  for i in range(N):
    if i in medioids:
      continue
    closest_medioid = None
    closest_distance = float('inf')
    for m in medioids:
      if distances[i,m] < closest_distance:
	closest_medioid = m
	closest_distance = distances[i,m]
    assignments[i] = closest_medioid

  return assignments

def compute_cost(assignments, distances):
  cost = 0
  for mutpair, medioid in assignments.items():
    cost += distances[mutpair, medioid]
  return cost

def test_all_pairwise_swaps(N, medioids, distances):

  best_medioids = medioids
  assignments = assign_medioids(N, best_medioids, distances)
  best_cost = compute_cost(assignments, distances)

  for m in medioids:
    candidate_medioids = medioids.copy()
    points = set(range(N)) - candidate_medioids
    candidate_medioids.remove(m)

    for point in points:
      candidate_medioids.add(point)
      assignments = assign_medioids(N, candidate_medioids, distances)
      cost = compute_cost(assignments, distances)
      if cost < best_cost:
	best_medioids = candidate_medioids.copy()
	best_cost = cost
      candidate_medioids.remove(point)

  return best_medioids

def cluster(N, K, distances):
  medioids = set(random.sample(range(N), K))

  while True:
    new_medioids = test_all_pairwise_swaps(N, medioids, distances)
    if new_medioids == medioids:
      break
    medioids = new_medioids
    #log('Choosing new medioids: %s\t\t%s' % (medioids, compute_cost(assign_medioids(N, medioids, distances), distances)))

  assignments = assign_medioids(N, medioids, distances)
  cost = compute_cost(assignments, distances)
  return (medioids, assignments)

def dumb(N, K, distances):
  combos = itertools.combinations(range(N), K)

  best_cost = float('inf')
  best_medioids = None

  f = math.factorial
  total = f(N) / (f(N - K) * f(K))
  i = 0

  for medioids in combos:
    i += 1
    if i % 10000 == 0:
      print(i, total)

    points = set(range(N)) - set(medioids)
    assignments = assign_medioids(N, medioids, distances)
    cost = compute_cost(assignments, distances)
    if cost < best_cost:
      best_cost = cost
      best_medioids = medioids

  return (best_medioids, assign_medioids(N, best_medioids, distances))

def log(*args):
  print(*args, file=sys.stderr)

def main():
  fdir = sys.argv[1]
  flist = glob.glob(fdir + "/mutpairs_-*")
  mutpairs = util.load_mutpairs(flist, limit=None, in_parallel=False)
  log('Done loading mutpairs')

  N = mutpairs.shape[0]
  distances = compute_distances(mutpairs)
  log('Done computing distances')

  mapping = {}
  maxK = 10
  for K in range(1, maxK + 1):
    medioids, assignments = cluster(N, K, distances)
    log('Done for k=%s' % K)

    mapping[K] = []
    for medioid in medioids:
      members = [point for (point, med) in assignments.items() if med == medioid]
      members = [util.extract_score(flist[idx]) for idx in members]
      med_score = util.extract_score(flist[medioid])
      members.append(med_score)
      mapping[K].append({
	'medioid': med_score,
	'members': members,
      })

  print(json.dumps(mapping))

def test_in_2d():
  K = 4
  per_class = 50
  y0 = np.random.multivariate_normal([0, 0], [[2, 0], [0, 0.1]], size=per_class)
  y1 = np.random.multivariate_normal([0, 0], [[0.1, 0], [0, 2]], size=per_class)
  y2 = np.random.multivariate_normal([2, 2], [[2, -1.5], [-1.5, 2]], size=per_class)
  y3 = np.random.multivariate_normal([-2, -2], [[0.5, 0], [0, 0.5]], size=per_class)
  y = np.vstack([y0, y1, y2, y3])
  N = y.shape[0]

  distances = compute_distances(y)
  medioids, assignments = cluster(N, K, distances)
  print(medioids)

  colours = N*['b']
  sizes = N*[15]

  palette = ['r', 'g', 'b', 'y']
  colour_ass = {}

  for m in medioids:
    colour_ass[m] = palette.pop()
    colours[m] = colour_ass[m]
    sizes[m] = 200
  for a, b in assignments.items():
    colours[a] = colour_ass[b]

  xcoords, ycoords = zip(*y)
  plt.scatter(xcoords, ycoords, c=colours, s=sizes)
  plt.savefig('pants.png')

if __name__ == '__main__':
  main()
  #test_in_2d()
