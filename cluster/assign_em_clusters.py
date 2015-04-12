import numpy as np
import sys
import glob
import util
import json

limit = None

def assign_clusters(rhos, mutpairs_flist):
  all_mutpairs = util.load_mutpairs(mutpairs_flist, in_parallel=False, limit=limit)
  assignments = [[] for _ in range(rhos.shape[0])]

  for mutpairs, fname in zip(all_mutpairs, mutpairs_flist):
    best_prob = float('-inf')
    best_model = None

    for model_idx, model in enumerate(rhos):
      model = np.exp(model)
      prob = model * mutpairs

      # Ensure no elements are zero.
      prob = prob[np.nonzero(prob)]
      assert prob.shape == model.shape[:-1]

      prob = np.sum(np.log(prob))
      if prob > best_prob:
	best_prob = prob
	best_model = model_idx

    #print(best_prob, best_model)
    assignments[best_model].append(fname)
  return assignments

def main():
  mutpairs_dir = sys.argv[1]
  mutpairs_flist = glob.glob(mutpairs_dir + "/mutpairs_-*")

  all_clusters = {}
  with np.load(sys.argv[2]) as em_rhos:
    for K in em_rhos.keys():
      clusters = assign_clusters(em_rhos[K], mutpairs_flist)
      clusters = [{'members': members} for members in clusters]
      all_clusters[K] = clusters
  print(json.dumps(all_clusters))

if __name__ == '__main__':
  main()
