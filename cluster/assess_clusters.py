from __future__ import print_function
import glob
import sys
import util
import numpy as np
import json

limit = None

def create_models(clusters, mutpairs_flist):
  mixing_props = np.zeros(len(clusters.keys()))
  medioids = clusters.keys()
  for i, med in enumerate(medioids):
    mixing_props[i] = len(clusters[med])
  mixing_props /= np.sum(mixing_props)
  log_mixing_props = dict(zip(medioids, np.log(mixing_props)))

  models = {}
  for medioid, members in clusters.items():
    all_members = [medioid] + members
    fpaths = []
    for fpath in mutpairs_flist:
      for member in all_members:
	if member in fpath:
	  fpaths.append(fpath)

    mutpairs = util.load_mutpairs(fpaths, in_parallel=False, limit=limit)
    num_trees = mutpairs.shape[0]
    # Force float type. Otherwise, this gets cast to int64 -- by default, NumPy
    # will use data type of first operand, unless that type is an integer with
    # less precision than the default platform integer. In this case, bool is
    # less precise than int64, so the latter is used. This messes up the
    # division by num_trees, since the desination array can't hold such values.
    mean_mutpairs = np.sum(mutpairs, axis=0, dtype=np.float_)
    # Add pseudocounts to ensure no probability in matrix is zero. This in turn
    # would give a zero in (model * example_tree) pairwise multiplication,
    # indicating that there is zero probaility that example_tree came from the
    # model associated with a given cluster. As we often see only five or six
    # pairs with a zero probability from this calculation, amongst >100k+ pairs
    # for a given tree, we don't want to conclude there's zero probability that
    # the tree came from the given cluster.
    pseudo_count = 1
    mean_mutpairs += pseudo_count * np.ones(mean_mutpairs.shape)
    # This ensures all values will be in [0, 1].
    mean_mutpairs /= (float(num_trees) + pseudo_count)
    models[medioid] = mean_mutpairs

  return (models, log_mixing_props)

def calc_prob(models, mixing_props, mutpairs_flist):
  all_mutpairs = util.load_mutpairs(mutpairs_flist, in_parallel=False, limit=limit)
  all_probs = []

  for mutpairs in all_mutpairs:
    model_probs = []
    for medioid, model in models.items():
      prob = mutpairs * model
      prob = prob[np.nonzero(prob)]
      # Compare tuple to tuple.
      assert prob.shape == model.shape[:-1]
      #print(len(np.nonzero(prob != 1.1)))
      prob = np.sum(np.log(prob))
      prob += mixing_props[medioid]
      model_probs.append(prob)
    all_probs.append(np.logaddexp.reduce(model_probs))
  # Return product of all_probs, which is in log space.
  return np.sum(all_probs)

def main():
  training_dir = sys.argv[1]
  test_dir = sys.argv[2]
  with open(sys.argv[3]) as clusterf:
    clusters = json.load(clusterf)

  training_flist = glob.glob(training_dir + "/mutpairs_*")
  test_flist = glob.glob(test_dir + "/mutpairs_*")

  for K, k_clusters in clusters.items():
    models, log_mixing_props = create_models(clusters[K], training_flist)
    a = calc_prob(models, log_mixing_props, test_flist)
    print(K, a)

if __name__ == '__main__':
  main()
  #test_in_2d()
