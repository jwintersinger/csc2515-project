from __future__ import print_function
import glob
import sys
import util
import numpy as np
import json

limit = 10

def create_kmed_models(clusters, mutpairs_flist):
  mixing_props = np.zeros(len(clusters))
  for i, cluster in enumerate(clusters):
    mixing_props[i] = len(cluster['members'])
  mixing_props /= np.sum(mixing_props)
  log_mixing_props = np.log(mixing_props)

  models = []
  for cluster in clusters:
    members = cluster['members']
    fpaths = []
    for fpath in mutpairs_flist:
      for member in members:
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
    models.append(mean_mutpairs)

  models = np.array(models)
  return (models, log_mixing_props)

def calc_prob(models, log_mixing_props, mutpairs_flist):
  all_mutpairs = util.load_mutpairs(mutpairs_flist, in_parallel=False, limit=limit)
  all_probs = []

  for mutpairs in all_mutpairs:
    model_probs = []
    for model, log_mixing_prop in zip(models, log_mixing_props):
      prob = mutpairs * model

      # Ensure no elements are zero.
      prob = prob[np.nonzero(prob)]
      assert prob.shape == model.shape[:-1]

      prob = np.sum(np.log(prob))
      prob += log_mixing_prop
      model_probs.append(prob)
    all_probs.append(np.logaddexp.reduce(model_probs))
  # Return product of all_probs, which is in log space.
  return np.sum(all_probs)

def assess_k_medioids(kmed_clusters, training_flist, test_flist):
  for K, k_clusters in kmed_clusters.items():
    models, log_mixing_props = create_kmed_models(k_clusters, training_flist)
    prob = calc_prob(models, log_mixing_props, test_flist)
    yield (int(K), prob)

def assess_em(rhos, pis, test_flist):
  assert pis.keys() == rhos.keys()
  for K in pis.keys():
    prob = calc_prob(np.exp(rhos[K]), pis[K], test_flist)
    yield (int(K), prob)

def main():
  training_dir = sys.argv[1]
  test_dir = sys.argv[2]
  with open(sys.argv[3]) as kmed_clusterf:
    kmed_clusters = json.load(kmed_clusterf)

  training_flist = glob.glob(training_dir + "/mutpairs_-*")
  test_flist = glob.glob(test_dir + "/mutpairs_-*")

  results = {'em': {}, 'kmed': {}}

  for K, prob in assess_k_medioids(kmed_clusters, training_flist, test_flist):
    print('kmed', K, prob, file=sys.stderr)
    results['kmed'][K] = prob
  with np.load(sys.argv[4]) as em_rhos:
    with np.load(sys.argv[5]) as em_pis:
      for K, prob in assess_em(em_rhos, em_pis, test_flist):
	print('em', K, prob, file=sys.stderr)
	results['em'][K] = prob

  assert sorted(results['kmed'].keys()) == sorted(results['em'].keys())
  keys = sorted(results['kmed'].keys())
  output = []
  cols = []
  for method in ('kmed', 'em'):
    for K in keys:
      output.append(results[method][K])
      cols.append('%s%s' % (method, K))
  for arr in (cols, output):
    print(','.join([str(s) for s in arr]))

if __name__ == '__main__':
  main()
  #test_in_2d()
