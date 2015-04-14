import glob
import sys
import numpy as np
import os
import csv

def find_most_likely_K(row, prefix):
  scores = {key: val for (key, val) in row.items() if key.startswith(prefix)}
  max_score = float('-inf')
  max_K = None
  for K, score in scores.items():
    score = float(score)
    if score > max_score:
      max_K = K
      max_score = score

  assert max_K.startswith(prefix)
  K = int(max_K[len(prefix):])
  return K

def extract_tree_fraction(tree_fn):
  frac = tree_fn.rsplit('_', 1)[1]
  frac = frac.rsplit('.', 1)[0]
  return float(frac)

def calc_num_empty_clusters(clusters):
  num_empty_clusters = len([c for c in clusters if len(glob.glob(os.path.join(c, '-*'))) == 0])
  return num_empty_clusters

def calc_mean_cluster_quality(clusters):
  all_trees_count = [len(glob.glob(os.path.join(c, '-*'))) for c in clusters]
  unique_trees_count = [len(glob.glob(os.path.join(c, '*.png'))) for c in clusters]
  all_trees_count = np.array([c for c in all_trees_count if c != 0], dtype=np.float_)
  unique_trees_count = np.array([c for c in unique_trees_count if c != 0], dtype=np.float_)
  mean_cluster_quality = np.mean(all_trees_count / unique_trees_count)
  return mean_cluster_quality

def calc_mean_top_tree_fraction(clusters):
  unique_trees = [glob.glob(os.path.join(c, 'tree_0_*.png')) for c in clusters]
  for uniq in unique_trees:
    assert len(uniq) in (0, 1)
  unique_trees = [u[0] for u in unique_trees if len(u) == 1]
  top_tree_fractions = np.array([extract_tree_fraction(u) for u in unique_trees], dtype=np.float_)
  mean_top_tree_fraction = np.mean(top_tree_fractions)
  return mean_top_tree_fraction

def main():
  cluster_type = sys.argv[1]
  assert cluster_type in ('kmed', 'em')

  with open(sys.argv[2]) as likelihoods_csv:
    reader = csv.DictReader(likelihoods_csv)
    rows = list(reader)
  assert len(rows) == 1
  best_K = find_most_likely_K(rows[0], cluster_type)

  members_dir = sys.argv[3]
  cluster_dir = os.path.join(members_dir, cluster_type + str(best_K))
  clusters = glob.glob(os.path.join(cluster_dir, 'cluster*'))

  cols = ['most_likely_K', 'num_empty_clusters', 'mean_cluster_quality', 'top_tree_fraction']
  output = [best_K, calc_num_empty_clusters(clusters), calc_mean_cluster_quality(clusters), calc_mean_top_tree_fraction(clusters)]
  for arr in (cols, output):
    print(','.join([str(s) for s in arr]))

main()

