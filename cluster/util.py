import multiprocessing
import numpy as np

def load_mutpairs(fnames, in_parallel=True, limit=None):
  if limit:
    fnames = fnames[:limit]
  if in_parallel:
    results = multiprocessing.Pool(8).map(load_single_mutpairs, fnames)
  else:
    results = map(load_single_mutpairs, fnames)

  mutpairs = np.zeros(shape=(len(fnames), results[0].shape[0], 4), dtype=np.bool)
  for i, fname in enumerate(fnames):
    mutpairs[i,:,:] = results[i]
  return mutpairs

def load_single_mutpairs(fname):
  t = np.loadtxt(fname)
  t = t.astype(np.bool, copy=False)
  t = t.T
  return t

def extract_score(fname):
  fname = fname.rsplit('_', 1)[1]
  fname = fname.rsplit('.', 2)[0]
  return fname
