import json
import sys
import numpy as np

def main():
  with open(sys.argv[1]) as jsonf:
    clusters = json.load(jsonf)

  all_stats = []

  for K in sorted([int(i) for i in clusters.keys()]):
    lengths = sorted([len(c['members']) for c in clusters[str(K)]], reverse=True)

    non_zero = [e for e in lengths if e != 0]
    stats = [K - len(non_zero), np.mean(non_zero), np.median(non_zero)]
    all_stats += stats
  print(','.join([str(s) for s in all_stats]))

if __name__ == '__main__':
  main()
