import sys
import json

def main():
  prefix = sys.argv[1]
  tree_dir = sys.argv[2]
  with open(sys.argv[3]) as jsonf:
    all_clusters = json.load(jsonf)
  for K, clusters in all_clusters.items():
    dirname = '%s%s' % (prefix, K)
    print('mkdir %s && cd %s' % (2*(dirname,)))

    for idx, cluster in enumerate(clusters):
      members = cluster['members']
      dirname = 'cluster%s' % idx
      print('mkdir %s && cd %s' % (2*(dirname,)))
      for score in members: 
	print('ln -s %s/%s' % (tree_dir, score))
      print('cd ..')
    print('cd ..')

  print('cd ..')

main()
