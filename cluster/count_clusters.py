import json
import sys

def main():
  with open(sys.argv[1]) as jsonf:
    clusters = json.load(jsonf)
  for K in sorted([int(i) for i in clusters.keys()]):
    lengths = sorted([len(c['members']) for c in clusters[str(K)]], reverse=True)
    print(K, lengths)

if __name__ == '__main__':
  main()
