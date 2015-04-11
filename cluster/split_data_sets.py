import glob
import os
import sys
import random

def main():
  input_dir = sys.argv[1]
  input_files = glob.glob(os.path.join(input_dir, 'mutpairs_*'))
  random.shuffle(input_files)

  training_size = 2000
  training_files = input_files[:training_size]
  test_files = input_files[training_size:]

  print('mkdir mutpairs_training')
  print('mkdir mutpairs_test')
  for tfile in training_files:
    print('ln -s %s mutpairs_training/' % tfile)
  for tfile in test_files:
    print('ln -s %s mutpairs_test/' % tfile)

main()
