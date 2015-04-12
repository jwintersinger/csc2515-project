import glob
import os
import sys
import random

def process(suffix, files):
  print('mkdir mutpairs_%s' % suffix)
  print('cd mutpairs_%s' % suffix)
  for tfile in files:
    print('ln -s ../%s' % tfile)
  print('cd ..')

def main():
  input_dir = sys.argv[1]
  input_files = glob.glob(os.path.join(input_dir, 'mutpairs_-*'))
  random.shuffle(input_files)

  training_size = 2000
  training_files = input_files[:training_size]
  test_files = input_files[training_size:]

  process('training', training_files)
  process('test', test_files)

main()
