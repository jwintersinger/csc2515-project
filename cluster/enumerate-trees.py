import argparse
import os
import cPickle as pickle
import numpy as np
from collections import Counter

# removes the empty nodes from the tssb tree
# Does not removes root as it is not required
# root: root of the current tree
# parent: parent of the root
def remove_empty_nodes(root, parent):
  for child in list(root['children']):
    remove_empty_nodes(child, root)
  if (root['node'].get_data() == []):
    if (root['children'] == []): # leaf
      if (parent != None):
        parent['children'].remove(root)
        root['node'].kill()
      return
    else:
      if (parent != None):
        parent_ = root['node'].parent()
        for child in list(root['children']):
          parent['children'].append(child)
          root['children'].remove(child)
        for child in list(root['node'].children()):
          child._parent = parent_
          parent_.add_child(child)
          root['node'].remove_child(child)
        parent['children'].remove(root)
        root['node'].kill()

def generate_matrix(tree):
  def _count_verts(node):
    return 1 + sum([_count_verts(c) for c in node.children()])
  num_verts = _count_verts(tree.root['node'])
  # Boolean should suffice, since this is an adjacency matrix, but I prefer
  # working with integer values.
  mat = np.zeros((num_verts, num_verts), dtype=np.int8)

  # List rather than scalar -- hack to access outer scope without making
  # variable global. Easier in Python 3, which has "nonlocal" keyword.
  node_idx = [0]

  def _traverse_r(node):
    parent_idx = node_idx[0]
    for child in node.children():
      node_idx[0] += 1
      child_idx = node_idx[0]
      mat[parent_idx,child_idx] = 1
      mat[child_idx,parent_idx] = 1
      _traverse_r(child)
  _traverse_r(tree.root['node'])

  return mat

def generate_tex(adjmat):
  tree_tex  = '\documentclass{standalone}\n'  
  tree_tex += '\usepackage{tikz}\n'
  tree_tex += '\usepackage{multicol}\n'
  tree_tex += '\usetikzlibrary{fit,positioning}\n'
  tree_tex += '\\begin{document}\n'
  tree_tex += '\\begin{tikzpicture}\n'
  tree_tex += '\\node (a) at (0,0){\n'
  tree_tex += '\\begin{tikzpicture}\n'  
  tree_tex += '[grow=east, ->, level distance=20mm,\
  every node/.style={circle, minimum size = 8mm, thick, draw =black,inner sep=2mm},\
  every label/.append style={shape=rectangle, yshift=-1mm},\
  level 2/.style={sibling distance=50mm},\
  level 3/.style={sibling distance=20mm},\
  level 4/.style={sibling distance=20mm},\
  every edge/.style={-latex, thick}]\n'
  tree_tex += '\n\\'

  tree_tex =  generate_tree_tex(adjmat, 0, tree_tex)

  tree_tex += ';\n'
  tree_tex += '\\end{tikzpicture}\n'
  tree_tex += '};\n'  
  tree_tex += '\\end{tikzpicture}\n'
  tree_tex += '\\end{document}\n'

  return tree_tex

def generate_tree_tex(adjmat, node_idx, tree_tex):
  tree_tex += 'node {%s}' % node_idx

  children = np.nonzero(adjmat[node_idx])[0]
  children = [c for c in children if c > node_idx]
  for child_idx in children:
    tree_tex += 'child {'
    tree_tex = generate_tree_tex(adjmat, child_idx, tree_tex)
    tree_tex += '}'
  return tree_tex

def enumerate(tree_dir):
  flist = sorted(os.listdir(tree_dir))
  mats = []
  buffers_to_mats = {}

  for tree_fname in flist:
    score = float(tree_fname)
    with open(os.path.join(tree_dir, tree_fname)) as tree_file:
      tree = pickle.load(tree_file)
    remove_empty_nodes(tree.root, None)

    mat = generate_matrix(tree)
    mat.flags.writeable = False # Permit hashing for Counter
    buffers_to_mats[mat.data] = mat
    mats.append(mat)

  # Must use mat.data rather than just mat, as mat is unhashable.
  counter = Counter([m.data for m in mats])
  total_trees = len(list(counter.elements()))
  tree_idx = 0

  for mat_buffer, count in counter.most_common():
    mat = buffers_to_mats[mat_buffer]
    tex = generate_tex(mat)
    fname = 'tree_%s_%s.tex' % (tree_idx, round(float(count) / total_trees, 4))
    with open(fname, 'w') as fout:
      fout.write(tex)
    tree_idx += 1

def main():
  parser = argparse.ArgumentParser(
    description='Plot posterior trees resulting from PhyloWGS run',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument('trees_dir',
    help='Directory where the MCMC trees/samples are saved')
  args = parser.parse_args()

  enumerate(args.trees_dir)

main()
