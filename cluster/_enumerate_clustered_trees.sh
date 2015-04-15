#!/bin/bash
set -euo pipefail
shopt -s nullglob

# Fix "libgomp: Thread creation failed: Resource temporarily unavailable" error.                                                                                                                                                                                                 
export MAGICK_THREAD_LIMIT=1 

$PYTHON $PROT_DIR/group_trees_by_cluster.py $cluster_type $RUN_DIR/trees/$sampid/trees ../${cluster_type}_clusters.all.$sampid.json | bash

for clustered_dir in ${cluster_type}*; do
  cd $clustered_dir
  for cluster in cluster*; do
    cd $cluster

    $PYTHON $PROT_DIR/enumerate-trees.py .

    for tree in *.tex; do
      $PDFLATEX -interaction=nonstopmode $tree > /dev/null
    done

    for pdf in *.pdf; do
      convert -density 96 $pdf -quality 90 $(echo $pdf | sed 's/pdf$/png/')
    done

    cd ..
  done

  cd ..
done

