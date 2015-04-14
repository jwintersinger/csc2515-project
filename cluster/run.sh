
#!/bin/bash
set -euo pipefail
shopt -s nullglob

function set_vars {
  export PYTHONPATH=~/.apps/phylowgs:$PYTHONPATH
  export PYTHON=~/.apps/bin/python2
  export PDFLATEX=~/.apps/bin/pdflatex

  export PROT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  export RUN_NAME=train63.sample-5000.psub_final_bb.run-1
  export RUN_DIR=$HOME/work/exultant-pistachio/data/runs/$RUN_NAME
  export TREES_DIR=$RUN_DIR/trees
  export MUTPAIRS_DIR=$RUN_DIR/mutpairs

  export LOG_DIR=$HOME/jobs/cluster-trees/$RUN_NAME
}

function store_guids {
  cd $MUTPAIRS_DIR
  export GUIDS_PATTERN="????????-????-????-????-????????????"
  #export GUIDS=$(ls -d $GUIDS_PATTERN)
  export GUIDS="0c7af04b-e171-47c4-8be5-5db33f20148e 1c00925b-7328-4db0-b930-04aab2d80719 41a7b031-d928-4a1f-891b-82fb3f6d548f 6847e993-1414-4e6f-a2af-39ebe218dd7c 6dac8ca0-f776-4ea2-85c4-aefba4966be7 7f1ca00a-84ba-4b26-a1a5-4d83c363ac69 81a8b064-e735-455f-b2db-af7ae11daac4 84ca6ab0-9edc-4636-9d27-55cdba334d7d 878a7fe7-20ff-4651-9587-b4d6fd42e929 8d4cb709-c95c-4bdc-844b-c0bfa2a3028e 9d29543e-8601-4fd0-8e76-3df3de465cab af96db5a-684f-41d1-a910-5a5193393d9c c9e7c629-7b57-4ede-b315-0cea8c97c48e cd3d7559-b583-4474-81df-4bf9232de3c2 d8f0becd-fda8-41f4-a424-e082f9eae22c f64e9609-d75d-400c-a92d-d77fd54d6c29 fb0c6353-a90c-45e2-9355-7cd16cf756ff"
  #export GUIDS="0c7af04b-e171-47c4-8be5-5db33f20148e"
  echo $GUIDS
}

function create_dirs {
  rm -rf $LOG_DIR && mkdir -p $LOG_DIR
}

function split_data_sets {
  cd $MUTPAIRS_DIR
  for sampid in $GUIDS; do
    cd $sampid
    mkdir mutpairs_all
    mv mutpairs_-* mutpairs_all/
    $PYTHON $PROT_DIR/split_data_sets.py mutpairs_all/ | bash
    cd ..
  done
}

function cluster_datasets {
  cd $MUTPAIRS_DIR
  for dclass in training all; do
    for sampid in $GUIDS; do
      cd $sampid/mutpairs_$dclass

      job_name=cluster_$dclass.kmed.$sampid
      log_prefix=$LOG_DIR/$job_name
      output=$PWD/kmed_clusters.$dclass.$sampid.json
      rm -f $output
      qsub -b y -cwd -V -l h_vmem=20G -N $job_name \
	-o $output -e $log_prefix.stderr \
	">&2 echo \$HOSTNAME && $PYTHON $PROT_DIR/kmedioids.py ./"

      job_name=cluster_$dclass.em.$sampid
      log_prefix=$LOG_DIR/$job_name
      qsub -b y -cwd -V -l h_vmem=32G -N $job_name \
	-o $log_prefix.stdout -e $log_prefix.stderr \
	">&2 echo \$HOSTNAME && $PYTHON $PROT_DIR/em.py ./"

      cd ../..
    done
  done
}

function assign_em_clusters {
  cd $MUTPAIRS_DIR
  for sampid in $GUIDS; do
    cd $sampid/mutpairs_all

    job_name=assign_em_clusters.$sampid
    log_prefix=$LOG_DIR/$job_name
    output=$PWD/em_clusters.all.$sampid.json
    rm -f $output
    qsub -b y -cwd -V -l h_vmem=16G -N $job_name -hold_jid cluster_all.em.$sampid \
      -o $output -e $log_prefix.stderr \
      ">&2 echo \$HOSTNAME && $PYTHON $PROT_DIR/assign_em_clusters.py ./ rho.npz"

    cd ../..
  done
}

function enumerate_clustered_trees {
  for sampid in $GUIDS; do
    cd $MUTPAIRS_DIR/$sampid/mutpairs_all
    rm -rf members && mkdir members && cd members

    for cluster_type in em kmed; do
      if [[ $cluster_type == "em" ]]; then
	hold_jid=assign_em_clusters.$sampid
      else
	hold_jid=cluster_all.kmed.$sampid
      fi

      job_name=enumerate_clustered_trees.$cluster_type.$sampid
      log_prefix=$LOG_DIR/$job_name
      export cluster_type
      export sampid
      export PDFLATEX

      rm -f $log_prefix.std{err,out}
      qsub -b y -cwd -V -N $job_name -hold_jid $hold_jid \
	-o $log_prefix.stdout -e $log_prefix.stderr \
	$PROT_DIR/_enumerate_clustered_trees.sh
    done
  done
}

function eval_test_datasets {
  for sampid in $GUIDS; do
    cd $MUTPAIRS_DIR/$sampid
    job_name=eval_test_datasets.$sampid
    log_prefix=$LOG_DIR/$job_name
    output=test_likelihood.$sampid.txt
    rm -f $log_prefix.stderr $output
    qsub -b y -cwd -V -l h_vmem=16G -N $job_name \
      -hold_jid cluster_training.kmed.$sampid,cluster_training.em.$sampid \
      -o $output -e $log_prefix.stderr \
      ">&2 echo \$HOSTNAME && $PYTHON $PROT_DIR/assess_clusters.py mutpairs_training mutpairs_test" \
      "mutpairs_training/kmed_clusters.training.$sampid.json mutpairs_training/{rho,pi}.npz"
  done
}

function main {
  set_vars
  store_guids
  #create_dirs
  #split_data_sets
  #cluster_datasets
  #assign_em_clusters
  #enumerate_clustered_trees
  eval_test_datasets
}

main
