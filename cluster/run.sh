
#!/bin/bash
set -euo pipefail
shopt -s nullglob

function set_vars {
  export PYTHONPATH=~/.apps/phylowgs:$PYTHONPATH
  export PYTHON=~/.apps/bin/python2

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
  export GUIDS=$(ls -d $GUIDS_PATTERN)
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
      qsub -b y -cwd -V -l h_vmem=20G -N $job_name \
	-o $PWD/kmed_clusters.$dclass.$sampid.json -e $log_prefix.stderr \
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
    qsub -b y -cwd -V -l h_vmem=16G -N $job_name -hold_jid cluster_all.em.$sampid \
      -o $PWD/em_clusters.all.$sampid.json -e $log_prefix.stderr \
      ">&2 echo \$HOSTNAME && $PYTHON $PROT_DIR/assign_em_clusters.py ./ rho.npz"

    cd ../..
  done
}

function eval_test_datasets {
  echo Penis
}

function main {
  set_vars
  store_guids
  create_dirs
  #split_data_sets
  cluster_datasets
  assign_em_clusters
}

main
