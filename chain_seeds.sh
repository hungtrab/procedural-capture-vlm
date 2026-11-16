#!/usr/bin/env bash
# run_seed.sh for seed 43 then 44 of one tag on one GPU.
T=$1; G=$2; cd $HOME/emergent_poc/nb
bash run_seed.sh $T 43 $G && bash run_seed.sh $T 44 $G
