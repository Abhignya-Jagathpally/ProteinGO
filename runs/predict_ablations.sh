#!/bin/bash
# predictions (tv, test, fr24p) for each ablation once its training process has exited
cd /home/aj0486@students.ad.unt.edu/PFA/tracego
export PYTHONPATH=$PWD
G=$1; shift
export CUDA_VISIBLE_DEVICES=$G
B="decoder.anchor=true decoder.anchor_l2=0.05"
for job in "$@"; do
  tag=${job%%:*}; extra=${job#*:}; [ "$extra" = "$job" ] && extra=""
  while pgrep -f "tag $tag " > /dev/null; do tail --pid=$(pgrep -f "tag $tag " | head -1) -f /dev/null; done
  ../bin/python -m tracego.eval.predict --config configs/full.yaml --tag $tag --readouts step0,ccms,mix --benches tv,test,fr24p $B $extra > runs/logs/full_pred_$tag.log 2>&1
done
