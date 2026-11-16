#!/bin/bash
# GPU 1: predictions of the anchored decoder, then the unconstrained and frozen-encoder ablations
cd /home/aj0486@students.ad.unt.edu/PFA/tracego
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1
tail --pid=1598665 -f /dev/null
../bin/python -m tracego.eval.predict --config configs/full.yaml --tag anchor --readouts step0,ccms,mix --benches tv,test,fr24p decoder.anchor=true decoder.anchor_l2=0.05 > runs/logs/full_pred_anchor.log 2>&1
tail --pid=1603517 -f /dev/null
tail --pid=1603204 -f /dev/null
B="decoder.epochs=2 decoder.anchor=true decoder.anchor_l2=0.05"
../bin/python -m tracego.train.train_decoder --config configs/full.yaml --tag abl_frozen $B encoder.tag=frozen35m encoder.lora_r=0 > runs/logs/full_dec_abl_frozen.log 2>&1
../bin/python -m tracego.train.train_decoder --config configs/full.yaml --tag abl_free $B decoder.constrained=false > runs/logs/full_dec_abl_free.log 2>&1
