#!/bin/bash
# decoder ablations, anchored config, 2 epochs each (GPU 0)
cd /home/aj0486@students.ad.unt.edu/PFA/tracego
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0
B="decoder.epochs=2 decoder.anchor=true decoder.anchor_l2=0.05"
../bin/python -m tracego.train.train_decoder --config configs/full.yaml --tag abl_noevid $B decoder.evidence=false decoder.copy_bias=false > runs/logs/full_dec_abl_noevid.log 2>&1
../bin/python -m tracego.train.train_decoder --config configs/full.yaml --tag abl_nodef $B decoder.def_init=false > runs/logs/full_dec_abl_nodef.log 2>&1
../bin/python -m tracego.train.train_decoder --config configs/full.yaml --tag abl_noia $B decoder.ia_weight=false > runs/logs/full_dec_abl_noia.log 2>&1
