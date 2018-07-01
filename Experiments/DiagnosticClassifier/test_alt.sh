#!/usr/bin/env bash
#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:20:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine


MDL_DC="Model-DC-alt/acc_0.80_ppl_0.34_s30"
W_VEC=".01,.99"

echo "Testing for Correction"
M_DATA="data/disfluency-masks/correction_test_alteration-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

echo "Testing for Restart"
M_DATA="data/disfluency-masks/restart_test_alteration-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --weight_vec $W_VEC

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "
