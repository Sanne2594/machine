#!/usr/bin/env bash
#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:20:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

#cd $HOME/machine/sanne/machine


MDL_DC="Model-DC-ET/acc_0.94_ppl_0.11_s100"
W_VEC=".1,.9"

#M_DATA="DataGeneration/disfluency-masks/train_ET-masks.txt"
#python3 ../../test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --print_wrong 1


echo "Testing for Correction"
M_DATA="DataGeneration/disfluency-masks/correction_tst_ET-masks.txt"
python3 ../../test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC
python3 ../../test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --stats

echo "Testing for Restart"
M_DATA="DataGeneration/disfluency-masks/restart_tst_ET-masks.txt"
python3 ../../test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --weight_vec $W_VEC
python3 ../../test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --stats


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "
