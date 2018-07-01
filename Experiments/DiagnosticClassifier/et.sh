#!/usr/bin/env bash
#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:20:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

#cd $HOME/machine/sanne/machine

NUM_EP=70 #Number of Epochs
MDL_LOC="../Results-Final/model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"

MDL_DC="Model-DC-ET"
M_DATA="DataGeneration/disfluency_masks/ET-masks.txt"

NUM_CLASS=2
W_VEC=".01,.99"

python3 ../../extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "
