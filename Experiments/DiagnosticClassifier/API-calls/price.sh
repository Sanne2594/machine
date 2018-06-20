#!/usr/bin/env bash
#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:50:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

#cd $HOME/machine/sanne/machine

NUM_EP=100 #Number of Epochs
MDL_LOC="../../Results-Final/model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"
MDL_DC="Model-DC-price"

MDL_LOC="Model-DC-price/acc_0.51_ppl_0.96_s10"

#M_DATA="data/Data-1/price_range_masks.txt"
M_DATA="Data-1/price_range_masks.txt"
NUM_CLASS=3
W_VEC="1,.9,.9"

#python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC --print_wrong 100
python3 ../../../extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC --print_wrong 100


echo "Made it"



