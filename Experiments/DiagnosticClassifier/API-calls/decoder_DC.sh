#!/usr/bin/env bash

MDL_LOC="../../Results-Final/model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"
M_DATA="Data-normal/test_d.txt"
NUM_CLASS=2

#MDL_LOC="../../Results-Final/model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600"

python3 ../../../evaluate.py --checkpoint_path $MDL_LOC --test_data $M_DATA --max_len 75 --select_eval --print_wrong 1

#NUM_EP=6
#MDL_LOC="results/Models-1/acc_0.46_ppl_1.30_s1000"
#
#MDL_DC="Model-DC-cor"
#M_DATA="Data-normal/test_d.txt"
#
#
#python3 ../../../extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC --print_wrong 1
