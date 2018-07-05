#!/usr/bin/env bash

MDL_LOC="../../Results-Final/model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"
M_DATA="Data-normal/clean-babi-tst.txt"
NUM_CLASS=2

#MDL_LOC="../../Results-Final/model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600"

python3 ../../../evaluate.py --checkpoint_path $MDL_LOC --test_data $M_DATA --max_len 75 --select_eval --print_wrong 1 >out.txt

