#!/bin/sh

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"

echo "start visualising"
goal_direc="images-attention-plusplus-allapi/"
PLS_LOC="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"

python3 evaluate.py --checkpoint_path $PLS_LOC --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75 --attviz $goal_direc


echo "Finished"