#!/bin/sh

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"

echo "start visualising"
MDL_LOC="model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600"
goal_direc="images-attention/"

python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --attviz $goal_direc

goal_direc="images-attention-minplus/"

python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75 --attviz $goal_direc

echo "start visualising plus"
PLS_LOC="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"
goal_direc="images-attention-plus/"

python3 evaluate.py --checkpoint_path $PLS_LOC --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --attviz $goal_direc

goal_direc="images-attention-plusplus/"

python3 evaluate.py --checkpoint_path $PLS_LOC --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75 --attviz $goal_direc


echo "Finished"