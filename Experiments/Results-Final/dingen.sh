#!/bin/sh

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"


echo "\nbabi on babi plus"
python3 evaluate.py --checkpoint_path "model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600" --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600" --test_data "data/CLEANED-BABI/no-api/task1-tst-plus-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600" --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75 --predict


