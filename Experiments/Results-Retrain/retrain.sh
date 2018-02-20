#!/bin/sh

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"

EMB_SIZE=128
H_SIZE=500
MDL_LOC="model-retrain"

babi_model="model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600"

echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-trn.txt" --max_len 75
echo "Test scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75

echo " "

NUM_EP=1 #Number of Epochs
echo "Training on babi+"
python3 train_model.py --train "data/CLEANED-BABI/babi+dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi+dialog/task1-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention --checkpoint $babi_model




echo "\nAll the test sets"

echo "\nbabi on babi"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75

echo "\nbabi plus on babi plus - test set "
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75



echo "\nbabi plus on babi"
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75


