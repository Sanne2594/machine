#!/bin/sh

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"

NUM_EP=100 #Number of Epochs
EMB_SIZE=128
H_SIZE=1024

echo "Training on babi"
MDL_LOC="model-task6/"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task6-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task6-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task6-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task6-tst.txt" --max_len 75
#python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task2-tst-dialog.txt" --max_len 75
