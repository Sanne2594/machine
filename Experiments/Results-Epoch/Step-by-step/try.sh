#!/bin/bash

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"

NUM_EP=5 #Number of Epochs: Don't change!!! unless you implement a for-loop first
EMB_SIZE=128
H_SIZE=500


echo "Training on babi"
MDL_LOC="model-epochs-2/"
#python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task2-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --predict

#
#model_loc="acc_0.26_seq_acc_0.11_ppl_49.46_s100"
#
#echo "ReTraining on babi until 5th epch"
#MDL_LOC="model-epochs-2/"
#python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task2-dev.txt" --output_dir $MDL_LOC --load_checkpoint $model_loc --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention --resume
#
#echo " "
#echo "Train-scores"
#python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --max_len 75
#
#echo "Test-scores"
#python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
#python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --predict
