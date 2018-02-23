#!/bin/sh

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"

NUM_EP=30 #Number of Epochs
EMB_SIZE=128
H_SIZE=500

#echo "Training on babi"
#MDL_LOC="model-task3/"
#python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task3-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task3-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention
#
#echo " "
#echo "Train-scores"
#python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task3-trn.txt" --max_len 75
#
#echo "Test-scores"
#python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task3-tst.txt" --max_len 75



echo " "
echo " "
echo "Training on babi - task 4"
MDL_LOC="model-task4/"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task4-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task4-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task4-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task4-tst.txt" --max_len 75


echo " "
echo " "
echo "Training on babi - task 5"
MDL_LOC="model-task5/"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task5-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task5-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task5-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task5-tst.txt" --max_len 75
