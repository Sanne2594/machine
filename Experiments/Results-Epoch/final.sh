#!/bin/bash

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"

NUM_EP=3 #Number of Epochs: Don't change!!! unless you implement a for-loop first
EMB_SIZE=128
H_SIZE=500

echo " "
echo " "
echo "1-3"

echo "Training on babi"
MDL_LOC="model-epochs/"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task2-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --predict

echo " "
echo " "
echo "4-6"

echo "Training on babi"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task2-dev.txt" --output_dir $MDL_LOC --load_checkpoint $(ls -t $MDL_LOC | head -1) --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --predict

echo " "
echo " "
echo "7-9"

echo "Training on babi"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task2-dev.txt" --output_dir $MDL_LOC --load_checkpoint $(ls -t $MDL_LOC | head -1) --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --predict

echo " "
echo " "
echo "10-12"

echo "Training on babi"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task2-dev.txt" --output_dir $MDL_LOC --load_checkpoint $(ls -t $MDL_LOC | head -1) --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --predict


echo " "
echo " "
echo "13-15"

echo "Training on babi"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task2-dev.txt" --output_dir $MDL_LOC --load_checkpoint $(ls -t $MDL_LOC | head -1) --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --predict


echo " "
echo " "
echo "16-18"

echo "Training on babi"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task2-dev.txt" --output_dir $MDL_LOC --load_checkpoint $(ls -t $MDL_LOC | head -1) --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --predict

echo " "
echo " "
echo "19-21"

echo "Training on babi"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task2-dev.txt" --output_dir $MDL_LOC --load_checkpoint $(ls -t $MDL_LOC | head -1) --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " "
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75 --predict
