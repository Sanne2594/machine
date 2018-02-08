#!/bin/sh

NUM_EP=15 #Number of Epochs

EMB_SIZE=128
H_SIZE=500

echo "Train model on babi data task 1"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task1-dev.txt" --output_dir "model-final" --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Train model on babi plus data task 1"
python3 train_model.py --train "data/CLEANED-BABI/babi+dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi+dialog/task1-dev.txt" --output_dir "model-final-plus" --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention


echo "\nEvaluate babi model on babi data"
python3 evaluate.py --checkpoint_path "model-dialog/"$(ls -t "model-dialog/" | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-dialog/"$(ls -t "model-dialog/" | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75

echo "\nEvaluate babi plus model on babi plus data"
python3 evaluate.py --checkpoint_path "model-plus-dialog/"$(ls -t "model-plus-dialog/" | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-plus-dialog/"$(ls -t "model-plus-dialog/" | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75

echo "\nEvaluate babi model on babi plus data"
python3 evaluate.py --checkpoint_path "model-dialog/"$(ls -t "model-dialog/" | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-dialog/"$(ls -t "model-dialog/" | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75

echo "\nEvaluate babi plus model on babi data"
python3 evaluate.py --checkpoint_path "model-plus-dialog/"$(ls -t "model-plus-dialog/" | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-plus-dialog/"$(ls -t "model-plus-dialog/" | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75

