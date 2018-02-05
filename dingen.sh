#!/bin/sh

## Start training
#echo "Train model on babi data"
#python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task1-dev.txt" --output_dir "model-dialog" --print_every 200 --max_len 75 --embedding_size 128 --hidden_size 128 --epoch 50 --teacher_forcing .5 --attention
#
## Start training
#echo "Train model on babi plus data"
#python3 train_model.py --train "data/CLEANED-BABI/babi+dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi+dialog/task1-dev.txt" --output_dir "model-plus-dialog" --print_every 200 --max_len 75 --embedding_size 128 --hidden_size 128 --epoch 6 --teacher_forcing .5 --attention
#
#
#echo "\nEvaluate babi model on babi data"
#python3 evaluate.py --checkpoint_path "model-dialog/"$(ls -t "model-dialog/" | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75
#python3 evaluate.py --checkpoint_path "model-dialog/"$(ls -t "model-dialog/" | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
#
#echo "\nEvaluate babi plus model on babi plus data"
#python3 evaluate.py --checkpoint_path "model-plus-dialog/"$(ls -t "model-plus-dialog/" | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75
#python3 evaluate.py --checkpoint_path "model-plus-dialog/"$(ls -t "model-plus-dialog/" | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75
#
#echo "\nEvaluate babi model on babi plus data"
#python3 evaluate.py --checkpoint_path "model-dialog/"$(ls -t "model-dialog/" | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-dialog/"$(ls -t "model-dialog/" | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75 --predict


#echo "\nEvaluate babi model on babi data"
#python3 evaluate.py --checkpoint_path "model-plus-dialog/"$(ls -t "model-plus-dialog/" | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-tst.txt" --max_len 75
#python3 evaluate.py --checkpoint_path "model-plus-dialog/"$(ls -t "model-plus-dialog/" | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75



