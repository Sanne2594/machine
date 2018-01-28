#!/bin/sh

# Start training
#echo "Train model on babi data"
#python3 train_model.py --train "data/CLEANED-BABI/babi-line/task1-trn.txt" --output_dir "model-line" --print_every 50 --embedding_size 128 --hidden_size 128 --epoch 6 --teacher_forcing .5 #--attention
# --dev "data/CLEANED-SCAN/length_split/tasks_test_length.txt"

# Start training
#echo "Train model on babi plus data"
#python3 train_model.py --train "data/CLEANED-BABI/babi+line/task1-trn.txt" --output_dir "model-plus-line" --print_every 50 --embedding_size 128 --hidden_size 128 --epoch 6 --teacher_forcing .5 #--attention
# --dev "data/CLEANED-SCAN/length_split/tasks_test_length.txt"

echo "\nEvaluate babi model on babi data"
python3 evaluate.py --checkpoint_path "model-line/"$(ls -t "model-line/" | head -1) --test_data "data/CLEANED-BABI/babi-line/task1-tst.txt"

echo "\nEvaluate babi plus model on babi plus data"
python3 evaluate.py --checkpoint_path "model-plus-line/"$(ls -t "model-plus-line/" | head -1) --test_data "data/CLEANED-BABI/babi+line/task1-tst.txt"

echo "\nEvaluate babi model on babi plus data"
python3 evaluate.py --checkpoint_path "model-line/"$(ls -t "model-line/" | head -1) --test_data "data/CLEANED-BABI/babi+line/task1-tst.txt"

echo "\nEvaluate babi plus model on babi data"
python3 evaluate.py --checkpoint_path "model-plus-line/"$(ls -t "model-plus-line/" | head -1) --test_data "data/CLEANED-BABI/babi-line/task1-tst.txt"


git remote add upstream https://github.com/i-machine-think/machine.git

git fetch upstream
git checkout master
git rebase upstream/master