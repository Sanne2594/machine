#!/bin/sh

NUM_EP=20 #Number of Epochs
EMB_SIZE=128

echo "Make the table of Lemon"

H_SIZE=500

echo "start training babi"
#MDL_LOC="model-comp/"
#python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task1-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "start training babi plus"
MDL_LOC_PLS="model-comp-plus/"
python3 train_model.py --train "data/CLEANED-BABI/babi+dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi+dialog/task1-dev.txt" --output_dir $MDL_LOC_PLS --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo " \nStart experiments "

echo "babi on babi"
python3 evaluate.py --checkpoint_path "model-comp/acc_1.00_seq_acc_1.00_ppl_1.00_s3780" --test_data "data/CLEANED-BABI/babi-dialog/task1-dev.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-comp/acc_1.00_seq_acc_1.00_ppl_1.00_s3780" --test_data "data/CLEANED-BABI/api-only/task1-dev-dialog.txt" --max_len 75

echo "babi plus on babi plus"
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-dev.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/api-only/task1-dev-plus-dialog.txt" --max_len 75

echo "babi on babi plus"
python3 evaluate.py --checkpoint_path "model-comp/acc_1.00_seq_acc_1.00_ppl_1.00_s3780" --test_data "data/CLEANED-BABI/babi+dialog/task1-dev.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-comp/acc_1.00_seq_acc_1.00_ppl_1.00_s3780" --test_data "data/CLEANED-BABI/api-only/task1-dev-plus-dialog.txt" --max_len 75

echo "babi plus on babi"
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task1-dev.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/api-only/task1-dev-dialog.txt" --max_len 75


