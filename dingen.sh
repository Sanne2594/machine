#!/bin/sh

NUM_EP=15 #Number of Epochs

EMB_SIZE=128
H_SIZE=500

#echo "Train model on api calls:"
#python3 train_model.py --train "data/CLEANED-BABI/api-only/task1-trn-dialog.txt" --dev "data/CLEANED-BABI/api-only/task1-dev-dialog.txt" --output_dir "model-api-first" --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention
#
echo "Train model on full dialogs:"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task1-dev.txt" --output_dir "model-duality" --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

#echo "Evaluate model:"
#python3 evaluate.py --checkpoint_path "model-duality/"$(ls -t "model-duality/" | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
#
#NUM_EP=50
#
#echo "Retrain model on api-calls:"
#python3 train_model.py --train "data/CLEANED-BABI/api-only/task1-trn-dialog.txt" --dev "data/CLEANED-BABI/api-only/task1-dev-dialog.txt" --output_dir "model-duality" --print_every 200 --max_len 75 --epoch $NUM_EP --load_checkpoint $(ls -t "model-duality/" | head -1)

#echo "Retrain model on full dialogs:"
#python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task1-dev.txt" --output_dir "model-api-first" --print_every 200 --max_len 75 --epoch $NUM_EP --load_checkpoint $(ls -t "model-api-first/" | head -1)

echo "Evaluate model:"
python3 evaluate.py --checkpoint_path "model-duality/acc_1.00_seq_acc_1.00_ppl_1.00_s2600" --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75

echo "Evaluate model on full dialogs:"
python3 evaluate.py --checkpoint_path "model-duality/acc_1.00_seq_acc_1.00_ppl_1.00_s2600" --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75
