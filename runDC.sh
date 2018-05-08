#!/bin/sh

NUM_EP=10 #Number of Epochs
EMB_SIZE=128
H_SIZE=128
#MDL_LOC="model-cpu"
MDL_LOC="model-cpu/acc_0.87_seq_acc_0.77_ppl_1.41_s1890"
MDL_DC="Experiments/DiagnosticClassifier/Model/"

M_DATA="Experiments/DiagnosticClassifier/DataGeneration/mask_data/task1-trn.txt"  #task1-trn.txt"
#echo "Train model with cpu"
#python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task1-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention --max_len 75

#python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1

echo "Made it"