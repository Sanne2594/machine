#!/usr/bin/env bash
#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:20:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

NUM_EP=100 #Number of Epochs
MDL_LOC="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"

MDL_DC="Model-DC-alt"
M_DATA="data/disfluency-masks/train_alteration-masks.txt"

NUM_CLASS=2
W_VEC=".1,.9"

echo "Training Diagnostic Classifier on Alteration"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC

echo "Testing for Correction"
M_DATA="data/disfluency-masks/correction_test_alteration-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_LOC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

echo "Testing for Restart"
M_DATA="data/disfluency-masks/restart_test_alteration-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_$(ls -t $MDL_LOC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 1 --weight_vec $W_VEC

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "
