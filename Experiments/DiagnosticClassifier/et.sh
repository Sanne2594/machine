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

NUM_CLASS=2
W_VEC=".1,.9"

MDL_DC="Model-DC-ET/acc_0.94_ppl_0.12_s100"
#M_DATA="data/disfluency-masks/train_ET-masks.txt"
#
#echo "Training Diagnostic Classifier on Editing Term"
#python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC

echo "Testing for Correction"
M_DATA="data/disfluency-masks/correction_tst_ET-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --stats


echo "Testing for Restart"
M_DATA="data/disfluency-masks/restart_tst_ET-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 1 --weight_vec $W_VEC --stats

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

MDL_DC="Model-DC-ET-1/acc_0.94_ppl_0.12_s100"
#M_DATA="data/disfluency-masks/train_ET-masks.txt"
#
#echo "Training Diagnostic Classifier on Editing Term"
#python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC

echo "Testing for Correction"
M_DATA="data/disfluency-masks/correction_tst_ET-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --stats


echo "Testing for Restart"
M_DATA="data/disfluency-masks/restart_tst_ET-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --weight_vec $W_VEC --stats

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

MDL_DC="Model-DC-ET-2/acc_0.94_ppl_0.12_s100"
#M_DATA="data/disfluency-masks/train_ET-masks.txt"
#
#echo "Training Diagnostic Classifier on Editing Term"
#python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC

echo "Testing for Correction"
M_DATA="data/disfluency-masks/correction_tst_ET-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --stats


echo "Testing for Restart"
M_DATA="data/disfluency-masks/restart_tst_ET-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --weight_vec $W_VEC --stats

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

MDL_DC="Model-DC-ET-3/acc_0.94_ppl_0.11_s100"
#M_DATA="data/disfluency-masks/train_ET-masks.txt"
#
#echo "Training Diagnostic Classifier on Editing Term"
#python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC

echo "Testing for Correction"
M_DATA="data/disfluency-masks/correction_tst_ET-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --stats


echo "Testing for Restart"
M_DATA="data/disfluency-masks/restart_tst_ET-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --weight_vec $W_VEC --stats

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

MDL_DC="Model-DC-ET-4/acc_0.94_ppl_0.12_s100"
#M_DATA="data/disfluency-masks/train_ET-masks.txt"
#
#echo "Training Diagnostic Classifier on Editing Term"
#python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC

echo "Testing for Correction"
M_DATA="data/disfluency-masks/correction_tst_ET-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --stats


echo "Testing for Restart"
M_DATA="data/disfluency-masks/restart_tst_ET-masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --weight_vec $W_VEC --stats

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "