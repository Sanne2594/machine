#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:09:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

echo "Training Sequence"

NUM_EP=70 #Number of Epochs
MDL_LOC="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"

MDL_DC="Model-DC-cuis/"
M_DATA="data/Data-loop/train_cuisine_masks.txt"

NUM_CLASS=10
W_VEC=".8,.8,.8,.8,.8,.8,.8,.8,.8,.8"

#python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC

echo "Testing Sequence"

M_DATA="data/Data-loop/test0_cuisine_masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test1_cuisine_masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test2_cuisine_masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test3_cuisine_masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test4_cuisine_masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test5_cuisine_masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test6_cuisine_masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test7_cuisine_masks.txt"
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "
