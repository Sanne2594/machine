#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:50:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

NUM_EP=100 #Number of Epochs
MDL_LOC="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"
#MDL_DC="Model-DC-size-loop/"

#M_DATA="data/Data-loop/train_party_size_masks.txt"
NUM_CLASS=4
W_VEC="1,1,1,1"

#python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC

echo " "
echo "Testing Sequence"

M_DATA="data/Data-loop/test0_party_size_masks.txt"
MDL_DC="Model-DC-size-0/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test1_party_size_masks.txt"
MDL_DC="Model-DC-size-1/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test2_party_size_masks.txt"
MDL_DC="Model-DC-size-2/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test3_party_size_masks.txt"
MDL_DC="Model-DC-size-3/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test4_party_size_masks.txt"
MDL_DC="Model-DC-size-4/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test5_party_size_masks.txt"
MDL_DC="Model-DC-size-5/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test6_party_size_masks.txt"
MDL_DC="Model-DC-size-6/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/test7_party_size_masks.txt"
MDL_DC="Model-DC-size-7/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC




