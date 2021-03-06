#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:50:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

NUM_EP=100 #Number of Epochs
MDL_LOC="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"
#M_DATA="data/Data-loop/train_price_range_masks.txt"
NUM_CLASS=3
W_VEC="1,.9,.9"


M_DATA="data/Data-loop/train_price_range_masks.txt"
MDL_DC="Model-DC-price-0/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC --print_wrong 100
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/train_price_range_masks.txt"
MDL_DC="Model-DC-price-1/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC --print_wrong 100
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/train_price_range_masks.txt"
MDL_DC="Model-DC-price-2/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC --print_wrong 100
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/train_price_range_masks.txt"
MDL_DC="Model-DC-price-3/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC --print_wrong 100
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC

M_DATA="data/Data-loop/train_price_range_masks.txt"
MDL_DC="Model-DC-price-4/"
python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 32 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC --print_wrong 100
python3 test-DC.py --checkpoint_path $MDL_DC$(ls -t $MDL_DC | head -1) --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC
