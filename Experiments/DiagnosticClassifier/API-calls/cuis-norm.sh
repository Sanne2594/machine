#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=05:00:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

NUM_EP=100 #Number of Epochs
MDL_LOC="model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600/"

MDL_DC="Model-DC-norm-cuis"
M_DATA="data/Data-api-forcing/cuisine_masks.txt"

NUM_CLASS=10
W_VEC=".8,.8,.8,.8,.8,.2,.2,.2,.2,.2"

python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC


