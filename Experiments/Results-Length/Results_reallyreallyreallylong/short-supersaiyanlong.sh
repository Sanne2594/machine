#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:50:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

NUM_EP=30 #Number of Epochs
EMB_SIZE=128
H_SIZE=500

MDL_LOC="model-short/acc_1.00_seq_acc_1.00_ppl_1.00_s4100"
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/BABI_short/cleaned/task1-trn.txt" --max_len 75

echo "short on short - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/BABI_short/cleaned/task1-tst.txt" --max_len 75

echo "short on reallylong - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/BABI_saiyanlong/cleaned/task1-tst.txt" --max_len 75
