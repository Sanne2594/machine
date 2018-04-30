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
MDL_LOC="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"

echo "Mistakes bAbI plus on bAbI plus- all"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75 --batch_size 1 --print_wrong 1.0

MDL_LOC="model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600"

echo "Mistakes bAbI on bAbI plus- all"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75 --batch_size 1 --print_wrong 1.0

