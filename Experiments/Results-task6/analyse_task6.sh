#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=01:00:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"

cd $HOME/machine/sanne/machine

NUM_EP=100 #Number of Epochs
EMB_SIZE=128
H_SIZE=1024
MDL_LOC="model-task6/acc_0.78_seq_acc_0.47_ppl_2.60_s2300"

echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi-dialog/task6-trn.txt" --max_len 75

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi-dialog/task6-tst.txt" --max_len 75
