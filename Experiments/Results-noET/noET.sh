#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:50:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

MDL_LOC="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"
echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi+dialog/task1-trn.txt" --max_len 75

echo "model on model - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75

echo "model on noET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/noET/cleaned/task1-tst.txt" --max_len 75
