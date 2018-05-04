#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:20:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine


NUM_EP=30 #Number of Epochs
EMB_SIZE=128
H_SIZE=500

MDL_LOC="model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600/"

echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi-dialog/task1-trn.txt" --max_len 75


MDL_LOC_PLS="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000/"

echo "Train-scores- plus"
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS --test_data "data/CLEANED-BABI/babi+dialog/task1-trn.txt" --max_len 75


echo " "
echo "All the test sets - OOV"

echo " "
echo "babi on babi"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi-dialog/task1-tst-oov.txt" --max_len 75

echo " "
echo "babi plus on babi plus - test set "
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS --test_data "data/CLEANED-BABI/babi+dialog/task1-tst-oov.txt" --max_len 75
#python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75

echo " "
echo "babi on babi plus"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi+dialog/task1-tst-oov.txt" --max_len 75
#python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75

echo " "
echo "babi plus on babi"
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS --test_data "data/CLEANED-BABI/babi-dialog/task1-tst-oov.txt" --max_len 75


