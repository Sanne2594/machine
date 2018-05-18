#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:50:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

NUM_EP=10 #Number of Epochs
EMB_SIZE=128
H_SIZE=128
MDL_LOC="model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600"
MDL_LOC_PLS="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"
DAT_DIR="data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt"
OUT_DIR="output/"

#python3 get_coocurrences.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/api-only/task1-tst-dialog.txt" --max_len 75

python3 get_coocurrences.py --checkpoint_path $MDL_LOC_PLS --test_data $DAT_DIT --output_dir $OUT_DIR --max_len 75

echo "Made it"