#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=03:00:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

echo "This script will run five different versions of 2 models (10 evaluations) on data without correction"

NUM_EP=30 #Number of Epochs
EMB_SIZE=128
H_SIZE=500

MDL_LOC="model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600/"
MDL_LOC_PLS="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"

echo "tested on restarthesi"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/BABIrestarthesi/clean/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS --test_data "data/BABIrestarthesi/clean/task1-tst.txt" --max_len 75


echo"Again and again"

#echo "Training on babi"
MDL_LOC="model-final-new/acc_1.00_seq_acc_1.00_ppl_1.00_s5670/"
MDL_LOC_PLS="model-final-plus-2/acc_1.00_seq_acc_1.00_ppl_1.00_s4500"

echo "tested on restarthesi"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/BABIrestarthesi/clean/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS --test_data "data/BABIrestarthesi/clean/task1-tst.txt" --max_len 75

echo " "
MDL_LOC="model-final-new2/acc_1.00_seq_acc_1.00_ppl_1.00_s5670/"
MDL_LOC_PLS="model-final-plus-2/acc_1.00_seq_acc_1.00_ppl_1.00_s4800/"

echo "tested on restarthesi"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/BABIrestarthesi/clean/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS --test_data "data/BABIrestarthesi/clean/task1-tst.txt" --max_len 75

echo " "
MDL_LOC="model-final-new3/acc_1.00_seq_acc_1.00_ppl_1.00_s5670/"
MDL_LOC_PLS="model-final-plus-2/acc_1.00_seq_acc_1.00_ppl_1.00_s5040"

echo "tested on restarthesi"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/BABIrestarthesi/clean/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS --test_data "data/BABIrestarthesi/clean/task1-tst.txt" --max_len 75

echo " "
MDL_LOC="model-final-new1/acc_1.00_seq_acc_1.00_ppl_1.00_s5670/"
MDL_LOC_PLS="model-final-plus-2/acc_1.00_seq_acc_1.00_ppl_1.00_s3900"

echo "tested on restarthesi"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/BABIrestarthesi/clean/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS --test_data "data/BABIrestarthesi/clean/task1-tst.txt" --max_len 75
