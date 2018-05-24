#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:50:00 (reservation time)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "
MDL_LOC="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"

cd $HOME/machine/sanne/machine

echo "babi plus on babi plus"
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/no-api/task1-tst-plus-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path $MDL_LOC --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75


