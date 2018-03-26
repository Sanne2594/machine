#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=01:00:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"

cd $HOME/machine/sanne/machine


echo "start visualising minplus"
goal_direc="images-attentions-minplus/"
PLS_LOC="model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600"

python3 evaluate.py --checkpoint_path $PLS_LOC --test_data "shorter.txt" --max_len 75 --attviz $goal_direc

echo "start visualising plusplus"
goal_direc="images-attentions-plusplus/"
PLS_LOC="model-final-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s4000"

python3 evaluate.py --checkpoint_path $PLS_LOC --test_data "shorter.txt" --max_len 75 --attviz $goal_direc

echo "Finished"
