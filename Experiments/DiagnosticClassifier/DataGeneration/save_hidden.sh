#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=02:00:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

MDL_LOC="model-diag-plus/acc_1.00_seq_acc_1.00_ppl_1.00_s3200"
#Somehow run the model on the inputs word by word and save the [hidden layer] + mask of that word.

