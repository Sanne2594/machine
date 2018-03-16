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

echo "Training on babi+"
MDL_LOC_PLS="model-retest-plus/"
python3 train_model.py --train "data/OWNPLS/cleaned/task1-trn.txt" --dev "data/OWNPLS/cleaned/task1-dev.txt" --output_dir $MDL_LOC_PLS --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/OWNPLS/cleaned/task1-trn.txt" --max_len 75

echo "babi plus on babi plus - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/OWNPLS/cleaned/task1-tst.txt" --max_len 75
#python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75
