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

echo "Training on realET"
MDL_LOC="model-realET-2/"
#python3 train_model.py --train "data/BABI_midlong/cleaned/task1-trn.txt" --dev "data/realET/cleaned/task1-trn.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/realET/cleaned/task1-trn.txt" --max_len 75

echo "realET on realET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/realET/cleaned/task1-tst.txt" --max_len 75

echo "realET on fullET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75

echo "realET on noET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/noET/cleaned/task1-tst.txt" --max_len 75


echo " "
echo " somehting else"

echo "Training on realET"
MDL_LOC="model-realET-3/"
#python3 train_model.py --train "data/BABI_midlong/cleaned/task1-trn.txt" --dev "data/realET/cleaned/task1-trn.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/realET/cleaned/task1-trn.txt" --max_len 75

echo "realET on realET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/realET/cleaned/task1-tst.txt" --max_len 75

echo "realET on fullET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75

echo "realET on noET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/noET/cleaned/task1-tst.txt" --max_len 75

echo " "
echo " somehting else"

echo "Training on realET"
MDL_LOC="model-realET-4/"
#python3 train_model.py --train "data/BABI_midlong/cleaned/task1-trn.txt" --dev "data/realET/cleaned/task1-trn.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/realET/cleaned/task1-trn.txt" --max_len 75

echo "realET on realET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/realET/cleaned/task1-tst.txt" --max_len 75

echo "realET on fullET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75

echo "realET on noET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/noET/cleaned/task1-tst.txt" --max_len 75

echo " "
echo " somehting else"

echo "Training on realET"
MDL_LOC="model-realET-5/"
#python3 train_model.py --train "data/BABI_midlong/cleaned/task1-trn.txt" --dev "data/realET/cleaned/task1-trn.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/realET/cleaned/task1-trn.txt" --max_len 75

echo "realET on realET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/realET/cleaned/task1-tst.txt" --max_len 75

echo "realET on fullET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75

echo "realET on noET - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/noET/cleaned/task1-tst.txt" --max_len 75
