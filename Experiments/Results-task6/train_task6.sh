#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=21:00:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt \n"

cd $HOME/machine/sanne/machine

NUM_EP=100 #Number of Epochs
EMB_SIZE=128
H_SIZE=1024

echo "Training on babi"
MDL_LOC="model-task6-1/"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task6-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task6-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task6-tst.txt" --max_len 75


echo "Training on babi"
MDL_LOC="model-task6-2/"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task6-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task6-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task6-tst.txt" --max_len 75


echo "Training on babi"
MDL_LOC="model-task6-3/"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task6-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task6-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task6-tst.txt" --max_len 75


echo "Training on babi"
MDL_LOC="model-task6-4/"
python3 train_model.py --train "data/CLEANED-BABI/babi-dialog/task6-trn.txt" --dev "data/CLEANED-BABI/babi-dialog/task6-dev.txt" --output_dir $MDL_LOC --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Test-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC$(ls -t $MDL_LOC | head -1) --test_data "data/CLEANED-BABI/babi-dialog/task6-tst.txt" --max_len 75
