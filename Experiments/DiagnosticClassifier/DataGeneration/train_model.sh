#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=02:00:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

NUM_EP=30 #Number of Epochs
EMB_SIZE=128
H_SIZE=500

train_path="/home/sanne/machine/Experiments/DiagnosticClassifier/DataGeneration/cleaned_data/task1-trn.txt"
dev_path="/home/sanne/machine/Experiments/DiagnosticClassifier/DataGeneration/cleaned_data/task1-dev.txt"
test_path="/home/sanne/machine/Experiments/DiagnosticClassifier/DataGeneration/cleaned_data/task1-tst.txt"

echo "Training on babi+"
MDL_LOC_PLS="model-diag-plus/"
python3 train_model.py --train $train_path --dev $dev_path --output_dir $MDL_LOC_PLS --print_every 200 --max_len 75 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --epoch $NUM_EP --teacher_forcing .5 --attention

echo "Train-scores"
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data $train_path --max_len 75

echo "babi plus on babi plus - test set"
python3 evaluate.py --checkpoint_path $MDL_LOC_PLS$(ls -t $MDL_LOC_PLS | head -1) --test_data $test_path --max_len 75
