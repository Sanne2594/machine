#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:50:00 (reservation time)


dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

cd $HOME/machine/sanne/machine

module load python/3.5.0
module load eb
module load cuda/9.0.176

pip3 install --user -r requirements.txt
pip3 install --user http://download.pytorch.org/whl/cu90/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
pip3 install --user numpy matplotlib

echo "babi on babi plus"
python3 evaluate.py --checkpoint_path "model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600" --test_data "data/CLEANED-BABI/babi+dialog/task1-tst.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600" --test_data "data/CLEANED-BABI/no-api/task1-tst-plus-dialog.txt" --max_len 75
python3 evaluate.py --checkpoint_path "model-final/acc_1.00_seq_acc_1.00_ppl_1.00_s4600" --test_data "data/CLEANED-BABI/api-only/task1-tst-plus-dialog.txt" --max_len 75 --predict


