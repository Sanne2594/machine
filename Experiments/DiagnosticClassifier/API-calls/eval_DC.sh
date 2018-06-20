#!/usr/bin/env bash
#PBS -S /bin/bash (which shell to use)
#PBS -lnodes=1 (number of nodes)
#PBS -qgpu (we need gpu)
#PBS -lwalltime=00:50:00 (reservation time)

dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "

#cd $HOME/machine/sanne/machine

MDL_LOC="Model-DC-nomem-cuis-4/acc_1.00_ppl_0.09_s100"
#MDL_LOC="Model-DC-price/acc_0.57_ppl_0.85_s100"


M_DATA="Data-nomem/4_cuisine_masks.txt"
#M_DATA="Data-1/price_range_masks.txt"
W_VEC=".8,.8,.8,.8,.8,.8,.8,.8,.8,.8"
#NUM_CLASS=3
#W_VEC=".9,.9,.9"

#python3 extract.py --checkpoint_path $MDL_LOC --output_dir $MDL_DC --mask_data $M_DATA --max_len 75 --batch_size 1 --epochs $NUM_EP --num_class $NUM_CLASS --weight_vec $W_VEC --print_wrong 100
python3 ../../../test-DC.py --checkpoint_path $MDL_LOC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --print_wrong .9


echo "Made it"



