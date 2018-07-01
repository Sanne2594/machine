#!/usr/bin/env bash
dt=$(date '+%d/%m/%Y %H:%M:%S');
echo "$dt "


# Correction model
MDL_LOC="Model-DC-cor/acc_0.89_ppl_0.25_s30"
M_DATA="DataGeneration/mask_data/correction-masks.txt"
#W_VEC="1,.9"


# Editing Term Model
#MDL_LOC="Model-DC-ET/acc_0.91_ppl_0.19_s30"
#M_DATA="DataGeneration/mask_data/ET-masks.txt"
#W_VEC=".2,.8"


# Reperandum
#MDL_LOC="Model-DC-rep/acc_0.83_ppl_0.35_s30"
#M_DATA="DataGeneration//mask_data/reperandum-masks.txt"
W_VEC=".9,.1"


# Compute recall and precision
#python3 ../../test-DC.py --checkpoint_path $MDL_LOC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --print_wrong 100 --stats

# Show output
python3 ../../test-DC.py --checkpoint_path $MDL_LOC --mask_data $M_DATA --max_len 75 --batch_size 32 --weight_vec $W_VEC --print_wrong 100


echo "Made it"



