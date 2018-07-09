#!/usr/bin/env bash
# Runs on laptop

broot="../data/BABIraw"
#proot="../data/OWNpls"
proot="../data/BABIrestarthesi/"

python3 ../data/babi_tools-master/babi_plus.py $broot $proot --result_size 1000
#TODO: for some reason mask's get interupted with \n after more than 17 numbers.

