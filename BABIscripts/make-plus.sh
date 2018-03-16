#!/usr/bin/env bash
# Runs on laptop

broot="../data/BABIraw"
proot="../data/BABI_long"

python3 ../data/babi_tools-master/babi_plus.py $broot $proot --result_size 1000