#!/usr/bin/env bash
# Runs on laptop

broot="../data/BABIraw"
proot="../data/OWNpls"

python3 ../data/babi_tools-master/babi_plus.py $broot $proot --result_size 1000
