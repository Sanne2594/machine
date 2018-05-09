import os

data_in = "../DataGeneration/mask_data/"
in_file = "task1-trn.txt"

def strip_API(fpath):
    f = open(fpath)
    lines = f.readlines()
    f.close()

    lines_api = [line for line in lines if 'api_call' in line]
    return lines_api


fpath = os.path.join(data_in,in_file)
api_only = strip_API(fpath)
cuisine_list = {1:'italian'}

for line in api_only:
    sent = line.split('\t')[0].split()





