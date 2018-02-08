import os


path = "/home/sanne/machine/data/CLEANED-BABI/babi-dialog"
new_path = '/home/sanne/machine/data/CLEANED-BABI/api-only'

file = "task1-dev.txt"


def readBabiApi(fpath,api_path):
    """
    Reads in babi dialogs from a file.
    Where the final format is saved into Field variables.

    :param fpath:
    :return data:
    """
    f = open(fpath)
    lines = f.readlines()
    lines2 = [line for line in lines if 'api_call' in line]
    print(lines2[0:3])
    f.close()

    f = open(api_path, 'w')
    for line in lines2:
        f.write(line)
    f.close()

    return

fpath = os.path.join(path,file)
api_path = os.path.join(new_path,file)

readBabiApi(fpath,api_path)
print("Done!")