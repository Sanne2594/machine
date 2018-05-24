import numpy as np

# Locations for average length computation
trn = "../data/BABIraw/dialog-babi-task1-API-calls-trn.txt"
tst = "../data/BABIraw/dialog-babi-task1-API-calls-tst.txt"
trnpls = "../data/BABI+raw/babi_plus/dialog-babi-task1-API-calls-trn.txt"
tstpls = "../data/BABI+raw/babi_plus/dialog-babi-task1-API-calls-tst.txt"

# Compute average length.
def compute_length(data):
    f = open(data, "r")
    lines = f.readlines()
    f.close()
    lines = [line.split("\t")[0] for line in lines if not line is "\n"]

    length = []
    for line in lines:
        length.append(len(line.split()))

    avg_len = np.array(length).sum()/len(length)
    return avg_len


#Compute frequencies

f = open(trnpls, "r")
lines = f.readlines()
f.close()
lines = [line.split("\t")[0] for line in lines if not line is "\n"]

cheap=0
c_nex = [0,0]
mod = 0
m_nex = [0,0]
exp = 0
e_nex = [0,0]
for line in lines:
    if "cheap" in line:
        cheap += 1
        if "moderate" in line:
            temp = line.split()
            if temp.index("cheap") < temp.index("moderate"):
                c_nex[0] += 1
            else:
                c_nex[1] += 1
        elif "expensive" in line:
            temp = line.split()
            if temp.index("cheap") < temp.index("expensive"):
                c_nex[0] += 1
            else:
                c_nex[1] += 1
    if "moderate" in line:
        mod += 1
        if "cheap" in line:
            temp = line.split()
            if temp.index("moderate") < temp.index("cheap"):
                m_nex[0] += 1
            else:
                m_nex[1] += 1
        elif "expensive" in line:
            temp = line.split()
            if temp.index("moderate") < temp.index("expensive"):
                m_nex[0] += 1
            else:
                m_nex[1] += 1
    if "expensive" in line:
        exp += 1
        if "moderate" in line:
            temp = line.split()
            if temp.index("expensive") < temp.index("moderate"):
                e_nex[0] += 1
            else:
                e_nex[1] += 1
        elif "cheap" in line:
            temp = line.split()
            if temp.index("expensive") < temp.index("cheap"):
                e_nex[0] += 1
            else:
                e_nex[1] += 1
print("\nCheap things:", cheap, c_nex)


print("\nmoderate things", mod, m_nex)

print("\nexpensive things", exp, e_nex)

    #
    # if "cheap" in line:
    #     cheap += 1
    #     if "moderate" or "expensive" in line:
    #         c_nex.append(line)