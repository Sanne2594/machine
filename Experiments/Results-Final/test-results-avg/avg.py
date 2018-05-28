import os
import numpy as np

files = os.listdir()

files = [file for file in files if ".o" in file]
att = [file for file in files if "final" in file]
noatt = [file for file in files if "noatt" in file]

fullw = []
fulls = []
now = []
nos = []
apiw = []
apis = []
for file in noatt:
    print(file)
    f = open(file, "r")
    lines = f.readlines()
    f.close()
    lines = [line for line in lines if "Loss" in line]
    print(len(lines))
    # first two are train loss for babi and babi+

    # then 4 x 3 accuracies:
    for i in [2,5,8,11]:
        t = lines[i].split()
        fullw.append(t[4])
        fulls.append(t[7])
        t = lines[i+1].split()
        now.append(t[4])
        nos.append(t[7])
        t = lines[i+2].split()
        apiw.append(t[4])
        apis.append(t[7])

print("Average accuracies:")

print(fulls)
fulls = np.array(fulls).sum()/len(fulls)
fullw = np.array(fullw).sum()/len(fullw)
nos = np.array(nos).sum()/len(nos)
now = np.array(now).sum()/len(now)
apis = np.array(apis).sum()/len(apis)
apiw = np.array(apiw).sum()/len(apiw)

print("full acc: ", fulls, "(", fullw, ")")
print("no api acc: ", nos, "(", now, ")")
print("api only acc: ", apis, "(", apiw, ")")

