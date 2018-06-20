import os
import numpy as np

in_dir = "../../../data/CLEANED-BABI/babi+dialog/"
in_file= "task1-trn.txt"

out_dir = "mask_data"
efile = "ET-masks.txt"
rfile = "reperandum-masks.txt"
cfile = "correction-masks.txt"

path = os.path.join(in_dir,in_file)
f = open(path, "r")
lines = f.readlines()
f.close()

edit = ["uhm sorry", "sorry", "oh no", "no sorry", "no "]
#restart = ["sorry yeah", "uhm yeah"]
#hesi = ["uhm","well"]
slots = ['italian',  'spanish', 'indian',  'french',  'british',   'korean',   'thai',  'cantonese',  'vietnamese',
         'japanese',  'paris',  'bombay',  'rome',  'london',  'madrid',   'seoul', 'tokyo',  'beijing', 'bangkok',
         'hanoi',  'four',  'six',  'eight',  'two',  'moderate',  'cheap',  'expensive']

ET = []  # Editing term masks
rep = [] # Reperandum masks
cor = [] # Correction masks

for line in lines:
    human,bot = line.split("\t")
    mask = np.zeros(len(human.split()))
    change = False
    for ed in edit:
        if ed in line:
            change = True
    if change is True:
        human = human.split()
        emask = np.zeros(len(human))
        cmask = np.zeros(len(human))
        rmask = np.zeros(len(human))
        #Find editing term
        si = [i for i,word in enumerate(human) if word in ["sorry", "no"] and not human[i+1] in ["yeah", "sorry"] and not human[i-1] in ["uhm", "no", "oh"]]
        di = [i for i,word in enumerate(human) if word in ["uhm", "no", "oh"] and human[i+1] in ["sorry", "no"]]
        for i in si:
            if(not human[i-1] in slots):
                corb = [i+ind for ind,word in enumerate(human[i:]) if word == human[i-1]][0]
            else:
                #Else the final word of the correction is the slot word.
                corb = [ind for ind,word in enumerate(human[i:]) if word in slots][0]

            if (not human[i+1] in slots):
                rorb = [i + ind for ind, word in enumerate(human[:i]) if word == human[i+1]][-1]
            else:
                rorb = [ind for ind,word in enumerate(human[:i]) if word in slots][-1]
            emask[i] = 1
            cmask[i:corb] = 1
            rmask[rorb:i] =1

        for i in di:
            if(not human[i-1] in slots):
                corb = [i+ind for ind,word in enumerate(human[i:]) if word == human[i-1]][0]
            else:
                #Else the final word of the correction is the slot word.
                corb = [i+ind for ind,word in enumerate(human[i:]) if word in slots][0]
            if (not human[i+2] in slots):
                rorb = [ind for ind,word in enumerate(human[:i]) if (word == human[i+2])][-1]
            else:
                rorb = [ind for ind,word in enumerate(human[:i]) if word in slots][-1]
            emask[i] = 1
            emask[i+1] = 1
            cmask[i+2:corb+1] = 1
            rmask[rorb:i] = 1

        human = " ".join(human)
        emask =  " ".join([str(i) for i in emask])
        rmask =  " ".join([str(i) for i in rmask])
        cmask =  " ".join([str(i) for i in cmask])
        #create the masks
        newline = human + "\t" + emask + "\n"
        ET.append(newline)
        newline = human + "\t" + rmask + "\n"
        rep.append(newline)
        newline = human + "\t" + cmask + "\n"
        cor.append(newline)

    else:
        nmask = " ".join([str(i) for i in mask])
        newline = human +  "\t" + nmask + "\n"
        ET.append(newline)
        rep.append(newline)
        cor.append(newline)

path = os.path.join(out_dir,efile)
f = open(path, "w")
f.writelines(ET)
f.close()

path = os.path.join(out_dir,rfile)
f = open(path, "w")
f.writelines(rep)
f.close()


path = os.path.join(out_dir,cfile)
f = open(path, "w")
f.writelines(cor)
f.close()

