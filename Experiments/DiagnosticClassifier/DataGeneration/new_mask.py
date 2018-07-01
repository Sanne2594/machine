import os
import numpy as np

in_dir = "../../../data/CLEANED-BABI/babi+dialog/"
in_file= "task1-trn.txt"

out_dir = "disfluency-masks"
efile = "ET-masks.txt"
rfile = "reperandum-masks.txt"
afile = "alteration-masks.txt"

path = os.path.join(in_dir,in_file)
f = open(path, "r")
lines = f.readlines()
f.close()

slots = ['italian',  'spanish', 'indian',  'french',  'british',   'korean',   'thai',  'cantonese',  'vietnamese',
         'japanese',  'paris',  'bombay',  'rome',  'london',  'madrid',   'seoul', 'tokyo',  'beijing', 'bangkok',
         'hanoi',  'four',  'six',  'eight',  'two',  'moderate',  'cheap',  'expensive']

edit = ["uhm sorry", "sorry", "oh no", "no sorry", "no ","sorry yeah", "uhm yeah", "uhm","well"]


ET = []  # Editing term masks
rep = [] # Reperandum masks
alt = [] # Alteration masks

"""
For every line of the dialog, find all disfluencies and annotate them.
Detects each disfluency-type seperately, by their editing term.
Based on the editing term and patterns determine the reparandum and alteration. 
Save all in the same mask with different numbers. 
Where Correction = 1; Restart = 2; and Hesitation = 3

Patterns:
correction = ["uhm sorry", "sorry", "oh no", "no sorry", "no "]
restart = ["sorry yeah", "uhm yeah"]
hesitation = ["uhm","well"]
"""

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
        amask = np.zeros(len(human))
        rmask = np.zeros(len(human))

        #Find hesitations
        hes_i = [i for i,word in enumerate(human) if word in ["uhm", "well"] and not human[i+1] in ["yeah", "sorry"]]
        for i in hes_i:
            emask[i] = 3

        #Find restarts:
        restart_i = [i for i,word in enumerate(human) if word in ["sorry", "uhm"] and human[i+1] == "yeah"]
        for i in restart_i:
            #TODO: rep_bound may always be zero
            rep_bound  = [ind for ind, word in enumerate(human[:i]) if word == human[i+2]][-1]
            alt_bound = [i+ind for ind,word in enumerate(human[i:]) if word == human[i-1]][0]
            emask[i] = 2
            emask[i+1] = 2
            amask[i+2:alt_bound+1] = 2
            rmask[rep_bound:i] = 2

        #Find editing term
        cor_i_one = [i for i,word in enumerate(human) if word in ["sorry", "no"] and not human[i+1] in ["yeah", "sorry"] and not human[i-1] in ["uhm", "no", "oh"]]
        cor_i_two = [i for i,word in enumerate(human) if word in ["uhm", "no", "oh"] and human[i+1] in ["sorry", "no"]]
        for i in cor_i_one:
            if(not human[i-1] in slots):
                alt_bound = [i+ind for ind,word in enumerate(human[i:]) if word == human[i-1]][0]
            else:
                #Else the final word of the correction is the slot word.
                alt_bound = [i+ind for ind,word in enumerate(human[i:]) if word in slots][0]

            if (not human[i+1] in slots):
                rep_bound = [ind for ind, word in enumerate(human[:i]) if word == human[i+1]][-1]
            else:
                rep_bound = [ind for ind,word in enumerate(human[:i]) if word in slots][-1]
            emask[i] = 1
            amask[i:alt_bound] = 1
            rmask[rep_bound:i] =1

        for i in cor_i_two:
            if(not human[i-1] in slots):
                alt_bound = [i+ind for ind,word in enumerate(human[i:]) if word == human[i-1]][0]
            else:
                #Else the final word of the correction is the slot word.
                alt_bound = [i+ind for ind,word in enumerate(human[i:]) if word in slots][0]
            if (not human[i+2] in slots):
                rep_bound = [ind for ind,word in enumerate(human[:i]) if (word == human[i+2])][-1]
            else:
                rep_bound = [ind for ind,word in enumerate(human[:i]) if word in slots][-1]
            emask[i] = 1
            emask[i+1] = 1
            amask[i+2:alt_bound+1] = 1
            rmask[rep_bound:i] = 1

        human = " ".join(human)
        emask =  " ".join([str(i) for i in emask])
        rmask =  " ".join([str(i) for i in rmask])
        amask =  " ".join([str(i) for i in amask])
        #create the masks
        newline = human + "\t" + emask + "\n"
        ET.append(newline)
        newline = human + "\t" + rmask + "\n"
        rep.append(newline)
        newline = human + "\t" + amask + "\n"
        alt.append(newline)

    else:
        nmask = " ".join([str(i) for i in mask])
        newline = human +  "\t" + nmask + "\n"
        ET.append(newline)
        rep.append(newline)
        alt.append(newline)

path = os.path.join(out_dir,efile)
f = open(path, "w")
f.writelines(ET)
f.close()

path = os.path.join(out_dir,rfile)
f = open(path, "w")
f.writelines(rep)
f.close()


path = os.path.join(out_dir,afile)
f = open(path, "w")
f.writelines(alt)
f.close()

