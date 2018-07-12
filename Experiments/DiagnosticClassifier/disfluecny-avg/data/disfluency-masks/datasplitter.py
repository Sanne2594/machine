import numpy as np


def split_data(trn_file,tst_file):
    #Read in datafile to be split
    f = open(trn_file)
    lines = f.readlines()
    f.close()

    train_lines = []
    train_tot=0
    train_one=0

    #Create training data
    for line in lines:
        human,mask = line.split("\t")
        train_mask = np.fromstring(mask,dtype="float", sep=" ")
        for i,num in enumerate(train_mask):
            if num in [2,3]:
                train_mask[i] = 1

        train_tot += len(train_mask)
        train_one += np.count_nonzero(train_mask)
        train_mask = " ".join([str(i) for i in train_mask])


        newline = human + "\t" + train_mask + "\n"
        train_lines.append(newline)

    train_file = "".join(["train_",trn_file])
    f= open(train_file, "w")
    f.writelines(train_lines)
    f.close()


    f = open(tst_file)
    lines = f.readlines()
    f.close()

    correction_test = []
    restart_test = []
    test_tot = 0
    cor_one=0
    res_one=0
    cor_min_one=0
    res_min_one=0

    for line in lines:
        human, mask = line.split("\t")
        correction_test_mask = np.fromstring(mask, dtype="float", sep=" ")
        restart_test_mask = np.fromstring(mask, dtype="float", sep=" ")

        for i, num in enumerate(correction_test_mask):
            if num == 2:
                correction_test_mask[i] = -1
                restart_test_mask[i] = 1
            elif num == 3:
                correction_test_mask[i] = -1
                restart_test_mask[i] = -1
            elif num == 1:
                restart_test_mask[i] = -1

        test_tot += len(correction_test_mask)
        cor_one += np.count_nonzero(correction_test_mask==1)
        cor_min_one += np.count_nonzero(correction_test_mask==-1)
        res_one += np.count_nonzero(restart_test_mask==1)
        res_min_one += np.count_nonzero(restart_test_mask==-1)

        correction_test_mask = " ".join([str(i) for i in correction_test_mask])
        restart_test_mask = " ".join([str(i) for i in restart_test_mask])

        newline = human + "\t" + correction_test_mask + "\n"
        correction_test.append(newline)
        newline = human + "\t" + restart_test_mask + "\n"
        restart_test.append(newline)

    test_correction = "".join(["correction_", tst_file])
    test_restart = "".join(["restart_", tst_file])

    f = open(test_correction, "w")
    f.writelines(correction_test)
    f.close()

    f = open(test_restart, "w")
    f.writelines(restart_test)
    f.close()

    print(trn_file.split(".")[0])
    print("train stats:", train_one/train_tot)
    print("test correction stats(1, -1): ", cor_one/test_tot, cor_min_one/test_tot)
    print("test restart stats(1, -1): ", res_one / test_tot, res_min_one/ test_tot, "\n")
    return

split_data("alteration-masks.txt", "tst_alteration-masks.txt")
split_data("ET-masks.txt", "tst_ET-masks.txt")
split_data("reperandum-masks.txt", "tst_reperandum-masks.txt")