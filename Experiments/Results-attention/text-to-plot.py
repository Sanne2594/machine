from os import listdir
from os.path import isfile, join

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    return fig

goal_folder = "RESIZED/plusplus/"
folder = "text-plusplus"
onlyfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
count = 0

for file in onlyfiles:
    f = open(file, 'r')
    data = f.readlines()
    input = data[0].split()
    output = data[1].split()
    ndim, dim0, dim1 = data[2].split()
    attentions = [float(i) for i in data[3].split()]
    attentions = np.asarray(attentions)
    attentions = attentions.reshape(int(dim0),int(dim1))
    f.close()

    #Generate figure, change sizing and create plot
    fig = showAttention(input,output,attentions)

    DefaultSize = fig.get_size_inches()
    fig.set_size_inches(DefaultSize[0]*2, DefaultSize[1]*2)

    fig_loc = goal_folder + "attn" + str(count) + ".png"
    count += 1
    fig.savefig(fig_loc)