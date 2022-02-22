import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import save_stats
from joblib import load
import glob
import os

sts_files = glob.glob(os.path.join('./sts_arr_temporalbp_MSCNthentemp_data/','*.z'))
unique_contents = list(set([os.path.splitext(os.path.basename(fsts))[0].split('_')[0] for fsts in sts_files]))
print(unique_contents)

fps_full = ['24fps','30fps','60fps','82fps','98fps','120fps']
color_list = ['b','g','r','c','m','y','k']
for des_content in unique_contents:
    plt.figure()
    plt.clf()
    data_list = []
    fps_list = []
    for fsts in sts_files:
        name = os.path.splitext(os.path.basename(fsts))[0]
        content = name.split('_')[0]
        
        crf = name.split('_')[2]
        fps =  name.split('_')[3]
        freq = name.split('_')[-1]
        if(content==des_content and crf=='0' and freq=='3'):
            print(name)
            data = load(fsts)
            ggd_params = save_stats.estimateggdparam(data.flatten())
            print(ggd_params)
            color_index = fps_full.index(fps)
            color = color_list[color_index]
            sns.kdeplot(data=data.flatten(),color=color,label=fps)
            data_list.append(data.flatten())
            fps_list.append(fps)

    plt.legend()
    plt.savefig('./images/sts_MSCNthentemp_'+des_content+'.png')
    plt.close()
