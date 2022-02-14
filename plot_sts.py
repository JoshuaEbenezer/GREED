import matplotlib.pyplot as plt
import save_stats
from joblib import load
import glob
import os

sts_files = glob.glob(os.path.join('./sts_arr_temporalbp_tempthenMSCN_data/','*.z'))
unique_contents = list(set([os.path.splitext(os.path.basename(fsts))[0].split('_')[0] for fsts in sts_files]))
print(unique_contents)

for des_content in unique_contents:
    plt.figure()
    plt.clf()
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
            plt.hist(data.flatten(),log=True,bins='auto',histtype='step',density=True,label=fps)

    plt.legend()
    plt.savefig('./images/sts_tempthenMSCN_'+des_content+'.png')
    plt.close()
