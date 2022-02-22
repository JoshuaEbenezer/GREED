import os
import glob
from joblib import Parallel,delayed,load,dump
import pandas as pd
from entropy.entropy_cal import video_process
from entropy.entropy_temporal_pool import entropy_temporal_pool
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Generate ChipQA features from a folder of videos and store them')
parser.add_argument('--input_folder',help='Folder containing input videos')
parser.add_argument('--results_folder',help='Folder where features are stored')

args = parser.parse_args()
def greed_feat(vid_path):
    
    filt = 'bior22'
    num_levels = 3

    height = 2160
    width =3840 
    gray = True
    fourkcontent = 
    bit_depth = 8
    
    if bit_depth == 8:
        multiplier = 1.5    #8 bit yuv420 video
    else:
        multiplier = 3      #10 bit yuv420 video
    
    if height < 1080:
        scales = [3,4]      #for lower than 1080p resolution
    elif height < 2160:
        scales = [4,5]      #1080p resolution
    else:
        scales = [5,6]      #for 4K resolution
    
    #calculate number of frames in reference and distorted    
    print(vid_path)
    vid_stream = open(vid_path,'r')
    vid_stream.seek(0, os.SEEK_END)
    vid_filesize = vid_stream.tell()
    vid_T = int(vid_filesize/(height*width*multiplier))
    
#     dist_stream = open(dist_path,'r')
#     dist_stream.seek(0, os.SEEK_END)
#     dist_filesize = dist_stream.tell()
#     dist_T = int(dist_filesize/(height*width*multiplier))
    
    
    #calculate spatial entropy
    mult_scale_chipqa_MSCNthentemp= video_process(vid_path, width, height, bit_depth, gray, \
                                   vid_T, filt, num_levels, scales)
    
    return mult_scale_chipqa_MSCNthentemp

import argparse
import os

def main(i,path_list,outfolder):
    vid_path = path_list[i]
    if(os.path.exists(vid_path)==False):
        print('input file does not exist')
        return

    base = os.path.splitext(os.path.basename(vid_path))[0]
    outname = os.path.join(outfolder,base+'.z')
    if(os.path.exists(outname)):
        print('output file exists')
        return
    
    
    bit_depth = 8
    if bit_depth == 8:
        pix_format = 'yuv420p'
    else:
        pix_format = 'yuv420p10le'
    
    GREED_feat = greed_feat(vid_path)
    print(GREED_feat)
    dump(GREED_feat,outname)
    return


if __name__ == '__main__':

    df = pd.read_csv('../LIVE-HFR/subjective_scores/subject_MOS.csv')

    files = df.columns[1:]
    path_list =glob.glob(os.path.join(args.input_folder,'*.yuv'))#  [os.path.join(args.input_folder,f.split('_')[0],f+'.yuv') for f in files]
    print(files)
    print(path_list)
    outfolder = args.results_folder
    os.makedirs(outfolder,exist_ok=True)
    Parallel(n_jobs=80)(delayed(main)\
            (i,path_list,outfolder)\
            for i in range(len(path_list)))
#    main(args.ref_path,outfolder)
