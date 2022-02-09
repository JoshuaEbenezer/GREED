import numpy as np
from joblib import dump
from numba import jit,prange,njit
import matplotlib.pyplot as plt
from entropy.yuvRead import yuvRead_frame
import os
from entropy.save_stats import brisque
from entropy.entropy_params import est_params_ggd_temporal
import scipy.signal
from skvideo.utils.mscn import gen_gauss_window
import scipy.ndimage


def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    if avg_window is None:
      avg_window = gen_gauss_window(3, 7.0/6.0)
    assert len(np.shape(image)) == 2
    h, w = np.shape(image)
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = np.array(image).astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, 1, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image**2, avg_window, 0, var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, 1, var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image**2))
    return (image - mu_image)/(var_image + C), var_image, mu_image


def compute_MS_transform(image, window, extend_mode='reflect'):
    h,w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    scipy.ndimage.correlate1d(image, window, 0, mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, window, 1, mu_image, mode=extend_mode)
    return image - mu_image


@jit(nopython=True)
def find_sts_locs(sts_slope,cy,cx,st_time_length,h,w):
    if(np.abs(sts_slope)<1):
        x_sts = np.arange(cx-int((st_time_length-1)/2),cx+int((st_time_length-1)/2)+1)
        y = (cy-(x_sts-cx)*sts_slope).astype(np.int64)
        y_sts = np.asarray([y[j] if y[j]<h else h-1 for j in range(st_time_length)])
    else:
        #        print(np.abs(sts_slope))
        y_sts = np.arange(cy-int((st_time_length-1)/2),cy+int((st_time_length-1)/2)+1)
        x= ((-y_sts+cy)/sts_slope+cx).astype(np.int64)
        x_sts = np.asarray([x[j] if x[j]<w else w-1 for j in range(st_time_length)]) 
    return x_sts,y_sts


@jit(nopython=True)
def find_kurtosis_slice(Y3d_mscn,cy,cx,rst,rct,theta,h,st_time_length):
    st_kurtosis = np.zeros((len(theta),))
    data = np.zeros((len(theta),st_time_length**2))
    for index,t in enumerate(theta):
        rsin_theta = rst[:,index]
        rcos_theta  =rct[:,index]
        x_sts,y_sts = cx+rcos_theta,cy+rsin_theta
        
        data[index,:] =Y3d_mscn[:,y_sts*h+x_sts].flatten() 
        data_mu4 = np.mean((data[index,:]-np.mean(data[index,:]))**4)
        data_var = np.var(data[index,:])
        st_kurtosis[index] = data_mu4/(data_var**2+1e-4)
    idx = (np.abs(st_kurtosis - 3)).argmin()
    
    data_slice = data[idx,:]
    return data_slice,st_kurtosis[idx]-3


def find_kurtosis_sts(img_buffer,st_time_length,cy,cx,rst,rct,theta):

    h, w = img_buffer[st_time_length-1].shape[:2]
    Y3d_mscn = np.reshape(img_buffer.copy(),(st_time_length,-1))
    sts= [find_kurtosis_slice(Y3d_mscn,cy[i],cx[i],rst,rct,theta,h,st_time_length) for i in range(len(cy))]

    st_data = [sts[i][0] for i in range(len(sts))]
    return st_data

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))
def video_process(vid_path, width, height, bit_depth, gray, T, filt, num_levels, scales):
    
    base = os.path.splitext(os.path.basename(vid_path))[0]


    st_time_length = 5

    # LUT for coordinate search
    theta = np.arange(0,np.pi,np.pi/6)
    ct = np.cos(theta)
    st = np.sin(theta)
    lower_r = int((st_time_length+1)/2)-1
    higher_r = int((st_time_length+1)/2)
    r = np.arange(-lower_r,higher_r)
    rct = np.round(np.outer(r,ct))
    rst = np.round(np.outer(r,st))
    rct = rct.astype(np.int32)
    rst = rst.astype(np.int32)

    #Load WPT filters
    
    filt_path = 'WPT_Filters/' + filt + '_wpt_' + str(num_levels) + '.mat'
    wfun = scipy.io.loadmat(filt_path)
    wfun = wfun['wfun']
        
    blk = 5
    sigma_nsq = 0.1
    win_len = 7
    
    mult_scale_temporal_brisque=[]
    mult_scale_spatialMS_brisque = []
    vid_stream = open(vid_path,'r')
    
    for scale_factor in scales:
        sz = 2**(-scale_factor)
        frame_data = np.zeros((int(height*sz),int(width*sz),T))
        h = height*sz
        w = width*sz
        cy, cx = np.mgrid[st_time_length:h-st_time_length*4:st_time_length*4, st_time_length:w-st_time_length*4:st_time_length*4].reshape(2,-1).astype(int) # these will be the centers of each block
        r1 = len(np.arange(st_time_length,h-st_time_length*4,st_time_length*4)) 
        r2 = len(np.arange(st_time_length,w-st_time_length*4,st_time_length*4)) 
        
        spatial_MS_brisque_list = []
        for frame_ind in range(0, T):
            frame,_,_ = \
            yuvRead_frame(vid_stream, width, height, \
                                  frame_ind, bit_depth, gray, sz)
            
            window = gen_gauss_window((win_len-1)/2,win_len/6)
            MS_frame = compute_MS_transform(frame, window)
#            MSCN_frame = compute_image_mscn_transform(frame, avg_window=window)

            frame_data[:,:,frame_ind] = MS_frame

            
        
#            spatial_MS_brisque = brisque(MS_frame)
#            spatial_MS_brisque_list.append(spatial_MS_brisque)
        
        #Wavelet Packet Filtering
        #valid indices for start and end points
        valid_lim = frame_data.shape[2] - wfun.shape[1] + 1
        start_ind = wfun.shape[1]//2 - 1
        dpt_filt = np.zeros((frame_data.shape[0], frame_data.shape[1],\
                             2**num_levels - 1, valid_lim))
#        temporal_brisque_feats = np.zeros((2**num_levels - 1, valid_lim,18))
        
        for freq in range(wfun.shape[0]):
            dpt_filt[:,:,freq,:] = scipy.ndimage.filters.convolve1d(frame_data,\
                    wfun[freq,:],axis=2,mode='constant')[:,:,start_ind:start_ind + valid_lim]



            index = 7
            Y_block = dpt_filt[:,:,freq,index*5:(index+1)*5]
            sts = find_kurtosis_sts(Y_block,st_time_length,cy,cx,rst,rct,theta)
            sts_arr = unblockshaped(np.reshape(sts,(-1,st_time_length,st_time_length)),r1*st_time_length,r2*st_time_length)

#            feats =  ChipQA.save_stats.brisque(sts_arr)

            dump(sts_arr,'./sts_arr_temporalbp_MSthentemp_data/'+base+'_'+str(index*5)+'_'+str(freq)+'.z')

            plt.figure()
            plt.clf()
            plt.hist(sts_arr.flatten(),histtype='step',bins='auto',density=True)
            plt.savefig('./images/sts_arr_MS_temporalbp_'+base+'_'+str(index*5)+'_'+str(freq)+'.png')
            plt.clear()
            
#            for frame_ind in range(valid_lim):
#                temporal_brisque_feats[freq,frame_ind,:] = brisque(dpt_filt[:,:,freq,frame_ind])
        return 0,0
#        mult_scale_temporal_brisque.append(temporal_brisque_feats)
#        mult_scale_spatialMS_brisque.append(np.asarray(spatial_MS_brisque_list))
#    mult_scale_temporal_brisque = np.asarray(mult_scale_temporal_brisque)
#    mult_scale_spatialMS_brisque=np.asarray(mult_scale_spatialMS_brisque)
#    print(mult_scale_temporal_brisque.shape)
#    print(mult_scale_spatialMS_brisque.shape)
#    return mult_scale_spatialMS_brisque,mult_scale_temporal_brisque
