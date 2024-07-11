import numpy as np
from freqency import plot_fft_plus_power
from scipy.signal import butter,lfilter,detrend
from utilities import *
from skimage import exposure
from skimage.transform import resize


unit_time = 1/8.333
Fs = 1/unit_time
SIZE_Y=224
pixel_scale_y = pixel_scale_x = 0.01
TYPE = "new"
MAX_MINUTE = 16

def remap(transistion_point,array):
    k1 = transistion_point[1]/transistion_point[0]
    k2 = (1-transistion_point[1])/(1-transistion_point[0])
    cutoff = transistion_point[0]
    return np.where(array<cutoff, (k1*array),(k2*array-k2+1))

def STmap(data,y_ticks):
    #data = resample(data,SIZE_Y,axis=1)
    b, a = butter(8, 0.2, fs=Fs, btype='low')
    STmap = lfilter(b, a, data, axis=0)
    min_img = np.min(STmap)
    max_img = np.max(STmap)
    normalized = 1-(STmap-min_img)/(max_img-min_img)
    p2, p98 = np.percentile(normalized, (2, 98))
    normalized = exposure.rescale_intensity(normalized, in_range=(p2, p98))
    normalized = remap((0.6,0.3),normalized)
    ST_MAP = Interactive_frequencyplot(unit_time)
    ST_MAP.plot_stmap(normalized,y_ticks)

def freq_map(data):
    time_scale = np.arange(0,data.shape[0])*unit_time
    fig1,(ax1,ax2) = plt.subplots(2,1,figsize=(16,9))
    ax1.set_xlabel("time(s)")
    ax1.set_ylabel("diameter(cm)")
    ax1.invert_yaxis()
    ax2.set_xlabel("Frequency(Hz)")
    ax2.set_ylabel("Normalized PSD")
    ax1.plot(time_scale,data)
    plot_fft_plus_power(ax2,time_scale, detrend(data),True)

def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))

    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out

MAX_FRAME = int(MAX_MINUTE*60*Fs)

data =np.load(r"E:\Haustral_motility_figures\Code\main\data\All_Figures\Maxwell_E15\C8\temp.npz")
if TYPE == "new":
    max_index = min((data['mean_distance']).shape[0],MAX_FRAME)
    diameter_data = (data['STmap'])[:max_index,:]
    mean_distance = (data['mean_distance'])[:max_index]
    breathing = (data['breathing'])[:max_index]
    pixel_scale_x , pixel_scale_y = data['scales']
    #Fs = data['sample_rate']
else:
    diameter_data = (data['arr_0'])
    mean_distance = (data['arr_1'])
    pixel_scale_x , pixel_scale_y = data['arr_2']

diameter_data_detrend = detrend(diameter_data,axis=0)
#diameter_data_detrend = np.zeros_like(diameter_data)
#for i in range(diameter_data.shape[1]):
#    diameter_data_detrend[:,i] =diameter_data[:,i] - Utilities.moving_average(diameter_data[:,i],200)

yticks = ["{:.1f}".format(i) for i in pixel_scale_y * np.linspace(0,SIZE_Y,5)]
#freq_map(breathing)
#ST map
STmap(diameter_data_detrend,yticks)
#FFT
freq_map(mean_distance)
#Wavelet
wavelet = Interactive_frequencyplot(unit_time)
wavelet.plot_wavelet(diameter_data_detrend,mean_distance,np.linspace(2,256,200),yticks)



