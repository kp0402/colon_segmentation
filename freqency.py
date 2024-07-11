#adapted from https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/

import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fftpack import fft
from scipy.signal import welch

def get_fft_values(y_values, T, N):
    if(N > 2047):
        f_values, Pxx_den = welch(y_values, 1.0/(T),nperseg=1024,detrend="constant")
        fft_values = Pxx_den
    else:
        f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
        fft_values_ = fft(y_values)
        fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
        fft_values = fft_values**2
    return f_values, fft_values

def plot_fft_plus_power(ax, time, signal, label, cutoff=0.6, linewidth=1):
    dt = time[1] - time[0]
    N = len(signal)
    f_values, fft_power = get_fft_values(signal, dt, N)
    
    # Determine the cutoff index based on the desired frequency cutoff
    cutoff_index = np.argmax(f_values > cutoff)
    f_values, fft_power = f_values[:cutoff_index], fft_power[:cutoff_index]
    
    # Normalize fft_power to have a maximum value of 1 (if needed)
    fft_power /= max(fft_power)

    # Set x-axis limits
    ax.set_xlim(0.02, 0.6)
    
    if not label:
        ax.plot(f_values, fft_power, linewidth=linewidth, label='FFT Power Spectrum')
    else:
        from US_colon_experiment.utilities import SnaptoCursor     
        cursor = SnaptoCursor(ax, f_values, fft_power)
        plt.connect('motion_notify_event', cursor.mouse_move)
        plt.connect('button_press_event', cursor.onclick)
        ax.plot(f_values, fft_power, linewidth=linewidth)
        
        plt.show()


# def plot_fft_plus_power(ax, time, signal, label, cutoff=0.6):
#     dt = time[1] - time[0]
#     N = len(signal)
#     f_values, fft_power = get_fft_values(signal, dt, N)
    
#     # Determine the cutoff index based on the desired frequency cutoff
#     cutoff_index = np.argmax(f_values > cutoff)
#     f_values, fft_power = f_values[:cutoff_index], fft_power[:cutoff_index]
    
#     # Normalize fft_power to have a maximum value of 1 (if needed)
#     fft_power /= max(fft_power)
    
#     if not label:
#         ax.plot(f_values, fft_power, linewidth=1, label='FFT Power Spectrum')
#     else:
#         from US_colon_experiment.utilities import SnaptoCursor     
#         cursor = SnaptoCursor(ax, f_values, fft_power)
#         plt.connect('motion_notify_event', cursor.mouse_move)
#         plt.connect('button_press_event', cursor.onclick)
#         ax.plot(f_values, fft_power)
        
#         plt.show()

#     ax.set_xlim(0.02, 0.6)  # Set the x-axis limit to 0.6

# def plot_fft_plus_power(time, signal, label=None, cutoff=0.6):
#     dt = time[1] - time[0]
#     N = len(signal)
#     f_values, fft_power = get_fft_values(signal, dt, N)
    
#     # Determine the cutoff index based on the desired frequency cutoff
#     cutoff_index = np.argmax(f_values > cutoff)
#     f_values, fft_power = f_values[:cutoff_index], fft_power[:cutoff_index]
    
#     # Normalize fft_power to have a maximum value of 1 (if needed)
#     fft_power /= max(fft_power)
    
#     # Create a new figure and axis if not provided
#     fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figsize to make it more compact
    
#     if not label:
#         ax.plot(f_values, fft_power, linewidth=1, label='FFT Power Spectrum')
#     else:
#         from US_colon_experiment.utilities import SnaptoCursor     
#         cursor = SnaptoCursor(ax, f_values, fft_power)
#         plt.connect('motion_notify_event', cursor.mouse_move)
#         plt.connect('button_press_event', cursor.onclick)
#         ax.plot(f_values, fft_power)
    
#     ax.set_xlim(0, 0.6)  # Set the x-axis limit to 0.6
#     ax.set_xlabel('Frequency (Hz)')
#     ax.set_ylabel('Power')
#     ax.legend()
#     ax.grid(True)
    
#     # Adjust layout to make it more compact
#     plt.tight_layout()
    
#     plt.show()

# def plot_fft_plus_power(ax, time, signal, label,cutoff=1):
#     dt = time[1] - time[0]
#     N = len(signal)
#     f_values, fft_power = get_fft_values(signal, dt, N)
#     cutoff_index = int(cutoff/f_values[1]-f_values[0])
#     f_values, fft_power = f_values[:cutoff_index],fft_power[:cutoff_index]
#     fft_power /= max(fft_power)

    
#     if not label:
#         ax.plot(f_values, fft_power, linewidth=1, label='FFT Power Spectrum')
#     else:
#         from US_colon_experiment.utilities import SnaptoCursor     
#         cursor = SnaptoCursor(ax, f_values, fft_power)
#         plt.connect('motion_notify_event', cursor.mouse_move)
#         plt.connect('button_press_event', cursor.onclick)
#         ax.plot(f_values, fft_power)
#         plt.show()


def get_ave_values(xvalues, yvalues, n = 5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return x_ave, y_ave

def plot_signal_plus_average(ax, time, signal, average_over = 5):
    time_ave, signal_ave = get_ave_values(time, signal, average_over)
    ax.plot(time, signal, label='signal')
    ax.plot(time_ave, signal_ave, label = 'time average (n={})'.format(5))
    ax.set_xlim([time[0], time[-1]])
    ax.set_ylabel('Amplitude', fontsize=16)
    ax.set_title('Signal + Time Average', fontsize=16)
    ax.legend(loc='upper right')

def plot_wavelet(time, signal, scales, 
                 waveletname = 'fbsp', 
                 cmap = plt.cm.seismic, 
                 title = 'Power Spectrum of signal', 
                 ylabel = 'Period (second)', 
                 xlabel = 'Time'):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4]
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()
    