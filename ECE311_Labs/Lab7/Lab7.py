#import libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from skimage.io import imread
from skimage.io import imsave

#Provided function with triangle magnitude spectrum
def toy_signal(w_c):
    N = 1000
    gain = 2*np.pi/w_c
    return gain * np.array([(np.sin(w_c/2*(n+1))/(np.pi*(n+1)))**2 if n+1 != 0 else (w_c/(2*np.pi))**2 for n in range(-N,N+1)])




# Fill in this function for part 1.a:
def upsample(U, x):
    upsampled_x = np.zeros((len(x) + (len(x)-1)*(U-1)))
    for i in range(len(x)):
        upsampled_x[i + i*(U-1)] = x[i]
    return upsampled_x

# Fill in this function for part 1.c:
def lowpass(C):
    
    return lpf

# Test code for part 1.a:
test_signal = np.array([1, 2, 3, 4, 5, 6])


# Code for part 1.b:


# Code for part 1.c:
upsampled_test = upsample(3,toy_signal(np.pi/2))
toy_signal_fft = np.fft.fft(toy_signal(np.pi/2))
toy_signal_fft = np.fft.fftshift(toy_signal_fft)

upsampled_test_fft = np.fft.fft(upsampled_test)
upsampled_test_fft = np.fft.fftshift(upsampled_test_fft)

w1 = np.linspace(-np.pi,np.pi,len(upsampled_test_fft))
w2 = np.linspace(-np.pi,np.pi,len(toy_signal_fft))

"""
plt.figure(figsize=(15,6))
plt.subplot(121)
plt.plot(w2,np.abs(toy_signal_fft))
plt.subplot(122)
plt.plot(w1,np.abs(upsampled_test_fft))
"""

a = [0,1/3,0.4333,1]
b = [1,0]
lpf = signal.remez(50,a,b,fs = 2)
lpf_fft = np.fft.fft(lpf)
lpf_fft = np.fft.fftshift(lpf_fft)
filtered_test = signal.lfilter(lpf,[1,0],upsampled_test)
filtered_test_fft = np.fft.fft(filtered_test)
filtered_test_fft = np.fft.fftshift(filtered_test_fft)
w3 = np.linspace(-np.pi,np.pi,len(filtered_test_fft))

"""
plt.figure(figsize=(15,6))
plt.plot(w3,np.abs(filtered_test_fft))
"""


# Fill in this function for part 2.a:
def downsample(D, x):
    downsampled = []
    for i in range(len(x)):
        if i+(D-1)*i < len(x):
            downsampled.append(x[i+(D-1)*i])
        else:
            break

    return downsampled

# Test code for part 2.a:
test_signal = np.array([1, 2, 3, 4, 5, 6])

# Code for part 2.b:

downsampled_test = downsample(3,toy_signal(np.pi/2))
toy_signal_fft = np.fft.fft(toy_signal(np.pi/2))
toy_signal_fft = np.fft.fftshift(toy_signal_fft)

downsampled_test_fft = np.fft.fft(downsampled_test)
downsampled_test_fft = np.fft.fftshift(downsampled_test_fft)

w1 = np.linspace(-np.pi,np.pi,len(downsampled_test_fft))
w2 = np.linspace(-np.pi,np.pi,len(toy_signal_fft))

"""
plt.figure(figsize=(15,6))
plt.subplot(121)
plt.plot(w2,np.abs(toy_signal_fft))
plt.ylim(0,1.1)
plt.subplot(122)
plt.plot(w1,np.abs(downsampled_test_fft))
plt.ylim(0,1.1)
plt.show()
"""

# Code for part 2.d:

up3 = upsample(3,toy_signal(np.pi/2))
a = [0,1/3,0.4333,1]
b = [1,0]
lpf = signal.remez(50,a,b,fs = 2)
filtered_up3 = signal.lfilter(lpf,[1,0],up3)

down5 = downsample(5,filtered_up3)
down5_fft = np.fft.fft(down5)
down5_fft = np.fft.fftshift(down5_fft)
w_sampled = np.linspace(-np.pi,np.pi,len(down5_fft))

plt.figure(figsize=(15,6))
plt.subplot(121)
plt.plot(w2,np.abs(toy_signal_fft))
plt.ylim(0,1.1)
plt.subplot(122)
plt.plot(w_sampled,np.abs(down5_fft))
plt.ylim(0,1.1)
plt.show()
