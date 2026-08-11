# Import libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from skimage.io import imread
from numpy.random import randn
from IPython.display import Audio
from scipy.io import wavfile

# Utility function for dB scaling of magnitude spectra
def sig2db(mag_spec):
    return 20*np.log10(mag_spec)

# Fill in this function for part 1.a:
def windowed_lpf(omega_c, N, window=np.ones):
    # create ideal impulse response in time domain before windowing
    lpf = np.zeros(N)
    for i in range(N):
        if i == (N-1)/2:
            lpf[i] = omega_c/np.pi
        else:
            lpf[i] = np.sin(omega_c * (i - (N-1)/2)) / (np.pi * (i - (N-1)/2))
    # apply window function 
    lpf = lpf * window(N)
    return lpf

# Code for part 1.b:
lpf1 = windowed_lpf(np.pi/3, 25, window = np.hamming)
lpf2 = windowed_lpf(np.pi/3, 25, window = np.hanning)
lpf3 = windowed_lpf(np.pi/3, 25, window = np.blackman)
lpf4 = windowed_lpf(np.pi/3, 25, window = np.ones)

w1, h1_ft = signal.freqz(lpf1)
w2, h2_ft = signal.freqz(lpf2)
w3, h3_ft = signal.freqz(lpf3)
w4, h4_ft = signal.freqz(lpf4)
"""
plt.figure(figsize=(20, 6))
plt.subplot(141)
plt.plot(w1, sig2db(np.abs(h1_ft)))
plt.subplot(142)
plt.plot(w2, sig2db(np.abs(h2_ft)))
plt.subplot(143)
plt.plot(w3, sig2db(np.abs(h3_ft)))
plt.subplot(144)
plt.plot(w4, sig2db(np.abs(h4_ft)))
"""
# Code for part 1.c:


# Code for 1.d:
hpf = lpf4 * np.array([(-1)**n for n in range(len(lpf4))])
w_hpf, h_hpf = signal.freqz(hpf)

"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(w_hpf, sig2db(np.abs(h_hpf)))
plt.title('High Pass Filter Frequency Response')
plt.subplot(122)
plt.plot(w4, sig2db(np.abs(h4_ft)))
plt.title('Low Pass Filter Frequency Response')
plt.show()
"""


"""
signal.remez(numtaps, bands, desired, fs=2)
Inputs:
numtaps - Filter length.
bands - List of band edges in normalized frequency [0, 1]
        where 1 correponds to pi.
desired - List of desired gains for each frequency bands.
          Must be half the length of the bands list.
fs=2 - This argument guarantees the input frequencies are on the normalized
       [0, 1] frequency axis.
Output:
out - Array of computed FIR filter coefficients.
"""

N = 50
a = [1, 0]
# Low-pass Filter
# 0-0.5 is passband, 0.7-1 is the stopband, transition band is then 0.5-0.7
lpf_bands = [0, 0.5, 0.7, 1]
lpf_desired = [1, 0] # gain of one for passband, gain of zero for stopband
lpf = signal.remez(N, lpf_bands, lpf_desired, fs=2)
w, H_lpf = signal.freqz(lpf, a)
# Band-pass Filter
# 0-0.2 is first stopband, 0.3-0.6 is passband, 0.7-1 is second stopband
bpf_bands = [0, 0.2, 0.3, 0.6, 0.7, 1]
bpf_desired = [0, 1, 0]
bpf = signal.remez(N, bpf_bands, bpf_desired, fs=2)
w, H_bpf = signal.freqz(bpf, a)

# Compare magnitude responses
"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.title('Low-pass Filter')
plt.plot(w, sig2db(abs(H_lpf)))
plt.subplot(122)
plt.title('Band-pass Filter')
plt.plot(w, sig2db(abs(H_bpf)))
plt.show()
"""

hpf_band1 = [0, 0.5, 0.7, 1]
hpf_band2 = [0, 0.4, 0.7, 1]
hpf_band3 = [0, 0.65, 0.7, 1]
hpf_desired = [0, 1]
hpf1 = signal.remez(101, hpf_band1, hpf_desired, fs=2)
hpf2 = signal.remez(101, hpf_band2, hpf_desired, fs=2)
hpf3 = signal.remez(101, hpf_band3, hpf_desired, fs=2)
w, H_hpf1 = signal.freqz(hpf1, a)
w, H_hpf2 = signal.freqz(hpf2, a)
w, H_hpf3 = signal.freqz(hpf3, a)

"""
plt.figure(figsize=(10, 6))
plt.subplot(131)
plt.title('High-pass Filter High (N=101)')
plt.plot(w, sig2db(abs(H_hpf2)))
plt.subplot(132)
plt.title('High-pass Filter Medium (N=101)')
plt.plot(w, sig2db(abs(H_hpf1)))
plt.subplot(133)
plt.title('High-pass Filter Low (N=101)')
plt.plot(w, sig2db(abs(H_hpf3)))
plt.show()
"""

fs, audio = wavfile.read(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab6\sound1.wav")
fft_sound = np.fft.rfft(audio)
omega = np.linspace(0, np.pi, len(fft_sound))

"""
plt.figure(figsize=(15, 6))
plt.subplot(131)
plt.title('Magnitude Spectrum of Sound')
plt.plot(omega, sig2db(np.abs(fft_sound)))
"""

lpf_band_sound = [0, 0.03 , 0.06, 1.64/3.114, 1.74/3.14,1 ]
lpf_desired_sound = [1, 0, 1]
lpf_sound = signal.remez(101, lpf_band_sound, lpf_desired_sound, fs=2)
H_lpf_sound = np.fft.rfft(lpf_sound)
omega_filter = np.linspace(0, np.pi, len(H_lpf_sound))

"""
plt.subplot(132)
plt.title('Magnitude Spectrum of Low-pass Filter')
plt.plot(omega_filter, sig2db(abs(H_lpf_sound)))
"""

filtered_sound = signal.lfilter(lpf_sound, a, audio)
H_filtered_sound = np.fft.rfft(filtered_sound)
omega_filtered = np.linspace(0, np.pi, len(H_filtered_sound))

"""
plt.subplot(133)
plt.title('Magnitude Spectrum of Filtered Sound')
plt.plot(omega_filtered, sig2db(abs(H_filtered_sound)))
"""


noisy = imread(r'C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab6\noisy-photo.jpg')
"""
plt.figure(figsize=(10,6))
plt.imshow(noisy,'gray')
plt.title('Noisy Image')
"""
shape = noisy.shape
n_rows, n_cols = shape[0], shape[1]

iir_num, iir_den = signal.iirdesign(ws=[0.2, 0.5], wp=[0.15, 0.55], gpass=1, gstop=25)
w, H_iir = signal.freqz(iir_num, iir_den)
bandstop_photo = [0,0.1, 0.2, 0.5, 0.6, 1]
bandstop_desired = [1,0,1]
bandstop_filter = signal.remez(25, bandstop_photo, bandstop_desired, fs=2)
w_fir, H_fir = signal.freqz(bandstop_filter)

"""
plt.figure(figsize=(15,6))
plt.subplot(121)
plt.plot(w,sig2db(np.absolute(H_iir)))
plt.subplot(122)
plt.plot(w_fir, sig2db(np.abs(H_fir)))

"""
"""
plt.figure(figsize=(15,6))
plt.subplot(121)
plt.plot(w,np.angle(H_iir))
plt.subplot(122)
plt.plot(w_fir, np.angle(H_fir))
"""
new_photo1 = np.zeros((n_rows,n_cols))
new_photo2 = np.zeros((n_rows,n_cols))

for i in range(n_rows):
    new_photo1[i,:] = signal.convolve(noisy[i,:],bandstop_filter,mode = 'same')

for i in range(n_cols):
    new_photo1[:,i] = signal.convolve(new_photo1[:,i],bandstop_filter,mode = 'same')



for i in range(n_rows):
    new_photo2[i,:] = signal.lfilter(iir_num,iir_den,noisy[i,:])

for i in range(n_cols):
    new_photo2[:,i] = signal.lfilter(iir_num,iir_den,new_photo2[:,i])

"""
plt.figure(figsize=(15,6))
plt.subplot(121)
plt.imshow(new_photo1,'gray')
plt.subplot(122)
plt.imshow(new_photo2,'gray')
plt.show()
"""




# Provided functions
def generate_xy(h, N):
    x = np.random.randint(low=-25, high=26, size=N)
    y = signal.convolve(h, x)
    return x, y

def generate_xy_noisy(h, N):
    x = np.random.randint(low=-25, high=26, size=N)
    y = signal.convolve(h, x).astype(np.float64)
    y += randn(len(y))*3 # Gaussian noise with standard deviation 3
    return x, y

# Fill in missing line for Part 5.a:
"""
lms(x, y, h, alpha, w_init):
Inputs:
x - input signal
y - output signal
h - your friend's filter (to check how well we learn it)
alpha - learning rate
w_init - initial filter coefficients

Outputs:
w_err - L2 errors (Euclidean distance) of LMS filter against your friend's filter
w - final estimate of your friend's filter
"""
def lms(x, y, h, alpha, w_init):
    N, L = len(x), len(h)
    w = w_init # adaptive filter
    w_err = np.zeros(N) # array to track filter coefficient errors
    X = np.zeros(L) # memory signal
    for n in range(N):
        X[0] = x[n] # update most recent (current) value
        d_n = y[n] # step to predict
        y_predict = np.dot(w, X) # predict y by w^Tx
        w_err[n] = np.linalg.norm(h - w) # error in filter coefficients
        e_n = d_n - y_predict # prediction error
        w = w + alpha * e_n * X[0:L] # filter update by gradient descent
        X = np.roll(X, 1) # circularly shift right by one
        
    return w_err,w

h = np.array([1,2,3,4,5])
N = 500
x, y = generate_xy_noisy(h, N)

w_err,w = lms(x,y,h, 10**(-4),[5,7,34,1,6])
print(w)
plt.figure(figsize=(15,6))
plt.plot(w_err)
plt.show()
