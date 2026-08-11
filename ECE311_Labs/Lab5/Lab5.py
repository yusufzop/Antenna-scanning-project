# import libraries
from timeit import timeit

import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Audio
import scipy
from skimage.io import imread
from scipy import signal
from scipy.io import wavfile
from linear_convolution import linear_convolution


# Function to complete for 1.a
def dfl_dft(x):
    N = len(x)
    dft = np.zeros(N, dtype=np.complex64)
    # for loop to iterate over 'k'
    for k in range(N):
        for n in range(N):
            dft[k] = dft[k] + x[n] * np.exp(-2j * np.pi * k * n / N)
    # for loop to iterate over 'n'

    return dft

# Functions to complete for 1.b
# Construct the matrix W
def dft_matrix(N):
    W = np.zeros((N, N), dtype=np.complex64)
    # fill in W however you see fit!
    for k in range(N):
        for n in range(N):
            W[k, n] = np.exp(-2j * np.pi * k * n / N)
    return W
# Take the DFT of signal x using the matrix W
def vectorized_dft(W, x):
    # this should only take one line...
    # refer to the above math if you are unsure!
    dft = W @ x
    return dft


# Provided code for 1.c
# Generate test signal
N = 500
x = np.random.uniform(size=N)

W = dft_matrix(N)

double_time = timeit(
    lambda: dfl_dft(x),
    number=1
)

vectorized_time = timeit(
    lambda: vectorized_dft(W, x),
    number=1
)

numpy_time = timeit(
    lambda: np.fft.fft(x),
    number=1
)

print("Double loop:", double_time)
print("Vectorized:", vectorized_time)
print("NumPy FFT:", numpy_time)

# Test if results are equivalent
print('Double for-loop and numpy fft are equivalent:', np.allclose(dfl_dft(x), np.fft.fft(x)))
print('Vectorized DFT and numpy fft are equivalent:', np.allclose(vectorized_dft(W, x), np.fft.fft(x)))

# Make longer signal that we will window in order to analyze a smaller portion
x = np.array([np.sin(np.pi/4*n) for n in range(200)])
# Case 1 
N1 = 80 # 10 full periods
x1 = x[:N1]
x1_fft = np.fft.rfft(x1)
omega_1 = np.linspace(0, np.pi, len(x1_fft))

# case 2
N2 = 78 # 7 full period and one fractional period
x2 = x[:N2]
x2_fft = np.fft.rfft(x2)
omega_2 = np.linspace(0, np.pi, len(x2_fft))
"""
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.stem(x1)
plt.title('Case 1 Signal')
plt.subplot(122)
plt.stem(x2)
plt.title('Case 2 Signal')

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.stem(omega_1, abs(x1_fft))
plt.title('FFT for Case 1')
plt.xlabel('$\omega$')
plt.subplot(122)
plt.stem(omega_2, abs(x2_fft))
plt.title('FFT for case 2')
plt.xlabel('$\omega$')
plt.show()
"""

N2 = 78 # number of points we use to create our window
x2 = x[:N2]
# quick function for converting a magnitude response to dB
def sig2db(mag_spec):
    return 20*np.log10(mag_spec)

hamming_window = scipy.signal.windows.hamming(N2)
kaiser_window = scipy.signal.windows.kaiser(N2, beta=3)

"""
plt.figure(figsize=(15,6))
plt.subplot(121)
plt.plot(hamming_window)
plt.title('Hamming Window')


plt.subplot(122)
plt.plot(kaiser_window)
plt.title('Kaiser Window')
"""


kaiser_window_fft = np.fft.rfft(kaiser_window, n=512)
hamming_window_fft = np.fft.rfft(hamming_window, n=512)
omega_axis = np.linspace(0, np.pi, len(kaiser_window_fft))
"""
plt.figure(figsize=(15,6))
plt.subplot(121)
plt.stem(omega_axis, sig2db(abs(kaiser_window_fft)))
plt.title('Kaiser Window FFT')
plt.subplot(122)
plt.stem(omega_axis, sig2db(abs(hamming_window_fft)))
plt.title('Hamming Window FFT')
"""

hamming_windowed_signal = x2 * hamming_window
kaiser_windowed_signal = x2 * kaiser_window
hamming_windowed_signal_fft = np.fft.rfft(hamming_windowed_signal, n=512)
kaiser_windowed_signal_fft = np.fft.rfft(kaiser_windowed_signal, n=512)

"""
plt.figure(figsize=(15,6))
plt.subplot(121)
plt.stem(omega_axis, sig2db(abs(hamming_windowed_signal_fft)))
plt.title('Hamming Windowed Signal FFT')
plt.subplot(122)
plt.stem(omega_axis, sig2db(abs(kaiser_windowed_signal_fft)))
plt.title('Kaiser Windowed Signal FFT')
"""

fs, music_stereo = wavfile.read(r'C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab5\Hallelujah_16k.wav') # Import the sound file
music_mono = music_stereo[:,0] # To obtain mono sound track

N = 256
music_mono_short = music_mono[:N] # if we are only given the first 256 samples
fft_short = np.fft.rfft(music_mono_short)
omega = np.linspace(0, np.pi, len(fft_short)) 
"""
plt.figure(figsize =(15, 6))
plt.subplot(121)
plt.plot(omega, abs(fft_short))
plt.title('DFT for length-256 audio signal')
plt.xlabel(r'$\omega$')
plt.ylabel('Magnitude Response')
"""

music_mono_padded = np.pad(music_mono_short, (0, 2048 - len(music_mono_short))) # zero pad the signal to length 512
fft_padded = np.fft.rfft(music_mono_padded)
omega_padded = np.linspace(0, np.pi, len(fft_padded))
"""
plt.subplot(122)
plt.plot(omega_padded, abs(fft_padded))
plt.title('DFT for length-2048 audio signal')
"""
N = 16 

music_mono_short = music_mono[:N] # if we are only given the first 256 samples
fft_short = np.fft.rfft(music_mono_short)
omega = np.linspace(0, np.pi, len(fft_short)) 
"""
plt.figure(figsize =(15, 6))
plt.subplot(121)
plt.plot(omega, abs(fft_short))
plt.title('DFT for length-256 audio signal')
plt.xlabel(r'$\omega$')
plt.ylabel('Magnitude Response')

"""
music_mono_padded = np.pad(music_mono_short, (0, 2048 - len(music_mono_short))) # zero pad the signal to length 512
fft_padded = np.fft.rfft(music_mono_padded)
omega_padded = np.linspace(0, np.pi, len(fft_padded))
"""
plt.subplot(122)
plt.plot(omega_padded, abs(fft_padded))
plt.title('DFT for length-2048 audio signal')
plt.show()
"""
fs_a, vowel_a = wavfile.read(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab5\a.wav") 
fs_r, cons_r = wavfile.read(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab5\r.wav") 
fs_b, cons_b = wavfile.read(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab5\b.wav") 
fs_robot, robot = wavfile.read(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab5\robot.wav")

nfft = 512
f_a, t_a, S_a = signal.spectrogram(vowel_a, fs_a, nperseg = nfft, noverlap = int(nfft/2), nfft = nfft)
f_r, t_r, S_r = signal.spectrogram(cons_r, fs_r, nperseg = nfft, noverlap = int(nfft/2), nfft = nfft)
f_b, t_b, S_b = signal.spectrogram(cons_b, fs_b, nperseg = nfft, noverlap = int(nfft/2), nfft = nfft)
f_robot, t_robot, S_robot = signal.spectrogram(robot, fs_robot, nperseg = nfft, noverlap = int(nfft/2), nfft = nfft)
"""
plt.figure(figsize=(20,5))
plt.subplot(231)
plt.pcolormesh(t_a, f_a, sig2db(S_a))
plt.title('Spectrogram for "a"')
plt.ylim([0, 1500])
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.subplot(232)
plt.pcolormesh(t_r, f_r, sig2db(S_r))
plt.title('Spectrogram for "r"')
plt.ylim([0, 1500])
plt.xlim([0, 1]) # limit the time axis to only see one iteration of the "rah" sound
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.subplot(233)
plt.pcolormesh(t_b, f_b, sig2db(S_b))
plt.title('Spectrogram for "b"')
plt.ylim([0, 1500])
plt.xlim([0, 1]) # limit the time axis to only see one iteration of the "bah" sound
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.subplot(234)
plt.pcolormesh(t_robot, f_robot, sig2db(S_robot))
plt.title('Spectrogram for "robot"')
plt.ylim([0, 1500])

plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()
"""


from scipy.linalg import toeplitz

def linear_convolution(x,h):
    col = np.append(h,np.zeros(len(x)-1))
    row = np.append(np.array([x[0]]),np.zeros(len(x)-1))
    toe = toeplitz(col,row)
    return np.dot(toe,x)


def fast_convolution(x, h):
    N = len(x) + len(h) - 1
    x_padded = np.pad(x, (0, N - len(x)))
    h_padded = np.pad(h, (0, N - len(h)))
    X = np.fft.fft(x_padded)
    H = np.fft.fft(h_padded)
    Y = X * H
    y = np.fft.ifft(Y)
    return np.real(y)

x1 = [1, 3, -2, -1]
x2 = [0, 2, 4, 1]


L = 2**12 # Change this for each part!
x = np.random.uniform(size=L)
h = np.random.uniform(size=L)


fast_time = timeit(
    lambda: fast_convolution(x, h),
    number=1
)

linear_time = timeit(
    lambda: linear_convolution(x, h),
    number=1
)




# Load and plot original image
img = imread(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab5\test-image.jpg")
n_rows = img.shape[0]
n_cols = img.shape[1]
plt.figure(figsize=(15, 6))
plt.subplot(131)
plt.imshow(img,'gray')
plt.title('Original Image')

# Compute 2D-DFT
fft2 = np.fft.fft2(img)

# Code for part 6.a:
part_a = fft2.copy() # create a hard copy to work with on part (a)
part_b = fft2.copy() # create a hard copy to work with on part (b)
for i in range(n_rows):
    part_a[i,:] = part_a[i,:] * np.exp(-2j * np.pi * 100  * i / n_rows)



part_a = np.fft.ifft2(part_a)

plt.subplot(132)
plt.imshow(np.real(part_a), 'gray')

# Code for part 6.b:
for j in range(n_cols):
    part_b[:,j] = part_b[:,j] * np.exp(-2j * np.pi * 100  * j / n_cols)
part_b = np.fft.ifft2(part_b)
plt.subplot(133)
plt.imshow(np.real(part_b), 'gray')
plt.show()

# Code for part 6.c:
