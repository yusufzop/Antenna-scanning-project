import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Audio
from scipy import signal
from scipy.io import wavfile
from skimage.io import imread
from sklearn.cluster import KMeans


x = [0, 1, 0, 2, 0, 2, 0, 1, 0] # test signal

full_fft = np.fft.fft(x)
real_fft = np.fft.rfft(x)

omega_full = np.linspace(0, 2*np.pi, len(full_fft)) # left limit, right limit, # pts
omega_real = np.linspace(0, np.pi, len(real_fft))

"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.title('Full FFT')
plt.xlabel('$\omega$')
plt.ylabel('Magnitude Response')
plt.plot(omega_full, np.absolute(full_fft))

plt.subplot(122)
plt.title('Real FFT')
plt.xlabel('$\omega$')
plt.ylabel('Magnitude Response')
plt.plot(omega_real,np.absolute(real_fft))

"""

x = [0, 1, 0, 2, 0, 2, 0, 1, 0]

full_fft = np.fft.fft(x, 512)
centered_fft = np.fft.fftshift(full_fft) # shifts central frequency to middle of array
real_fft = np.fft.rfft(x, 512)

omega_full = np.linspace(-np.pi, np.pi, len(centered_fft)) # new frequency axis
omega_real = np.linspace(0, np.pi, len(real_fft))
"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.title('Full FFT')
plt.xlabel('$\omega$')
plt.ylabel('Real Part')
plt.plot(omega_full, np.absolute(centered_fft))

plt.subplot(122)
plt.title('Real FFT')
plt.xlabel('$\omega$')
plt.ylabel('Real Part')
plt.plot(omega_real, np.absolute(real_fft))
"""


b = [np.sin((np.pi/2)*n)/(0.5*np.pi*n) if n != 0 else 1 for n in range(-100, 101)] # numerator coefficients
a = [1,0] #denominator coefficients
w, h = signal.freqz(b, a) # w = omega/digital frequencies, h = frequency response
"""
plt.figure(figsize=(10, 6))
plt.title('Toy Frequency Response')
plt.plot(w, 20*np.log10(np.absolute(h))) # plot magnitude of frequency response with db-scaling on y-axis
plt.xlabel('$\omega$')
plt.ylabel('Magnitude Response (dB)')

plt.show()
"""

# Function to implement for part 1.a:
"""
Inputs:
x - input signal (list or np.array)
w - frequencies we want to compute the DTFT for (list or np.array)
Output:
dtft - value of the DTFT for signal x at each frequency specified in w (list or np.array)
"""
def myDTFT(x, w):
    dtft = [] # create empty list to append resulting computation
    # Iterate over each frequency in w
    for freq in w:
        dtft_val = 0
        for n in range(len(x)):
            dtft_val = dtft_val + x[n]*np.exp(-1j*freq*n)
        dtft.append(dtft_val)
    # Compute summation for current frequency according to our DTFT definition, "1j" gives you the imaginary number
    
    # Append result for current frequency
    
    return dtft

# Code for part 1.b:
x = np.array([np.cos(0.5*np.pi*n) for n in range(50)])
# endpoint argument makes sure our definition aligns with the np.fft.fft result
w = np.linspace(-np.pi, np.pi, 50, endpoint=False)

dtft_result = myDTFT(x,w)
real_dtft = np.fft.fft(x,50)
real_dtft = np.fft.fftshift(real_dtft)

"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(w, np.absolute(dtft_result))
plt.subplot(122)
plt.plot(w, np.absolute(real_dtft))
plt.show()
"""

x1 = np.array([1/4,1/2,1,1/2,1/4])
dtft_x1 = np.fft.fft(x1,512)
dtft_x1 = np.fft.fftshift(dtft_x1)
omega_x1 = np.linspace(-np.pi, np.pi, len(dtft_x1))
"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(omega_x1, np.absolute(dtft_x1))
plt.subplot(122)
plt.plot(omega_x1, np.angle(dtft_x1))
plt.show()
"""
a = [1,-2,1]
b = [1,-1/2,1/4]
w,h = signal.freqz(a,b,512)

"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(w, np.absolute(h))
plt.subplot(122)
plt.plot(w, np.angle(h))
plt.show()
"""

h = np.array([1/2,0,1,0,1/2])
dtft_h = np.fft.fft(h,512)
dtft_h = np.fft.fftshift(dtft_h)
omega_h = np.linspace(-np.pi, np.pi, len(dtft_h),endpoint=False)
"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(omega_h, np.absolute(dtft_h))
plt.subplot(122)
plt.plot(omega_h, np.angle(dtft_h))
"""

x1 = 2 + 10*np.array([np.sin((np.pi/2)*n) for n in range(100)]) + 4*np.array([np.cos((np.pi)*n) for n in range(100)])
y = signal.lfilter(h,[1],x1)
dtft_y = np.fft.fft(y,512)
dtft_y = np.fft.fftshift(dtft_y)

dtft_x1 = np.fft.fft(x1,512)
dtft_x1 = np.fft.fftshift(dtft_x1)
"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(omega_h, np.absolute(dtft_x1))
plt.subplot(122)
plt.plot(omega_h, np.angle(dtft_x1))

plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(omega_h, np.absolute(dtft_y))
plt.subplot(122)
plt.plot(omega_h, np.angle(dtft_y))
plt.show()
"""

fs, audio = wavfile.read(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab4\audiofile.wav")
print(audio.shape)

filter1 = np.load(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab4\filter-one.npy")
filter2 = np.load(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab4\filter-two.npy")
w1,h1 = signal.freqz(filter1,[1],512)
w2,h2 = signal.freqz(filter2,[1],512)
fft_audio = np.fft.fft(audio,512)
fft_audio = np.fft.fftshift(fft_audio)
"""
plt.figure(figsize=(15, 15))
plt.subplot(221)
plt.plot(w1, np.absolute(h1))
plt.title('Filter 1 Frequency Response')
plt.subplot(222)
plt.plot(w2, np.absolute(h2))
plt.title('Filter 2 Frequency Response')
plt.subplot(223)
omega_audio = np.linspace(-np.pi, np.pi, len(fft_audio),endpoint=False)
plt.plot(omega_audio, np.absolute(fft_audio))
plt.title('Audio Signal Frequency Response')
"""

audio_filtered1 = signal.lfilter(filter1,[1],audio)
audio_filtered2 = signal.lfilter(filter2,[1],audio)
audio_filtered1_fft = np.fft.fft(audio_filtered1,512)
audio_filtered1_fft = np.fft.fftshift(audio_filtered1_fft)
audio_filtered2_fft = np.fft.fft(audio_filtered2,512)
audio_filtered2_fft = np.fft.fftshift(audio_filtered2_fft)
"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(omega_audio, np.absolute(audio_filtered1_fft))
plt.title('Filtered Audio Signal 1 Frequency Response')
plt.subplot(122)
plt.plot(omega_audio, np.absolute(audio_filtered2_fft))
plt.title('Filtered Audio Signal 2 Frequency Response')
plt.show()

def save_audio(filename, audio_data, fs):
    audio_data = audio_data.astype(float)

    max_val = np.max(np.abs(audio_data))
    if max_val != 0:
        audio_data = audio_data / max_val

    audio_int16 = np.int16(audio_data * 32767)
    wavfile.write(filename, fs, audio_int16)



)

print("filtered_audio_1.wav kaydedildi")
print("filtered_audio_2.wav kaydedildi")

"""

Fs = 44100 # sampling rate for audio clip in Hz
t1 = 5 # make clips 5 seconds
t = np.linspace(0, t1, t1*Fs) # make sure to specify the number of points to match desired sampling frequency!!!
f0 = 0 # start frequency (Hz)
f1 = 22050 # end frequency (Hz)
Fs2 = 29400
t2 = np.linspace(0, t1, t1*Fs2) # make sure to specify the number of points to match desired sampling frequency!!!
"""
instantanous frequency, f(t) = f0 + (f1-f0)*(t/t1)
"""

chirp_original = signal.chirp(t, f0=f0, t1=t1, f1=f1)
chirp2 = signal.chirp(t2, f0=f0, t1=t1, f1=f1)
chirp_int16 = np.int16(chirp2 * 32767)

wavfile.write(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab4\chirp2.wav",
    
    Fs2,
    chirp_int16
)

print("chirp2.wav kaydedildi")

"""
# wav olarak kaydet
chirp_int16 = np.int16(chirp_original * 32767)

wavfile.write(
    
    Fs,
    chirp_int16
)

print("chirp_original.wav kaydedildi")
"""
fft_audio = np.fft.fft(chirp_original, 512)
fft_audio = np.fft.fftshift(fft_audio)
fft_audio2 = np.fft.fft(chirp2, 512)
fft_audio2 = np.fft.fftshift(fft_audio2)
omega_audio2 = np.linspace(-np.pi, np.pi, len(fft_audio2), endpoint=False)
omega_audio = np.linspace(-np.pi, np.pi, len(fft_audio), endpoint=False)
"""
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.plot(omega_audio, 20 * np.log10(np.absolute(fft_audio)))
plt.subplot(122)
plt.plot(omega_audio2, 20 * np.log10(np.absolute(fft_audio2)))
plt.show()
"""


# Part 6.a
# Fill in the below function!
"""
image - image we want to color quantize
k - number of quantization levels (# of crayons)
For example, k = 4 means we will have levels at 0, 85, 170, and 255.

returns: q_image = color quantized image
"""
def uniform_quantizer(image, k):
    # create quantization levels
    levels = np.linspace(0, 255, k) # k evenly spaced colors from 0 (black) to 255 (white)
    # create a new/blank version of the image and compute quantization level spacing
    q_image = np.zeros(image.shape)
    spacing = 255/(k-1)
    # go through each pixel in the original image, assign quantized value to new/blank image
    # remember we choose the quantization level closest to the original value
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            q_image[i,j] = round(image[i,j]/spacing)*spacing

    # return your quantized image
    return q_image

gray_image = imread(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab4\grayscale-test.jpg")
quantized_image = uniform_quantizer(gray_image, 8)
plt.figure(figsize=(15, 6))
plt.subplot(121)
plt.imshow(quantized_image, 'gray')


def lm_quantizer(image, k):
    im_shape = image.shape
    n_rows = im_shape[0]
    n_cols = im_shape[1]

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)

    if len(im_shape) == 2:
        # grayscale image: each pixel is one number
        pixel_vals = image.reshape(-1, 1)

        color_labels = kmeans.fit_predict(pixel_vals)

        # cluster centers are shape (k, 1), make them shape (k,)
        colors = np.round(kmeans.cluster_centers_).flatten().astype(np.uint8)

        q_image = np.zeros(im_shape).astype(np.uint8)

        for i, label in enumerate(color_labels):
            q_image[i // n_cols, i % n_cols] = colors[label]

    else:
        # RGB image: each pixel is [R, G, B]
        pixel_vals = image.reshape(-1, im_shape[2])

        color_labels = kmeans.fit_predict(pixel_vals)

        colors = np.round(kmeans.cluster_centers_).astype(np.uint8)

        q_image = np.zeros(im_shape).astype(np.uint8)

        for i, label in enumerate(color_labels):
            q_image[i // n_cols, i % n_cols] = colors[label]

    return q_image

quantized_image2 = lm_quantizer(gray_image, 8)
plt.subplot(122)
plt.imshow(quantized_image2, 'gray')

quantized_image3 = lm_quantizer(imread(r"C:\Users\Bogazici\Desktop\kodlar\ECE311_Labs\Lab4\color-test.jpg"), 16)
plt.figure(figsize=(15, 6))
plt.imshow(quantized_image3)
plt.show()