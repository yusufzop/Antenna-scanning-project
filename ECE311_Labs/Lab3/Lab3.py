import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display # listening to audio
import os
from pz_plot import pz_plot # pole-zero plotting utility
from scipy import signal
from scipy.io import wavfile # audio i/o

b = [1, -0.5, 0] # numerator coefficients, make sure to align zeros where delay terms don't align
a = [1, -1, 1] # denominator coefficients
z, p, k = signal.tf2zpk(b, a) # k is system gain -> okay to ignore this

"""
pz_plot(zeros_list, poles_list, plot_title)
zeros_list = list or array of transfer function's zeros
poles_list = list or array of transfer function's poles
plot_title = string for title of your plot
"""

#pz_plot(z, p, 'Pole-Zero Plot Example')

"""

n,y = signal.dimpulse((b, a, 1), n=20) # the 1 is for the change in time per sample, n=20 means 20 points for output
h_n = y[0] # need to access output from returned object

plt.figure()
plt.stem(n, h_n) # plot first 20 samples of impulse response
plt.xlabel('n')
plt.ylabel('Output Signal')
plt.title('Impulse Response of Example System')
plt.show()
"""

c = [1,1,0,0]
d = [1,0,0,-1]
z,p,k = signal.tf2zpk(c,d)

"""
print('Poles:', p)
print('Zeros:', z)

pz_plot(z,p,"zort")
ax = plt.gca()
leg = ax.get_legend()
if leg is not None:
    leg.remove()
plt.show()

n,y = signal.dimpulse((c,d,1),n=20)
a = y[0]
plt.figure()
plt.stem(n, a) # plot first 20 samples of impulse response
plt.xlabel('n')
plt.ylabel('Output Signal')
plt.title('Impulse Response of Example System')
plt.show()

"""

h1_nom = [1/2,-1/2,1/2]
h2_nom = [1,0,0]
a = np.arange(0,11)

x1 = np.cos(np.pi / 2 * a)
y1 = signal.lfilter(h1_nom,h2_nom,x1)
#plt.plot(y1)
#plt.show()


"""
a = [ 1,0,0]
b = [1,-2 ** (1/2), 1]
z, p, k = signal.tf2zpk(a, b)
pz_plot(z,p,"a")
plt.show()

n = np.arange(0,200)

x1 = np.cos(np.pi / 2 * n)
y1 = signal.lfilter(a,b,x1)
plt.plot(y1)
plt.show()

"""

fs, clean = wavfile.read('ECE311_Labs/Lab3/clean-audio.wav') #reference for what the clean audio sounds like
fs, distorted = wavfile.read('ECE311_Labs/Lab3/distorted-audio.wav') #distorted audio signal
print(clean.shape) #shape of audio clip numpy arrays, only one channel for these clips
print(distorted.shape)

a = [2,0,-2]
b = [1,1,1/4]
y1 = signal.lfilter(b,a,distorted)
y1 = y1.astype(np.int16)

out_path = r"ECE311_Labs\Lab3\filtered_y1.wav"

wavfile.write(out_path, fs, y1)

os.startfile(out_path)