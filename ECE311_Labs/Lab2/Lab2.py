import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from skimage.io import imread

n1 = np.arange(0,20)
x1 = np.where(n1<10,1,0)

n2 = np.arange(0,40)
x2 = np.sin(np.pi * n2 / 20)

h1 = np.array([1,-1])
h2 = np.array([1/3,1/3,1/3])

y1 = signal.convolve(x1,h1)
y2 = signal.convolve(x1,h2)
y3 = signal.convolve(x2,h1)
y4 = signal.convolve(x2,h2)

#plt.stem(y1)
#plt.show()


bitcoin_data = np.load('bitcoin-christmas.npy', allow_pickle=True)
"""
n_points = len(bitcoin_data)
plt.figure(figsize = (8,6))
plt.plot(range(n_points), bitcoin_data)
plt.title('Bitcoin Prices Every Minute 12/24/2017')
plt.xlabel('Minute from Midnight')
plt.ylabel('Price (USD)')
"""



# Provided code
L = 51
half_L = 25
start = 50
end = len(bitcoin_data)-half_L
plotting_data = bitcoin_data[start:end] # plot against this data in parts a/b, pay attention to how we slice
result_length = len(plotting_data)

# Code for 2.a here, don't forget to plot original and filtered signals on same plot!

h1 = np.ones(51) / 51
y1 = signal.convolve(bitcoin_data,h1)
plotting_data2 = y1[start:end]

h2 = np.ones(51) / 51
y2 = signal.convolve(bitcoin_data,h1,'same')
plotting_data3 = y2[start:end]

"""
# Code for 2.b here
plt.plot(range(len(plotting_data)),plotting_data)
plt.plot(range(len(plotting_data)),plotting_data2)
plt.show()

plt.plot(range(len(plotting_data)),plotting_data)
plt.plot(range(len(plotting_data)),plotting_data3)
plt.show()

"""

# make filter and load image
image = imread('test-image.jpg')
#plt.figure(figsize=(10,6))
#plt.imshow(image, 'gray')
#plt.title('Original Image')
#plt.show()
L = 11
h = np.ones(L) * 1/11 # impulse response of length-11 moving average filter
n_rows, n_cols = image.shape
# Code for 3.a along rows (apply filter to each row independently)
image_row = np.zeros(image.shape)
# Hint: image_row[i, :] = signal.convolve(image[i, :], h, 'same') for each row

for i in range(n_rows):
    image_row[i,:] = signal.convolve(image[i,:],h,'same')

#plt.imshow(image_row, 'gray')
#plt.show()
# along the columns (3.b)
image_columns = np.zeros(image.shape)

for i in range(n_cols):
    image_columns[:,i] = signal.convolve(image[:,i],h,'same')

#plt.imshow(image_columns, 'gray')
#plt.show()
# columns then rows (3.c)

image3 = np.zeros(image.shape)

for i in range(n_cols):
    image3[:,i] = signal.convolve(image[:,i],h,'same')

for i in range(n_rows):
    image3[i,:] = signal.convolve(image[i,:],h,'same')

#plt.imshow(image3, 'gray')
#plt.show()


# load test-image.jpg
test = imread('test-image.jpg')
n_rows, n_cols = test.shape
h = np.array([1,-1])




# Code for 4.c here:
for i in range(n_rows):
    image_row[i,:] = signal.convolve(test[i,:],h,'same')

#plt.imshow(image_row, 'gray')
#plt.show()


for i in range(n_cols):
    image_columns[:,i] = signal.convolve(test[:,i],h,'same')

#plt.imshow(image_columns, 'gray')
#plt.show()

# Code for 4.d here:
final_image = np.zeros(image.shape)
final_image = (image_row ** 2 + image_columns ** 2) ** (1/2)
#plt.imshow(final_image, 'gray')
#plt.show()

# Code for 4.e here:







def convolve_2d(image, kernel):
    result = signal.convolve2d(image, kernel, 'same')
    result[result < 0] = 0 # Keep values non-negative
    return result

# identify horizontal lines
horiz_kernel = np.array([[-2,-2,-2,-2,-2],
                         [1,1,1,1,1],
                         [1,1,1,1,1],
                         [1,1,1,1,1],
                         [-2,-2,-2,-2,-2]])

# identify vertical lines
vert_kernel = np.array([[-2,1,1,1,-2],
                        [-2,1,1,1,-2],
                        [-2,1,1,1,-2],
                        [-2,1,1,1,-2],
                        [-2,1,1,1,-2]])


"""

one = imread('eight.jpg')

plt.figure(figsize=(16,10))
plt.subplot(131)
plt.title('Original')
plt.imshow(one, 'gray')

one_horiz = convolve_2d(one, horiz_kernel)
plt.subplot(132)
plt.title('Horizontal Features')
plt.imshow(one_horiz, 'hot')
plt.colorbar(fraction=0.05)

one_vert = convolve_2d(one, vert_kernel)
plt.subplot(133)
plt.title('Vertical Features')
plt.imshow(one_vert, 'hot')
plt.colorbar(fraction=0.05)
plt.show()

"""




clean_image = imread('clean-image.jpg')
noisy_image = imread('noisy-image.jpg')
plt.figure(figsize=(15,15))
plt.subplot(131)
plt.imshow(noisy_image, 'gray')
plt.title('Noisy Image')
plt.subplot(132)



mean_kernel = np.array([[1/25,1/25,1/25,1/25,1/25],
                        [1/25,1/25,1/25,1/25,1/25]
                        ,[1/25,1/25,1/25,1/25,1/25],
                        [1/25,1/25,1/25,1/25,1/25],
                        [1/25,1/25,1/25,1/25,1/25]])

mean_image = convolve_2d(clean_image,mean_kernel)
plt.subplot(132)
plt.imshow(mean_image, 'gray')
plt.title('Denoised with Mean Filter')

median_image = signal.medfilt2d(noisy_image,kernel_size=5)
plt.subplot(133)
plt.imshow(median_image, 'gray')
plt.title('Denoised with Median Filter')
plt.show()