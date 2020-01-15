from pims import ND2_Reader
import os
import numpy as np
from matplotlib import pyplot as plt
from numpy import array
import cv2 as cv

f = r'/media/devici/srinath_dhm02/srinath_confocal/pillars1/00050cs0015mum_inverted_r1'
os.chdir(f)

frames = ND2_Reader('00050cs0015mum_inverted_r1.nd2')

matr = array(frames)

channels = matr.shape[0]
z_stacks = matr.shape[1]
x_px = matr.shape[2]
y_px = matr.shape[3]

b = np.zeros((z_stacks, x_px))
z_intensity = np.zeros((z_stacks,2))

for i in range(0,x_px):
    for j in range(0,z_stacks):
        b[j,i] = matr[0,j,i,486]

kk = b > np.power(2,12)/2

for j in range(0,z_stacks):
    z_intensity[j,0] = j
    z_intensity[j,1] = np.mean(matr[0,j,:,:])

ret,thresh1 = cv.threshold(b,np.power(2,12)/2,np.power(2,12),cv.THRESH_BINARY)

# plt.figure()
# plt.imshow(b,'gray')
#
# plt.figure()
# plt.imshow(thresh1,'gray')

plt.figure()
plt.imshow(kk,cmap = 'inferno')

plt.figure()
plt.scatter(z_intensity[:,0],z_intensity[:,1])

# plt.figure()
# plt.scatter(z_intensity[:,0],z_intensity[:,1] > 1500)

plt.show()
