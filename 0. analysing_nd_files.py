from pims import ND2_Reader
import os
import numpy as np
from matplotlib import pyplot as plt
from numpy import array
import cv2 as cv
import glob

################################################################################

f = r'/media/devici/srinath_dhm02/srinath_confocal/pillars1/00200cs0015mum_inverted_r1'
os.chdir(f)

frames = ND2_Reader('00200cs0015mum_inverted_r1.nd2')
matr = array(frames)

channels = matr.shape[0]
z_stacks = matr.shape[1]
x_px = matr.shape[2]
y_px = matr.shape[3]

pillar_height = 10*(10**-6)
pillar_diameter = 30*(10**-6)
min_pillar_to_pillar_distance = 30*(10**-6)

n_ref = 1.403

z_res = 0.725*(10**-6)
r_res = 1.2429611*(10**-6)
phi = 1 - ((np.pi/4)*(((pillar_diameter)/(pillar_diameter + min_pillar_to_pillar_distance))**2))

################################################################################

b = np.zeros((z_stacks, x_px))
z_intensity = np.zeros((z_stacks,3))

for i in range(0,x_px):
    for j in range(0,z_stacks):
        b[j,i] = matr[0,j,i,int(y_px/2)]

for j in range(0,z_stacks):
    z_intensity[j,0] = j
    z_intensity[j,1] = np.mean(matr[0,j,:,:])
    # z_intensity[j,2] = np.amax(matr[0,j,:,:]) - np.amin(matr[0,j,:,:])
    z_intensity[j,2] = matr[0,j,:,:].std()

for k in range(0,len(z_intensity[:,0])):
    if z_intensity[k,1] > phi*max(z_intensity[:,1]):
        start_interface = k
        break

for k in range(start_interface,len(z_intensity[:,0])):
    if z_intensity[k,1] < phi*max(z_intensity[:,1]):
        end_interface = k - 1
        break

transistion = np.argmax(z_intensity[:,1]) + (np.argmax(z_intensity[:,1]) - start_interface)

plt.figure()
plt.scatter(z_intensity[:,0],z_intensity[:,1])
plt.plot(z_intensity[:,0],z_intensity[:,1],'green')
plt.axhline(y=phi*max(z_intensity[:,1]), linestyle='--', color='red')
plt.axvline(x=start_interface, linestyle='--', color='black')
plt.axvline(x=end_interface, linestyle='--', color='black')
plt.axvline(x=transistion, linestyle='--', color='black')
plt.grid(True)

plt.figure()
plt.scatter(z_intensity[:,0],z_intensity[:,2])
plt.plot(z_intensity[:,0],z_intensity[:,2],'green')
plt.axvline(x=start_interface, linestyle='--', color='black')
plt.axvline(x=end_interface, linestyle='--', color='black')
plt.axvline(x=transistion, linestyle='--', color='black')
plt.grid(True)

plt.figure()
plt.imshow(b,cmap = 'inferno')
# plt.imshow(b[start_interface:end_interface,:],cmap = 'inferno')
plt.axhline(y=start_interface, linestyle='--', color='green')
plt.axhline(y=end_interface, linestyle='--', color='green')
plt.axhline(y=transistion, linestyle='--', color='green')

pillar_height = abs(transistion - end_interface)*z_res*n_ref*(10**6)
oil_over_layer = abs(transistion - start_interface)*z_res*n_ref*(10**6)

print(r'pillar height = %0.2f $\mu m$' %pillar_height)
print(r'oil over layer = %0.2f $\mu m$' %oil_over_layer)

plt.show()

################################################################################
