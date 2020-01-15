from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import os
import glob
from scipy import ndimage
import math

f = r'/media/devici/srinath_dhm02/srinath_confocal/pillars1/00200cs0015mum_r1'
os.chdir(f)

dye_files = sorted(glob.glob('*Perylene.tif'), key=os.path.getmtime)
undye_files = sorted(glob.glob('*Reflection.tif'), key=os.path.getmtime)

n = np.shape(dye_files)[0]

#reference image to estiamter the tilt
ref_img = 75
tilt_ang = 00

extents = np.sin(np.deg2rad(ref_img))

x_px = Image.open(dye_files[ref_img]).size[0]
y_px = Image.open(dye_files[ref_img]).size[1]

s = (x_px + y_px)/4

initial = np.array(Image.open(dye_files[ref_img]))
final = ndimage.rotate(initial, tilt_ang)

final1 = final[int(s*(1-extents)):int(s*(1+extents)), int(s*(1-extents)):int(s*(1+extents))]

xnew_px = int(x_px*extents)
ynew_px = int(y_px*extents)

fig, ax = plt.subplots(1,2)

plt.subplot(1,2,1)
plt.title('Original image')
plt.xlim(1,x_px)
plt.ylim(1,y_px)
plt.xlabel('x [px]')
plt.ylabel('y [px]')
plt.xticks([1, x_px])
plt.yticks([1, y_px])
plt.imshow(initial, 'gray')

plt.subplot(1,2,2)
plt.title('Rotated and cropped image')
plt.xlim(1,xnew_px)
plt.ylim(1,ynew_px)
plt.xlabel('x [px]')
plt.ylabel('y [px]')
plt.xticks([1, xnew_px])
plt.yticks([1, ynew_px])
plt.imshow(final1,'gray')

plt.show()

kk = np.zeros(n)
kk1 = np.zeros(n)

count = -1

plt.figure()

for i in range(0,n):
    count = count + 1
    print(i)
    img = np.array(Image.open(dye_files[count]))
    img1 = np.array(Image.open(undye_files[count]))

    kk[count] = np.mean(img)
    kk1[count] = np.mean(img1)

    # plt.title('Histogram; Z = %03i' %count)
    # plt.xlim(1,256)
    # plt.ylim(1,xnew_px*ynew_px)
    # plt.ylim(1,50000)
    # plt.xticks([1, 256])
    # plt.yticks([1, xnew_px*ynew_px])
    # plt.hist(img.ravel(),256,[0,256])
    # # plt.hist(img1.ravel(),256,[0,256])
    #
    # plt.draw()
    # plt.pause(0.1)
    #
    # if i < n-1:
    #     plt.clf()

    kk[count] = img.sum()
    kk1[count] = img1.sum()

plt.figure()
plt.scatter(range(0,n),kk)
plt.scatter(range(0,n),kk1)
plt.show()
#
# plt.show()
