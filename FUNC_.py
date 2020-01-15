from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from scipy import stats

##########################################################################
#DHM background image analysis
def dhm_background_image_analysis(f, img1, img2, img3, conv):

    def_files = sorted(glob.glob('*.txt'), key=os.path.getmtime)
    n = np.shape(def_files)[0]                      #n = 2000

    x_px = np.shape(np.loadtxt(def_files[0]))[0]    #x_px = 900
    y_px = np.shape(np.loadtxt(def_files[0]))[1]    #y_px = 900

    #def_img = np.zeros((img3,x_px,y_px))           #Shows memory error. Why??
    avg_def = np.zeros((img3+1,2))
    avg_back = np.zeros((x_px,y_px))

    count = 0

    for i in range(0,img3+1):
        print(def_files[i])
        def_img = np.loadtxt(def_files[i])
        avg_def[i,0] = i
        avg_def[i,1] = def_img.mean()
        if i<img1:
            count = count + 1
            avg_back = avg_back + def_img

    avg_back = avg_back/(count)
    image = avg_back - np.loadtxt(def_files[img2+1])

    return image, avg_def, avg_back

##########################################################################

##########################################################################
#Deformation center at (img2+1)th image
def deformation_center(image):

    x_px = np.shape(image)[0]
    y_px = np.shape(image)[1]

    s = int(round((x_px + y_px)/4))                                 #s = 450

    delta_theta = round(np.degrees(np.arcsin(4/s)),1)   #delta_theta = 0.5
    n = int(round(360/delta_theta))                     #n = 720

    ph = np.zeros((n,s))
    s_max = np.zeros(n)
    x_max = np.zeros(n)
    y_max = np.zeros(n)

    for k in range(0,n):
        theta = k*(360/n)
        for j in range(0,s):
            xx = s+(j*np.cos(np.deg2rad(theta)))
            yy = s+(j*np.sin(np.deg2rad(theta)))
            ph[k,j] = image[int(round(yy)),int(round(xx))]

        s_max[k] = np.argmax(ph[k,:])
        x_max[k] = s + (s_max[k]*np.cos(np.deg2rad(theta)))
        y_max[k] = s + (s_max[k]*np.sin(np.deg2rad(theta)))

    xc = (max(x_max) + min(x_max))/2
    yc = (max(y_max) + min(y_max))/2

    return x_max, y_max, xc, yc, delta_theta
##########################################################################

##########################################################################
#read info file
def read_info_file(f):

    os.chdir(f)
    t = [x.split(' ')[3] for x in open('timestamps.txt').readlines()]

    os.chdir(os.getcwd() + r'/info')

    avg_back = np.loadtxt('avg_background.txt')
    avg_def = np.loadtxt('avg_deformation.txt')

    with open('details.txt') as f_lines:
        content = f_lines.readlines()

    img1 = int(content[1])
    img2 = int(content[2])
    img3 = int(content[3])

    xc = int(content[4])
    yc = int(content[5])

    return avg_back, avg_def, img1, img2, img3, xc, yc, t

##########################################################################

##########################################################################

def smooth(y, box_pts):

    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')

    return y_smooth

##########################################################################

##########################################################################

def power_law_fit(x, y):

    xx = np.log(x)
    yy = np.log(y)

    slope, intercept, *_ = stats.linregress(xx, yy)

    initial = np.exp(intercept)
    power = slope

    y_fit = initial*(x**power)

    return initial, power, y_fit

##########################################################################

##########################################################################

def average_profile(theta_start, theta_end, delta_theta, xc, yc, s, img):

    k1 = len(np.arange(theta_start, theta_end, delta_theta))
    k2 = len(range(0,s))

    haha = np.zeros(s)
    haha1 = np.zeros(s)
    count = -1

    for k in np.arange(theta_start, theta_end, delta_theta):
        count = count + 1
        theta = k
        b = image_profile(img, xc, yc, theta, s)
        haha = haha + b

    haha = haha/count

    haha1 = haha-np.mean(haha)
    # haha1 = haha+60

    # plt.figure()
    # plt.plot(haha, c='blue')
    # plt.plot(haha1, c='red')
    # plt.show()
    #
    # print(np.mean(haha))
    # input()

    # plt.plot(haha)
    # plt.show()

    return haha1

##########################################################################

##########################################################################

def image_profile(img_in, xc, yc, theta, s):

    profile_out = np.zeros(s)

    for j in range(0,s):
        xx = int(round(xc+(j*np.cos(np.deg2rad(-theta)))))
        yy = int(round(yc+(j*np.sin(np.deg2rad(-theta)))))
        profile_out[j] = img_in[yy,xx]

    return profile_out

################################################################################

################################################################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

################################################################################
