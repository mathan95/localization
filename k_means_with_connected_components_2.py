# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:29:13 2018

@author: MathanP
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2

img=mpimg.imread("7.jpg")
A=img/255
plt.figure(1)

ish=A.shape
#X=np.zeros((ish[0]*ish[1],4),np.float32)
#temp_1=np.zeros((ish[0],ish[1]),np.float32)
#for i in range(0,ish[0],1):
#    for j in range(0,ish[1],1):
#        temp_1[i,j]=(i+j)/300
#        
#temp_1=np.reshape(temp_1,(ish[0]*ish[1]))
X=np.reshape(A,(ish[0]*ish[1],ish[2]))
#
#X[:,0:3]=A_new
#X[:,3]=temp_1
centroids=np.array([0,1,0])
errlist=np.zeros((X.shape[0],1),np.float64)

for p in range(0,X.shape[0],1):
    mat=X[p,:]
    mat3=mat-centroids
    err=np.sum(np.multiply(mat3,mat3))
    errlist[p]=err

threshold=0.7
errlist=np.greater(errlist,threshold)
X=np.multiply(X,errlist)
X_out=np.reshape(X,(ish[0],ish[1],ish[2]))

#X_1=np.float32(X)
## Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
#criteria = (cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0)
## Set flags (Just to avoid line break in the code)
#flags = cv2.KMEANS_RANDOM_CENTERS
## Apply KMeans
#ret,labels,centers = cv2.kmeans(X_1,20,None,criteria,50,flags)
#
#for i in range(0,len(centers),1):
#    err=centers[i,0:3]-[0,1,0]
#    err=np.sum(np.multiply(err,err))
#    if err<0.6:
#        centers[i,:]=0
#
#centers=np.uint8(centers*255)
#
#centers=centers[:,0:3]
#res = centers[labels.flatten()]
#res2 = np.reshape(res,(ish[0],ish[1],ish[2]))
#plt.imshow(res2)


#plt.subplot(1,2,1),plt.imshow(X_out)
#plt.subplot(1,2,2),plt.imshow(mpimg.imread('7.jpg'))


X_out=np.uint8(X_out*255)

img2=cv2.cvtColor(X_out,cv2.COLOR_BGR2GRAY)



ret, thresh = cv2.threshold(img2,0,1,cv2.THRESH_BINARY)

kernel = np.ones((10,10), np.uint8)


thresh = cv2.dilate(thresh, kernel, iterations=2)
thresh=cv2.erode(thresh,kernel,iterations=4)

plt.imshow(thresh)
plt.figure()
# You need to choose 4 or 8 for connectivity type
connectivity = 4  
# Perform the operation
output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The fourth cell is the centroid matrix
centroids = output[3]

list_new=[]
for i in range(0,num_labels,1):
    list1=np.equal(labels,i)
    if (np.sum(np.sum(list1))>600):
        list_new.append(i)


plt.subplot(1,2,1),plt.imshow(X_out)
#plt.subplot(1,2,2),plt.imshow(mpimg.imread('vg15.jpg'))

n=len(list_new)
shape=np.shape(img2)

print(list_new)
list1=np.zeros((n,4),np.int)
list1[:,0]=shape[0]
list1[:,2]=shape[1]


for i in range(0,shape[0]-1,1):
    for j in range(0,shape[1]-1,1):
        if ((labels[i,j] in list_new)==True and (labels[i,j]!=0)):
            b=np.argmax(np.equal(list_new,labels[i,j]))
            if list1[b,0]>i:
                list1[b,0]=i
            elif list1[b,1]<i:
                list1[b,1]=i
            if list1[b,2]>j:
                list1[b,2]=j
            elif list1[b,3]<j:
                list1[b,3]=j


for j in range(1,n,1):
    filter1=np.zeros((shape[0],shape[1],3),np.float64)
    filter1[:,:,0]=np.equal(labels,list_new[j])
    filter1[:,:,1]=np.equal(labels,list_new[j])
    filter1[:,:,2]=np.equal(labels,list_new[j])
    plt.figure(j+1)
    imb=np.multiply(X_out/255,filter1)
    im=img/255
    #plt.imshow(X_out[list1[j,0]:list1[j,1],list1[j,2]:list1[j,3],:])
    plt.imshow(im[max(0,list1[j,0]-50):min(shape[0],list1[j,1]+50),max(0,list1[j,2]-50):min(shape[1],list1[j,3]+50),:])
    #plt.imshow(imb)
    
print(list1)