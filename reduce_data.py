## -*- coding: utf-8 -*-

import glob
from os import path
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pkl
import cv2

# Custom modules
import Functions as Fun
import VideoFunctions as VF
import ImageProcessingFunctions as IPF

# User parameters
folder = '..\\Data\\Cole\\220616\\temp\\'
fileString = '*_data1.pkl'
saveName = '../Data/Cole/220616/temp/Data.pkl'
step = 1
viewResults = True
diskScale = 15 # radius in [cm]
#QList = [500,1000,1500,2000,2500,3000]#,3250,3500]
#RPMList = [0]#,30,120,250,500,750,1000]

allData = {}

   
# Cleanup existing windows
plt.close('all')
    
# Get list of video files to process
pathToFiles = path.join(folder,fileString)
fileList = glob.glob(pathToFiles)
nFiles = len(fileList)

for i in range(0,nFiles):
    try:
        # Parse the filename to get video info
        dataPath = fileList[i]
        keyName = path.split(dataPath)[1][:-9]
        
        
        RPM = VF.get_RPM_from_file_name(dataPath)
        Q = VF.get_flowRate_from_file_name(dataPath)
        
        
        
        # Load data
        with open(dataPath,'rb') as f:
            container = pkl.load(f)
        
        # Parse stored data
        dataFrames = sorted(container['data'].keys())
        N = len(dataFrames)
        center = container['maskData']['diskCenter']
        diskRadius = container['maskData']['diskRadius']
        maskRadius = container['maskData']['maskRadius']
        fps = container['fps']
        t0Frame = container['t0Frame']
        flowRate = container['flowRate']
        nozzleMask = container['maskData']['nozzleMask']
        kernel = np.ones((5,5),np.uint8)
        dilation = cv2.erode(np.uint8(nozzleMask),kernel,iterations = 1)
        nozzleMask = (dilation==1)    
        
        # Initialize variables to plot
        aMax = np.zeros(N)
        aMin = np.zeros(N)
        aMean = np.zeros(N)
        aMax2 = np.zeros(N)
        time = np.zeros(N)
        aMin2 = np.zeros(N)
        area = np.zeros(N)
        perimeter = np.zeros(N)
        dTheta = np.zeros(N)
        
        
        # Loop through frames in the video and track wetted area
        for j in range(0,N,step):
            
            # Load data
            num = dataFrames[j]
            time[j] = num/fps
            contour = np.float32(container['data'][num])
            
            if len(contour) <= 2:
                continue
            temp = nozzleMask[np.int16(contour[:,1]),np.int16(contour[:,0])]
            contour = contour[temp]
            if len(contour) < 2:
                continue
            # Remove points along the nozzle mask
            xData = contour[:,0]
            yData = contour[:,1]
            diffX = xData[1:]-xData[:-1]
            diffY = yData[1:]-yData[:-1]
            if max(diffX) > max(diffY):
                ind = np.argmax(diffX) + 1
            else:
                ind = np.argmax(diffY) + 1
            x0 = np.array([center[0]])
            y0 = np.array([center[1]])
            xData = np.concatenate((x0,xData[ind:],xData[:ind],x0))
            yData = np.concatenate((y0,yData[ind:],yData[:ind],y0))
            points = [(xData[i],yData[i]) for i in range(len(xData))]
            temp = IPF.create_polygon_mask(nozzleMask,points)[0]
            area[j] = np.sum(temp)*(diskScale/diskRadius)**2
            
            # Shift data relative to center of disk and scale
            xData = (xData - center[0])/diskRadius*diskScale
            yData = (-yData + center[1])/diskRadius*diskScale
            # Compute radius, and angle of each point
            phi = np.arctan2(yData,xData)
            phi[phi<0] += 2*np.pi
            a = np.sqrt(xData**2 + yData**2)
            a[a>maskRadius/diskRadius*diskScale] = maskRadius/diskRadius*diskScale
    
            temp = phi[-2]-phi[1]
            if temp < 0:
                dTheta[j] = 2*np.pi + temp
            else:
                dTheta[j] = temp
            aMax[j] = np.max(a)
            aMin[j] = np.min(a)
            aMean[j] = np.mean(a)
            a98 = np.percentile(a,98)
            a02 = np.percentile(a,2)
            aMax2[j] = np.mean(a[a>=a98])
            aMin2[j] = np.mean(a[a<=a02])
            # Compute excess perimeter
            stride = 10
            x1 = xData[1:-1:stride]
            y1 = yData[1:-1:stride]
            ds = np.sqrt((x1[1:]-x1[:-1])**2 + (y1[1:]-y1[:-1])**2)
            perimeter[j] = np.sum(ds)
        
            if viewResults:
                plt.figure(66)
                plt.cla()
                plt.plot(xData,yData,'k.-')
                plt.plot(xData[1],yData[1],'g.',xData[-2],yData[-2],'r.')
                Fun.plot_circles(diskScale,(0,0),1)
                Fun.plot_circles(maskRadius/diskRadius*diskScale,(0,0),1)
                plt.axis('scaled')
                plt.pause(0.001)
             
        # store data
        data = {}
        data['condition'] = path.split(path.split(dataPath)[0])[1]
        data['flowRate'] = flowRate
        data['RPM'] = RPM
        data['fps'] = fps
        data['t0'] = t0Frame/fps
        data['time'] = time[time>0]
        data['aMax'] = aMax[time>0]
        data['aMax2'] = aMax2[time>0]
        data['aMin'] = aMin[time>0]
        data['aMin2'] = aMin2[time>0]
        data['aMean'] = aMean[time>0]
        data['area'] = area[time>0]
        data['dTheta'] = dTheta[time>0]
        data['perimeter'] = perimeter[time>0]
        data['excessPerimeter'] = perimeter[time>0]/np.sqrt(2*area[time>0]* \
                                        dTheta[time>0])
        allData[keyName] = data
        print 'Processed %s; file %i of %i' %(keyName,i,nFiles)
        
        plt.figure()
        plt.plot(time,perimeter/np.sqrt(2*area*dTheta))
        plt.pause(0.001)
    except:
        continue
    
with open(saveName,'wb') as f:
    pkl.dump(allData,f)
