## -*- coding: utf-8 -*-

import glob
from os import path
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pkl

# Custom modules
import Functions as Fun
import ImageProcessingFunctions as IPF

# User parameters
folder = '..\\Data\\Demo\\'
fileString = '*\\*_data.pkl'
saveName = folder + 'all_data1.pkl'
step = 1
diskScale = 15 # radius in [cm]
restrictAngle = False
t0 = 0
t1 = 90
stride = 10

# Parameters for testing
viewResults = False

# Cleanup existing windows
plt.close('all')
    
# Get list of video files to process and initialize data dictionary
pathToFiles = path.join(folder,fileString)
fileList = glob.glob(pathToFiles)
nFiles = len(fileList)
allData = {}

for i in range(0,nFiles):
    # Use Try/Catch to skip bad files without stopping analysis
    try:
        # Parse the filename to get video info and load data container
        dataPath = fileList[i]
        keyName = path.split(dataPath)[1][:-9]
        condition = path.split(path.split(dataPath)[0])[1]
        with open(dataPath,'rb') as f:
            container = pkl.load(f)
        
        # Parse stored data
        dataFrames = sorted(container['data'].keys())
        N = len(dataFrames)
        center = container['maskData']['diskCenter']
        pix_per_cm = container['maskData']['diskRadius']/diskScale
        maskRadius = container['maskData']['maskRadius']/pix_per_cm # [cm]
        fps = container['fps']
        RPM = container['RPM']
        t0Frame = container['t0Frame']
        flowRate = container['flowRate']
        nozzleMask = container['maskData']['nozzleMask']      
        
        # Operate on the nozzle mask so that it will remove contour around it
        dilatedMask = IPF.dilate_mask(nozzleMask)
        if RPM > 0: # Convert to 8 bit image for rotation
            dilatedMask = dilatedMask*np.uint8(255)
        else:
            mask = dilatedMask
        
        # Initialize variables to store in reduced data
        time = np.zeros(N)
        aMax = np.zeros(N)
        aMin = np.zeros(N)
        aMean = np.zeros(N)
        aMax2 = np.zeros(N)
        aMin2 = np.zeros(N)
        area = np.zeros(N)
        perimeter = np.zeros(N)
        dTheta = np.zeros(N)
        
        
        # Loop through frames in the video and track wetted area
        for j in range(0,N,step):
        
            try: # Use Try/Catch to skip bad frames
                # Load data 
                num = dataFrames[j]
                contour1 = np.float32(container['data'][num])
                if len(contour1) <= 2: # Skip frames without enough points
                    continue
                theta = container['theta'][num]%(-360)
                
                # Adjust the rotation of the mask to match the stored data
                if RPM > 0:
                    mask = IPF.rotate_image(dilatedMask,theta,
                                            size=np.shape(nozzleMask))
                    mask = (mask == 255)
                    
                # Remove contour points that are covered by the mask
                temp = mask[np.int16(contour1[:,1]),np.int16(contour1[:,0])]
                ind0 = np.argmax(temp == False)
                ind1 = len(temp) - np.argmax(temp[::-1] == False) - 1
                # Reorder temp if mask is covering beginning or end of temp
                if ((ind0 == 0) or (ind1 == len(temp)-1)) or (ind1-ind0>0.75*len(temp)):
                    halfInd = len(temp)/2
                    temp = np.concatenate((temp[halfInd:],temp[:halfInd]))
                    ind0 = np.argmax(temp == False)
                    ind1 = len(temp) - np.argmax(temp[::-1] == False) - 1
                    temp[ind0:ind1] = False
                    midInd = len(temp)-halfInd
                    temp = np.concatenate((temp[midInd:],temp[:midInd]))
                else:
                    temp[ind0:ind1] = False
                contour = contour1[temp]
                if len(contour) < 2:
                    continue
                # Convert to x and y values and reorder so that the first 
                # point starts at the right edge of the nozzle and goes
                # counter-clockwise along the wave front
                xData = contour[:,0]
                yData = contour[:,1]
                diffX = xData[1:]-xData[:-1]
                diffY = yData[1:]-yData[:-1]
                diffD = np.sqrt(diffX**2 + diffY**2)
                if np.max(diffD) > 3:
                    ind = np.argmax(diffD) + 1
                    xData = np.concatenate((xData[ind:],xData[:ind]))
                    yData = np.concatenate((yData[ind:],yData[:ind]))
                    
                if restrictAngle:
                    # Compute the angle of each point relative to disk center
                    xData1 = (xData - center[0])
                    yData1 = (-yData + center[1])
                    phi = np.arctan2(yData1,xData1)
                    phi[phi<0] += 2*np.pi
                    # Shift the indices so that it starts at the last point 
                    # where the angle crosses zero
                    diffPhi = phi[1:] - phi[:-1]
                    ind = len(diffPhi) - np.argmax(diffPhi[::-1]<-np.pi)
                    phi = np.concatenate((phi[ind:],phi[:ind]))
                    xData  = np.concatenate((xData[ind:],xData[:ind]))
                    yData  = np.concatenate((yData[ind:],yData[:ind]))
                    # Remove wrap around values from phi
                    diffPhi = phi[1:] - phi[:-1]
                    temp = np.concatenate(([False],(diffPhi < -np.pi)))
                    phi[temp] += 2*np.pi
                    # Rotate the cutoff angles and find corresponding indices
                    tStart = (t0 + theta)/180*np.pi
                    if tStart < 0:
                        tStart += 2*np.pi
                    tEnd = (t1 + theta)/180*np.pi
                    if tEnd < 0: 
                        tEnd += 2* np.pi
                    ind0 = np.argmax(phi>tStart)
                    ind1 = len(phi) - np.argmax(phi[::-1]<tEnd) - 1
                    # Remove points outside the specified angle range
                    if ind1>ind0:
                        xData = xData[ind0:ind1]
                        yData = yData[ind0:ind1]
                    else:
                        xData = np.concatenate((xData[ind0:],xData[:ind1]))
                        yData = np.concatenate((yData[ind0:],yData[:ind1]))
                    if len(xData) < 2*stride:
                        continue
                    
                # Produce a mask of the effective wetted area
                x0 = np.array([center[0]])
                y0 = np.array([center[1]])
                xData1 = np.concatenate((x0,xData[ind:],xData[:ind],x0))
                yData1 = np.concatenate((y0,yData[ind:],yData[:ind],y0))
                points = [(xData1[z],yData1[z]) for z in range(len(xData1))]
                wettedArea = IPF.create_polygon_mask(mask,points)[0]
                wettedArea = np.sum(wettedArea)/(pix_per_cm**2)
                
                # Shift data relative to center of disk and scale
                xData = (xData - center[0])/pix_per_cm
                yData = (-yData + center[1])/pix_per_cm
                        
                if viewResults:
                    plt.figure(66)
                    plt.cla()
                    plt.plot(xData,yData,'k.-')
                    plt.plot(xData[0],yData[0],'g.',xData[-1],yData[-1],'r.')
                    Fun.plot_circles(diskScale,(0,0),1)
                    Fun.plot_circles(maskRadius,(0,0),1)
                    plt.axis('scaled')
                    plt.pause(0.001)
                    
                # Compute radius, and angle of each point
                phi = np.arctan2(yData,xData)
                phi[phi<0] += 2*np.pi
                a = np.sqrt(xData**2 + yData**2)
                # Compute the range of angles analyzed on disk
                dPhi = phi[-1]-phi[0]
                if dPhi < 0:
                    dPhi += 2*np.pi
                # Compute contour length of points
                x1 = xData[::stride]
                y1 = yData[::stride]
                ds = np.sqrt((x1[1:]-x1[:-1])**2 + (y1[1:]-y1[:-1])**2)
                length = np.sum(ds)
                
                
                # Store all reduced variables of interest
                time[j] = num/fps
                area[j] = wettedArea
                dTheta[j] = dPhi
                aMax[j] = np.max(a)
                aMin[j] = np.min(a)
                aMean[j] = np.mean(a)
                aMax2[j] = np.percentile(a,98)
                aMin2[j] = np.percentile(a,2)
                perimeter[j] = length
                
            except:
                print 'Analysis failed on frame number %i' %j
             
        # Compute the excess perimeter
        eP = perimeter/(dTheta*np.sqrt(area/np.pi))
        # store reduced data in dictionary
        data = {}
        data['dilatedMask'] = dilatedMask
        data['stride'] = stride
        data['condition'] = condition
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
        data['excessPerimeter'] = eP[time>0]
        allData[keyName] = data
        print 'Processed %s; file %i of %i' %(keyName,i+1,nFiles)

    except:
        print 'Error on %s' %keyName
    
with open(saveName,'wb') as f:
    pkl.dump(allData,f)