## -*- coding: utf-8 -*-

import glob
import os
from cv2 import destroyAllWindows
import matplotlib.pyplot as plt
import copy
import numpy as np
import cPickle as pkl
from time import time as tic

# Custom modules
import Functions as Fun
import VideoFunctions as VF
import ImageProcessingFunctions as IPF
import UserInputFunctions as UIF

# User parameters
folder = '..\\Data\\New Disk\\6.5mM SDS\\'
fileString = '**.mp4'
suffix = '_data.pkl'
#fileString = '*Dry*750RPM*.mp4'
diskFraction = 0.95
splashFraction = np.linspace(0.15,1.1,65)#[0.25,0.33,0.4,0.5,0.67,1.1]
step = 1
threshold1_ = 3
threshold1_0RPM_ = 4
threshold2_ = 5
threshold2_0RPM_ = 4
tShift_ = 4
checkMasks = False # Check quality of predrawn masks for each video
maskFile = 'maskData_1APR2016.pkl'
refPointFile = 'refPoint_31OCT2015.pkl'
homographyFile = 'offCenterTopView_1APR2016.pkl' # Use None for no transform
viewResults = True
viewInterval = 1
saveInterval = 600
saveData = True
continueAnalysis = True
figNum = 123


   
# Cleanup existing windows
plt.close('all')
destroyAllWindows()
t0 = tic()
t1 = tic()
    
# Get list of video files to process
pathToFiles = os.path.join(folder,fileString)
fileList = glob.glob(pathToFiles)
dataList = glob.glob(os.path.join(folder,'*' + suffix))
N = len(fileList)
hMatrix = 0

for i in range(0,N,1):
    # Parse the filename to get video info
    videoPath = fileList[i]
    print os.path.split(videoPath)[1]
    dataFile = fileList[i][:-4] + suffix
    RPM = VF.get_RPM_from_file_name(videoPath)
    fps = VF.get_FPS_from_file_name(videoPath)
    flowRate = VF.get_flowRate_from_file_name(videoPath)
    if flowRate > 2000:
        splashFraction = np.append(np.linspace(0.2,0.67,30),
                                   np.linspace(0.7,1.1,10))
        threshold1_0RPM = 7
        threshold2_0RPM = 7
        tShift = 1
    else:
        splashFraction = np.linspace(0.2,1.1,20)
        threshold1_0RPM = threshold1_0RPM_
        threshold2_0RPM = threshold2_0RPM_
        tShift = tShift_
        
    if (flowRate >= 3000) and (RPM > 0):
        threshold1_ = 25
    else:
        threshold1_ = 20
    vid = VF.get_video_object(videoPath)
    
    # Set thresholds for image processing to find boundaries
    if RPM > 0:
        threshold1 = threshold1_
        threshold2 = threshold2_
    else:
        threshold1 = threshold1_0RPM
        threshold2 = threshold2_0RPM
#        if fps < 1000:
#            threshold1 *= 3
    
    # Ignore videos for which the reference frame has been rotated
    if '_rot' in videoPath:
        continue
    
    # Load homography matrix then mask data
    if hMatrix is 0:
        # Load homography matrix
        hMatrix = UIF.get_homography_matrix(homographyFile,vid)
        
        # Get mask data for ignoring areas not of interest
        maskData = UIF.get_mask_data(maskFile,vid,hMatrix,checkMasks)
        if diskFraction < 1:
            maskData = IPF.reduce_mask_radius(maskData,diskFraction)
        
        # Get the region at which image intensity will be tracked
        intensityRegion = UIF.get_intensity_region(refPointFile,vid)
    
    # Find the "first" frame for beginning the analysis of the video and load
    # the data structure for storing the image processing data
    if (dataFile not in dataList) or continueAnalysis:
        container = VF.get_data_structure(vid,fps,RPM,flowRate,dataFile,
                                          dataList,hMatrix,maskData,
                                          intensityRegion,splashFraction)
    else:
        print 'Already analyzed: %s' %videoPath
        continue
    
    # Parse video and stored data
    nFrames = int(vid.get(7)) # number of frames in video
    t0Frame = container['t0Frame']
    data = np.array([(0,0)])
    frameNumber = max(container['data'].keys())
    temp = os.path.split(videoPath)[1]
    print 'picking up analysis of %s from frame # %i of %i' %(temp,frameNumber,
                                                              nFrames)
    wettedArea = container['wettedArea']
    frame = VF.extract_frame(vid,frameNumber,hMatrix,maskData)
    theta = -RPM/60./fps*360*(frameNumber-t0Frame)
    check = copy.copy(maskData['mask'])
    
    # Create splash mask
    center = np.shape(frame)[0:2]#maskData['diskCenter']
    center = [center[1]/2.0,center[0]/2.0]
    R = splashFraction[container['splashIndex']]*maskData['diskRadius']
    splashMask = IPF.create_circular_mask(frame,R,center)
    
    # Loop through frames in the video and track wetted area
    for j in range(frameNumber,nFrames,step):
        
        # Show an update of results
        if ((viewResults) and (not j%viewInterval)) or (not j%saveInterval):
            plt.figure(figNum)
            plt.cla()
            if theta != 0:
                frame = IPF.rotate_image(frame,theta,size=frame.shape)
            tempFrame = copy.copy(frame)
            tempFrame[wettedArea] = 255

            tempFrame = np.dstack((tempFrame,frame,frame))
            Fun.plt_show_image(tempFrame)
            plt.plot(data[:,0],data[:,1],'b',center[0],center[1],'bx')
            plt.title(temp)
            plt.pause(.001)
            
        # Compute the rotation angle relative to the first frame
        theta = -RPM/60./fps*360*(j-t0Frame)
        frame = VF.extract_frame(vid,j,hMatrix,maskData)
        # Get the reference frame for background subtraction
        if RPM == 0:
            ind = max([j-tShift,t0Frame])
        else:
            if theta%(360) == 0:
                print 'Exactly integer rotation from t0Frame at %i' %(j)
                ind = t0Frame
        ref = VF.extract_frame(vid,ind,hMatrix,maskData)
            
        wettedArea = IPF.process_frame(frame,ref,wettedArea,theta,
                                       threshold1,threshold2)
#        wettedArea *= maskData['diskMask']
        if np.any(wettedArea):
            data = IPF.get_perimeter(wettedArea)
        else:
            data = np.array([(0,0)])
        # Store data in container                                  
        container['wettedArea'] = wettedArea
        container['data'][j] = data
        container['theta'][j] = theta
        # Save periodically
        if (not j%saveInterval) and (saveData):
            
            with open(dataFile,'wb') as f:
                pkl.dump(container,f)
            print '%i of %i completed after %f s.' %(j,nFrames,tic()-t1)
            t1 = tic()
            
        # Check progress of wavefront
        if len(data) > 10:
            rData = np.sqrt((data[:,0]-center[0])**2 + (data[:,1]-center[1])**2)
        else:
            rData = 0
        # Remove splash marks
        if (np.max(rData) >= 0.99*R)\
            and (not container['splashRemoved'][container['splashIndex']]):
            wettedArea *= splashMask
            container['splashRemoved'][container['splashIndex']] = True
            container['splashIndex'] += 1
            R = splashFraction[container['splashIndex']]*maskData['diskRadius']
            splashMask = IPF.create_circular_mask(frame,R,center)
            print 'Splash marks removed at frame # %i' %(j)
            
            
        check[wettedArea] = False
#        print 'Number of dry pixels = ', np.sum(check)
        if np.sum(check) > 100:
            continue
        else:
            print 'Disk fully wet. Moving to next video.'
            break
        
    # Save at end
    if saveData:
        with open(dataFile,'wb') as f:
            pkl.dump(container,f)
    print 'Video %g of %g complete. Elapsed time: %f s.' %(i+1,N,tic()-t0)