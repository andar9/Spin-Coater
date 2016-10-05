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
folder = '..\\Data\\Cole\\3.8.16\\CS\\Fixed RPM\\'
fileString = '*.mp4'
diskFraction = 0.90 # 0.98 if dry; 0.95 if wet
step = 1
threshold1_ = 32  #Change back to 12 
threshold1_0RPM = 5
threshold2_ = 8
threshold2_0RPM = 5
checkMasks = False # Check quality of predrawn masks for each video
maskFile = 'maskData_3OCT2016.pkl'
refPointFile = 'refPoint_3OCT2016.pkl'
homographyFile = 'offCenterTopView_3OCT2016.pkl' # Use None for no transform
viewResults = True
saveInterval = 600

saveData = True
continueAnalysis = True

#############################################################################
observedFractionStep = 0.01
probeFraction = 0.005
observedFraction = 0.1
#############################################################################
   
# Cleanup existing windows
plt.close('all')
destroyAllWindows()
t0 = tic()
t1 = tic()
    
# Get list of video files to process
pathToFiles = os.path.join(folder,fileString)
fileList = glob.glob(pathToFiles)
dataList = glob.glob(os.path.join(folder,'*_data1.pkl'))
N = len(fileList)
hMatrix = 0

for i in range(0,N,1):
    # Parse the filename to get video info
    videoPath = fileList[i]
    dataFile = fileList[i][:-4] + '_data1.pkl'
    RPM = VF.get_RPM_from_file_name(videoPath)
    fps = VF.get_FPS_from_file_name(videoPath)
    flowRate = VF.get_flowRate_from_file_name(videoPath)
    vid = VF.get_video_object(videoPath)
    # Set thresholds for image processing to find boundaries
    if RPM > 0:
        threshold1 = threshold1_
        threshold2 = threshold2_
    else:
        threshold1 = threshold1_0RPM
        threshold2 = threshold2_0RPM
        if fps < 1000:
            threshold1 *= 3
    
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
                                          intensityRegion)
    else:
        continue
    
    # Parse video and stored data
    nFrames = int(vid.get(7)) # number of frames in video
    t0Frame = container['t0Frame']
    data = np.array([(0,0)])
    frameNumber = max(container['data'].keys()) # Picks up at last analyzed frame
    temp = os.path.split(videoPath)[1]
    print 'picking up analysis of %s from frame # %i of %i' %(temp,frameNumber,
                                                              nFrames)
    wettedArea = container['wettedArea']
    frame = VF.extract_frame(vid,frameNumber,hMatrix,maskData)
    theta = -RPM/60./fps*360*(frameNumber-t0Frame)
    check = copy.copy(maskData['mask'])
    
    # Loop through frames in the video and track wetted area
    for j in range(frameNumber,nFrames,step):
        
        # Show an update of results
        if (viewResults) or (not j%saveInterval):
            plt.figure(66)
            plt.cla()
            if theta != 0:
                frame = IPF.rotate_image(frame,theta,size=frame.shape)
            tempFrame = copy.copy(frame)
            tempFrame[wettedArea] = 255

            tempFrame = np.dstack((tempFrame,frame,frame))
            Fun.plt_show_image(tempFrame)
            plt.plot(data[:,0],data[:,1],'b')
            plt.title(temp)
            plt.pause(.001)
            
        # Compute the rotation angle relative to the first frame
        theta = -RPM/60./fps*360*(j-t0Frame)
        frame = VF.extract_frame(vid,j,hMatrix,maskData)
        # Get the reference frame for background subtraction
        if RPM == 0:
            ind = j-1
        else:
            ind = t0Frame
        ref = VF.extract_frame(vid,ind,hMatrix,maskData)
            
        wettedArea = IPF.process_frame(frame,ref,wettedArea,theta,
                                       threshold1,threshold2)
        wettedArea *= maskData['diskMask']
        
#############################################################################
        testMaskData = copy.copy(maskData)
        testMaskData2 = copy.copy(testMaskData)
        observedMaskData = IPF.reduce_mask_radius(testMaskData,observedFraction)
        probeMaskData = IPF.reduce_mask_radius(testMaskData2,observedFraction + probeFraction)
        while True:
            if not np.any(wettedArea*probeMaskData['diskMask'] - wettedArea*observedMaskData['diskMask']):
                wettedArea *= observedMaskData['diskMask']
                break
            else:
                observedFraction += observedFractionStep
                testMaskData = copy.copy(maskData)
                testMaskData2 = copy.copy(testMaskData)
                observedMaskData = IPF.reduce_mask_radius(testMaskData,observedFraction)
                probeMaskData = IPF.reduce_mask_radius(testMaskData2,observedFraction + probeFraction)
                print 'Observed fraction extended to %f.' %observedFraction
#############################################################################                    

        
        
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
