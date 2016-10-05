# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:52:03 2015

@author: John
"""

import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
import cPickle as pkl
from skimage import filters
from skimage.morphology import disk
from pandas import unique

# Custom modules
import Functions as Fun
import UserInputFunctions as UIF
        

def get_mask(X,Y,imageShape):
    """
    Converts arrays of x- and y-values into a mask. The x and y values must be
    made up of adjacent pixel locations to get a filled mask.
    """
    # Take only the first two dimensions of the image shape
    if len(imageShape) == 3:
        imageShape = imageShape[0:2]
    # Convert to unsigned integer type to save memory and avoid fractional 
    # pixel assignment
    X = X.astype('uint16')
    Y = Y.astype('uint16')
    
    #Initialize mask as matrix of zeros
    mask = np.zeros(imageShape,dtype='uint8')
    # Set boundary provided by x,y values to 255 (white)
    mask[Y,X] = 255
    # Fill in the boundary (output is a boolean array)
    mask = ndimage.morphology.binary_fill_holes(mask)
    
    return mask
    
def create_circular_mask(image,R,center):
    """
    Create 2D mask (boolean array) with same dimensions as input image
    where everything outside of the circle is masked.
    """
    # Calculate the number of points needed based on the size of the radius in
    # pixels (2 points per unit pixel)
    nPoints = int(4*np.pi*R)
    # Generate X and Y values of points on circle
    x,y = Fun.generate_circle(R,center,nPoints)
    mask = get_mask(x,y,np.shape(image))
    
    return mask
    
def create_polygon_mask(image,points):
    """
    Create 2D mask (boolean array) with same dimensions as input image
    where everything outside of the polygon is masked.
    """
    # Calculate the number of points needed perimeter of the polygon in
    # pixels (4 points per unit pixel)
    points = np.array(points,dtype=int)
    perimeter = cv2.arcLength(points,closed=True)
    nPoints = int(2*perimeter)
    # Generate x and y values of polygon
    x = points[:,0]; y = points[:,1]
    x,y = Fun.generate_polygon(x,y,nPoints)
    points = [(int(x[i]),int(y[i])) for i in range(nPoints)]
    points = np.asarray(list(unique(points)))
    mask = get_mask(x,y,image.shape)

    return mask, points
    
def mask_image(image,mask):
    """
    Returns image with all pixels outside mask blacked out
    mask is boolean array or array of 0s and 1s of same shape as image
    """
    # Apply mask depending on dimensions of image
    temp = np.shape(image)
    maskedImage = np.zeros_like(image)
    if len(temp) == 3:
        for i in range(3):
            maskedImage[:,:,i] = mask*image[:,:,i]
    else:
        maskedImage = image*mask
    
    return maskedImage
    
def create_mask_data(image,maskFile):
    """
    Create a dictionary containing the mask information from the given image
    and save to a pickle file.
    """
    # parse input
    plt.gray()
    plt.close()
    message = 'Click on points at the outer edge of the disk for mask'
    R,center = UIF.define_outer_edge(image,'circle',message)
    mask1 = create_circular_mask(image,R,center)
    plt.figure()
    plt.imshow(image)
    image = mask_image(image,mask1)
    plt.figure()
    plt.imshow(image)
    message = 'Click points around the nozzle and tubing for mask'
    points = UIF.define_outer_edge(image,'polygon',message)
    mask2, temp = create_polygon_mask(image,points)
    # invert polygon mask and combine with circle mask
    mask2 = (mask2 != True)
    mask = (mask2 == mask1)*mask1
    image = mask_image(image,mask)
    Fun.plt_show_image(image)
    
    maskData = {}
    maskData['mask'] = mask
    maskData['diskMask'] = mask1
    maskData['nozzleMask'] = mask2
    maskData['diskCenter'] = center
    maskData['diskRadius'] = R
    maskData['nozzlePoints'] = points
    maskData['maskRadius'] = R
    
    with open(maskFile,'wb') as f:
        pkl.dump(maskData,f)
        
    return maskData
    
def reduce_mask_radius(maskData,fraction):
    """
    Reduce the radius of the mask that covers the outer edge of the disk to the
    requested percentage.
    """
    # parse input
    R = maskData['diskRadius']
    center = maskData['diskCenter']
    mask = maskData['mask']
    # Redine the outer circular edge
    mask1 = create_circular_mask(mask,R*fraction,center)
    mask = mask*mask1
    # Save new mask data
    maskData['mask'] = mask
    maskData['diskMask'] = mask1
    maskData['maskRadius'] = R*fraction
    
    return maskData
    
def define_homography_matrix(image,hFile,scale=1.1):
    """
    Use an image of the wafer chuck from a given camera orientation to create 
    a homography matrix for transforming the image so that the wafer appears
    to be a perfect circle with the center at the center of the image.
    
    The user will need to identify the wafer chuck posts in the image by 
    clicking on the center of the small cylinder on the top of the post 
    in the following order:
    First, the post farthest to the right, then proceeding in clockwise fashion
    around the edge of the chuck. The first 6 points will be used.
    If the first post is different it will rotate the image and if the 
    direction is changed it will mirror the image.
    """
    # Get the post locations in the original image
    message = 'Click on the center of the 6 posts holding the disk. \n' + \
        'Start at the right side and proceed clockwise. \n' + \
        'Center click when finished'
    oldPoints = UIF.define_outer_edge(image,'polygon',message)
    oldPoints = np.array(oldPoints[:6])
    # Get the size of the original image and use it define the size of the 
    # disk in the new image
    temp = np.shape(image)
    R = scale*Fun.get_distance(oldPoints[0,:],oldPoints[3,:])/2.0
    center = [np.mean(oldPoints[:,0]),np.mean(oldPoints[:,1])]
    t0 = Fun.get_angle([center[0]+10,center[1]],center,oldPoints[0])
    print t0
    # Define the locatoins of the posts in the new image
    x,y = Fun.generate_circle(R,center,7,t0)
    imageCenter = [temp[1]/2.0,temp[0]/2.0]
    dX = center[0] - imageCenter[0]
    dY = center[1] - imageCenter[1]
    x -= dX
    y -= dY
    newPoints = np.array([(x[i],y[i]) for i in range(6)])
    # Define the homography matrix
    H,M = cv2.findHomography(oldPoints,newPoints)
    # View the resulting transformation
    newImage = cv2.warpPerspective(image,H,(temp[1],temp[0]))
    plt.figure()
    plt.subplot(121)
    Fun.plt_show_image(image)
    plt.plot(x,y,'go',oldPoints[:,0],oldPoints[:,1],'ro')
    plt.subplot(122)
    Fun.plt_show_image(newImage)
    plt.plot(x,y,'go',oldPoints[:,0],oldPoints[:,1],'ro')
    plt.savefig('../Figures/Homography.png')
    plt.ginput()
    
    with open(hFile,'wb') as f:
        pkl.dump(H,f)
    
    return H
    
def mask_xy(xyvals,mask):
    """
    Returns only the xyvals of points that are not blocked by the mask
    """
    maskedEdgeLocations = np.array([[z[1],z[0]] for z in xyvals if mask[z[0],z[1]]])

    return maskedEdgeLocations

def rotate_image(im,angle,center=[],crop=False,size=None):
    """
    Rotate the image about the center of the image or the user specified 
    center. Rotate by the angle in degrees and scale as specified. The new 
    image will be square with a side equal to twice the length from the center
    to the farthest.
    """
    temp = im.shape
    height = temp[0]
    width = temp[1]
    # Provide guess for center if none given (use midpoint of image window)
    if len(center) == 0:        
        center = (width/2.0,height/2.0)
    if not size:
        tempx = max([height-center[1],center[1]])
        tempy = max([width-center[0],center[0]])
        # Calculate dimensions of resulting image to contain entire rotated image
        L = int(2.0*np.sqrt(tempx**2.0 + tempy**2.0))
        midX = L/2.0
        midY = L/2.0
        size = (L,L)
    else:
        midX = size[1]/2.0
        midY = size[0]/2.0
    
    # Calculation translation matrix so image is centered in output image
    dx = midX - center[0]
    dy = midY - center[1]
    M_translate = np.float32([[1,0,dx],[0,1,dy]])
    # Calculate rotation matrix
    M_rotate = cv2.getRotationMatrix2D((midX,midY),angle,1)
    # Translate and rotate image
    im = cv2.warpAffine(im,M_translate,(size[1],size[0]))
    im = cv2.warpAffine(im,M_rotate,(size[1],size[0]),flags=cv2.INTER_LINEAR)
    # Crop image
    if crop:
        (x,y) = np.where(im>0)
        im = im[min(x):max(x),min(y):max(y)]
        
    return im
    
def scale_image(im,scale):
    """
    Scale the image by multiplicative scale factor "scale".
    """
    temp = im.shape
    im = cv2.resize(im,(int(scale*temp[1]),int(scale*temp[0])))
    
    return im

def subtract_images(frame,refFrame):
    """
    Subtracts "prevFrame" from "frame", returning the absolute difference.
    This function is written for 8 bit images.
    """
    if frame.shape != refFrame.shape:
        raise('Error: frames must be the same shape for subtraction')
        
    # Convert frames from uints to ints for subtraction    
    frame = frame.astype(int) # Allow for negative values
    refFrame = refFrame.astype(int) # Allow for negative values
    
    # Take absolute difference and convert to 8 bit
    result = abs(frame - refFrame)
    result = result.astype('uint8')
    
    return result
    
def filter_frame(frame):
    """
    Apply the prescribed filter to the video frame to remove noise.
    """

    denoised = ndimage.gaussian_filter(frame,0.03)
    
    return denoised
    
def fft_image(image):
    """
    Take the fourier transform of an image and return the frequency spectrum
    with the zero mode at the center.
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    
    return fshift
    
def invfft_image(fshift):
    """
    Convert fourier transformed image back to real space.
    """
    f = np.fft.ifftshift(fshift)
    image = np.uint8(np.abs(np.fft.ifft2(f)))
    
    return image
    
def apply_butterworth_filter(image, cutoff, n):
    """
    Take the given image and apply a butterworth filter using Fourier 
    transforms.
    """
    # Transform the image to fourier space
    size = np.shape(image)
    f = fft_image(image)
    # Parse filter parameters
    if (cutoff <= 0) or (cutoff > 0.5):
        print 'cutoff frequency must be between (0 and 0.5]'
        cutoff = 0.5
    n = np.uint8(n)
    # Create the filter
    x,y = np.meshgrid(np.linspace(0,1,size[1]),np.linspace(0,1,size[0]))
    x = x - 0.5
    y = y - 0.5
    radius = np.sqrt(x**2.0 + y**2.0)
    butter = 1.0/(1.0 + (radius/cutoff)**(2.0*n))
    #Apply the filter and transform back
    f = f*butter
    filtered = invfft_image(f)

    return filtered
    
def apply_local_otsu(image,radius=50):
    """
    
    """
    selem = disk(radius)
    newImage = np.uint16(image)
    local_otsu = filters.rank.otsu(newImage,selem)
    newImage = np.uint8(newImage >= local_otsu)
    
    return newImage
    
def process_frame(frame,ref,wettedArea,theta,threshold1,threshold2):
    """    
    Compare the frame of interest to a reference frame and already known 
    wetted area and determine what additional areas have becomed wetted using
    the absolute difference and the difference in edge detection.
    """
    # For testing and optimizing, use these flags to turn off either step
    simpleSubtraction = True
    edgeSubtraction = True
    cutoff = 254
    
    # Initialize storage variables
    image1 = np.zeros_like(frame)
    image2 = np.zeros_like(frame)
    tempRef = np.zeros_like(frame)
    comp1 = np.zeros_like(ref)
    comp2 = np.zeros_like(ref)
    
    # Generate comparison data between reference and image
    if simpleSubtraction:
        image1 = subtract_images(frame,ref)
    if edgeSubtraction:
        tempFrame = np.uint8(filters.prewitt(frame)*255)
        tempRef = np.uint8(filters.prewitt(ref)*255)
        image2 = subtract_images(tempFrame,tempRef)
    
    # Prepare matrices for thresholding the results
    # Apply different thresholds at different intensities
    comp1[:] = threshold1
    comp2[:] = threshold2
#    comp1[ref>=30] = threshold1
#    comp1[ref<30] = 2*threshold1
#    comp2[tempRef<=128] = threshold2
#    comp2[tempRef>128] = 2*threshold2
#    comp2[tempRef<30] = threshold2*.75
        
    # Convert the results to 8 bit images for combining
    image1 = np.uint8((image1 > comp1)*255)
    image2 = np.uint8((image2 > threshold2)*255)
#    wettedArea = np.uint8(wettedArea*255)
    
    # Depending on whether or not the disk is rotating, apply thresholding
    if theta != 0:
        image1 = rotate_image(image1,theta,size=image1.shape)
        image2 = rotate_image(image2,theta,size=image2.shape)
        
    wettedArea = wettedArea + (image1>cutoff) + (image2>cutoff)

    # Fill holes in the wetted area
#    wettedArea = ndimage.morphology.binary_fill_holes(wettedArea)
    
    return wettedArea
    
def get_perimeter(wettedArea):
    """
    Find the perimeter of the wetted area which is identified as the largest
    region of contiguous wetted area.
    """
    contours = cv2.findContours(np.uint8(wettedArea*255),cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_NONE)[0]
    
#    data1 = max(contours,key=len)
    data2 = max(contours,key=cv2.contourArea)
    data = data2.squeeze()
    data = np.reshape(data,(-1,2))
    
    return data
    
def dilate_mask(mask,size=6,iterations=2):
    """
    Increase the size of the masked area by dilating the mask to block 
    additional pixels that surround the existing blacked pixels.
    """
    kernel = np.ones((size,size),np.uint8)
    dilation = cv2.erode(np.uint8(mask),kernel,iterations=iterations)
    mask = (dilation==1)
    
    return mask
    
if __name__ == '__main__':
    pass
#    plt.close('all')
#    filePath = '../Data/Prewetting Study/Water_1000RPM_2000mLmin_2000FPS.avi'
#    homographyFile = 'offCenterTopView_15AUG2015.pkl' # Use None for no transform
#    with open(homographyFile) as f:
#        hMatrix = pkl.load(f)
#    Vid = VF.get_video_object(filePath)
#    image = VF.extract_frame(Vid,20,hMatrix=hMatrix)
#    image1 = image[:,:,0]
##    f = fft_image(image)
#    
##    image2 = image1
#    image2 = apply_local_otsu(image1,50)
#    
#    thresh = ski.filters.threshold_otsu(image1)
#    image3 = image1 >= thresh
#    plt.figure()
#    plt.subplot(1,3,1)
#    plt.imshow(image1,cmap='gray')
#    plt.subplot(1,3,2)
#    plt.imshow(image2,cmap='gray')
#    plt.subplot(1,3,3)
#    plt.imshow(image3,cmap='gray')