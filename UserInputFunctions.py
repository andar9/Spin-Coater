# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:52:03 2015

@author: John
"""
import matplotlib.pyplot as plt
import numpy as np
import ctypes
import cPickle as pkl

#Custom modules
import Functions as Fun
import ImageProcessingFunctions as IPF
import VideoFunctions as VF
        

def define_outer_edge(image,shapeType,message=''):
    """
    Displays image for user to outline the edge of a shape. Tracks clicks as
    points on the image. If shapeType is "polygon", the points will be
    connected by lines to show the polygon shape. If the shapeType is "circle",
    the points will be fit to a circle after 4 points have been selected,
    after which point the user can continue to click more points to improve
    the fit.
    
    Possible "shapeType"s:
    'polygon': Returns array of tuples of xy-values of vertices
    'circle': Returns radius and center of circle
    'ellipse': Returns radius1, radius2, center, and angle of rotation
    """
    # Parse input parameters
    if shapeType == 'circle':
        shapeAdj = 'Circular'
        guess = np.shape(image)
        guess = (guess[1]/2,guess[0]/2)
    elif shapeType == 'ellipse':
        shapeAdj = 'Ellipsular'
    elif shapeType == 'polygon':
        shapeAdj = 'Polygonal'
    else: 
        print "Please enter a valid shape type (\"circle\",\"ellipse\"" + \
        "or \"polygon\")."
        return
        
    # Initialize point lists and show image
    x = []; y = []
    figName = 'Define %s Edge - Center click when satisfied' %shapeAdj
    plt.figure(figName)
    plt.rcParams.update({'figure.autolayout': True})
    plt.set_cmap('gray')
    plt.imshow(image)
    plt.axis('image')
    plt.axis('off')
    plt.title(message)
    
    # Get data points until the user closes the figure or center-clicks
    while True:
        pp = get_points(1)
        lims = plt.axis()
        if len(pp) < 1: 
            break
        else:
            pp = pp[0]
        # Reset the plot
        plt.cla()
        Fun.plt_show_image(image)
        plt.title(message)
        plt.axis(lims)
        # Add the new point to the list of points and plot them
        x += [pp[0]]; y += [pp[1]]
        plt.plot(x,y,'r.',alpha=0.5)
        # Perform fitting and drawing of fitted shape
        if shapeType == 'circle':        
            if len(x) > 2:
                xp = np.array(x)
                yp = np.array(y)
                R,center,temp =  Fun.fit_circle(xp,yp,guess)
                guess = center
                X,Y = Fun.generate_circle(R,center)
                plt.plot(X,Y,'y-',alpha=0.5)
                plt.plot(center[0],center[1],'yx',alpha=0.5)
        elif shapeType == 'ellipse':
            if len(x) > 3:
                xp = np.array(x)
                yp = np.array(y)
                R1,R2,center,theta =  Fun.fit_ellipse(xp,yp)
                X,Y = Fun.generate_ellipse(R1,R2,center,theta)
                plt.plot(X,Y,'y-',alpha=0.5)
                plt.plot(center[0],center[1],'yx',alpha=0.5)
        elif shapeType == 'polygon':
            plt.plot(x,y,'y-',alpha=0.5) 
                    
    plt.close()
    if shapeType == "circle":
        return R,center
    elif shapeType == 'ellipse':
        return R1,R2,center,theta
    elif shapeType == "polygon":
        xyVals = [(x[i],y[i]) for i in range(len(x))]
        return xyVals

def get_mask_data(maskFile,vid,hMatrix=None,check=False):
    """
    Shows user masks overlayed on given image and asks through a dialog box
    if they are acceptable. Returns True for 'yes' and False for 'no'.
    """
    
    # Parse input parameters
    image = VF.extract_frame(vid,1,hMatrix=hMatrix)
    try:
        with open(maskFile) as f:
            maskData = pkl.load(f)
    except:
        print 'Mask file not found, please create it now.'
        maskData = IPF.create_mask_data(image,maskFile)
        
    while check:
        plt.figure('Evaluate accuracy of predrawn masks for your video')
        maskedImage = IPF.mask_image(image,maskData['mask'])
        temp = np.dstack((maskedImage,image,image))
        plt.imshow(temp)
        center = maskData['diskCenter']
        plt.plot(center[0],center[1],'bx')
        plt.axis('image')
        
        response = ctypes.windll.user32.MessageBoxA(0, 'Do you wish to keep' + \
                            ' the current mask?','User Input Required', 4)
        plt.close()
        if response == 6: # 6 means yes
            return maskData

        else: # 7 means no
            print 'Existing mask rejected, please create new one now.'
            maskData = IPF.create_mask_data(image,maskFile)
            
    return maskData
    
def get_points(Npoints=1,im=None):
    """ Alter the built in ginput function in matplotlib.pyplot for custom use.
    This version switches the function of the left and right mouse buttons so
    that the user can pan/zoom without adding points. NOTE: the left mouse 
    button still removes existing points.
    INPUT:
        Npoints = int - number of points to get from user clicks.
    OUTPUT:
        pp = list of tuples of (x,y) coordinates on image
    """
    if im is not None:
        plt.imshow(im)
        plt.axis('image')
        
    pp = plt.ginput(n=Npoints,mouse_add=3, mouse_pop=1, mouse_stop=2,
                    timeout=0)
    return pp            
    
def get_homography_matrix(fileName,vid):
    """
    Load the homography matrix from file or create it if it does not exist.
    """
    
    # Handle homography data file
    if fileName is not None:
        try:
            with open(fileName,'rb') as f:
                hMatrix = pkl.load(f)
        except:
            image = VF.extract_frame(vid,0)
            hMatrix = IPF.define_homography_matrix(image,fileName)
    else:
        hMatrix = None
        
    return hMatrix
    
#def get_mask_data(fileName,vid,hMatrix,check=False):
#    """
#    Load the mask data from file or create if it does not exist.
#    """    
#        
#    try:
#        with open(fileName,'rb') as f:
#            maskData = pkl.load(f)
#    except:
#        check = True
#    if check:
#        maskData = check_mask(fileName,vid,hMatrix)
#    
#    return maskData
    
def get_intensity_region(fileName,vid):
    """
    Identify a region of the video frames to use for monitoring light 
    intensity.
    """
    try:
        with open(fileName,'rb') as f:
            mask = pkl.load(f)
    except:
        image = VF.extract_frame(vid,0)
        plt.gray()
        plt.close()
        points = define_outer_edge(image,'polygon','Define a region for \n' + 
                                    'monitoring light intensity.')
        mask,points = IPF.create_polygon_mask(image,points)
        with open(fileName,'wb') as f:
            pkl.dump(mask,f)
        
    return mask
    
def ok_to_proceed():
    """
    Present a dialog to the user to get permission to continue script.
    """
    response = ctypes.windll.user32.MessageBoxA(0, 'Ready to proceed?',
                                                'User Input Required', 4)
    return response == 6

if __name__ == '__main__':
    print 'here'
    ok_to_proceed()
    print 'done'
    pass
