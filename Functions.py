# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:52:03 2015

@author: John
"""
import matplotlib.pyplot as plt
import numpy as np
from math import atan2
from scipy import optimize
import cv2
from scipy.interpolate import interp1d

# Custom modules


def fit_circle(x,y,center_estimate=(0,0)):
    """
    Fit the x and y points to a circle. Returns the circle's radius, center,
    and residue (a measure of error)
    """
    def calc_R(center):
        """
        Calculate the distance of each 2D point from the center (xc, yc) 
        """
        xc = center[0]
        yc = center[1]
        return np.sqrt((x-xc)**2 + (y-yc)**2)
    
    def f_2(center):
        """
        Calculate the algebraic distance between the data points and the mean
        circle centered at (xc, yc)
        """
        Ri = calc_R(center)
        return Ri - Ri.mean()

    center, ier = optimize.leastsq(f_2,center_estimate)
    
    Ri = calc_R(center)
    R = np.mean(Ri)
    residue   = sum((Ri - R)**2)
    return R, center, residue
    
def generate_circle(R,center,N=100,t0=0.0,t1=2.0*np.pi):
    """
    Generate an array of x and y values that lie evenly spaced on a circle 
    with the specified center and radius.
    """
    theta = np.linspace(t0,t0+t1,N)
    y = R*np.sin(theta) + center[1]
    x = R*np.cos(theta) + center[0]
    return x,y
    
def fit_ellipse(x,y):
    """
    Fit the x and y points to an ellipse. Returns the radii, center,
    and and angle of rotation. Taken directly from:
        http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
    
    def fit(x,y):
        x = x[:,np.newaxis]
        y = y[:,np.newaxis]
        D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T,D)
        C = np.zeros([6,6])
        C[0,2] = C[2,0] = 2; C[1,1] = -1
        E, V =  np.linalg.eig(np.dot(np.linalg.inv(S), C))
        n = np.argmax(np.abs(E))
        a = V[:,n]
        return a
        
    def ellipse_center(a):
        b,c,d,f,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[0]
        num = b*b-a*c
        x0=(c*d-b*f)/num
        y0=(a*f-b*d)/num
        return np.array([x0,y0])
    
    def ellipse_angle_of_rotation(a):
        b,c,a = a[1]/2, a[2], a[0]
        return 0.5*np.arctan(2*b/(a-c))
    
    def ellipse_axis_length(a):
        b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
        down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        res1=np.sqrt(up/down1)
        res2=np.sqrt(up/down2)
        return np.array([res1, res2])
    
    a = fit(x,y)
    center = ellipse_center(a)
    theta = ellipse_angle_of_rotation(a)
    [R1,R2] = ellipse_axis_length(a)

    return R1, R2, center, theta
    
def generate_ellipse(R1,R2,center,theta,N=100):
    """
    Generate an array of x and y values that lie on an ellipse with the 
    specified center, radii, and angle of rotation (theta)
    """
    t = np.linspace(0.0,2.0*np.pi,N)
    x = R1*np.cos(t)*np.cos(theta) - R2*np.sin(t)*np.sin(theta) + center[0]
    y = R1*np.cos(t)*np.sin(theta) + R2*np.sin(t)*np.cos(theta) + center[1]
    return x,y
    
def plot_circles(R,center,N=15):
    """
    Plot a set of concentric circles on an existing plot where the outermost
    circle is defined by the specified radius and center.
    """
    Ri = np.linspace(0,R,N+1)[1:]
    color = ['r--','b--','m--','y--']
    for i in range(N):
        xi, yi = generate_circle(Ri[i],center)
        j = i % 4
        plt.plot(xi,yi,color[j],alpha=0.3)
        
def plt_show_image(image):
    """
    This removes tick marks and numbers from the axes of the image and fills 
    up the figure window so the image is easier to see.
    """
    plt.imshow(image)
    plt.axis('off')
    plt.axis('image')
    plt.tight_layout(pad=0)
    
def generate_polygon(x,y,N):
    """
    Generate an array of x and y values that lie evenly spaced along a polygon
    defined by the x and y values where it is assumed that the first value
    is also the last value to close the polygon
    """
    # Add the first point to the end of the list and convert to array if needed
    if type(x) == list:
        x = np.array(x + [x[0]])
        y = np.array(y + [y[0]])
    else:
        x = np.append(x,x[0])
        y = np.append(y,y[0])
        
    # Parameterize the arrays and interpolate
    d = [get_distance((x[i],y[i]),(x[i+1],y[i+1])) for i in range(len(x)-1)]
    d = np.cumsum([0]+d)
    t = np.linspace(0,d[-1],N)
    fx = interp1d(d,x)
    fy = interp1d(d,y)
    x = fx(t)
    y = fy(t)
    
    return x,y
    
#def polygon_perimeter(pts):
#    """
#    Calculates the perimeter of the polygon defined by the tuples in the array
#    pts
#    """
#    # Calculate length of edge from last point to first point
#    perimeter = 0
#    d = get_distance(pts[len(pts)-1],pts[0])
#    perimeter += d
#    # Calculate length of remaining edges
#    for i in range(0,len(pts)-1):
#        d = get_distance(pts[i],pts[i+1])
#        perimeter += d
#    return perimeter
    
def generate_periodic_circle(R, var, center, freq, fn):
    """
    Generates points that map out a circle with a given periodic function
    overlayed.
    
    fn can be the following:
    Fun.sine -> overlays sine wave with amplitude "var"
    Fun.sawtooth -> overlays the increasing sawtooth that begins at (R-var) and
    increases up to (R+var) in each period
    Fun.triangle -> overlays the increasing-decreasing triangle, which begins at
    (R-var), reaches (R+var) halfway through the period, and then decreases
    back to (R-var) by the end of the period.
    """
    xCenter = center[0]
    yCenter = center[1]
    # The perimeter for a triangle wave is about sqrt((circum)^2+(2*var*freq)^2),
    # A little bit is added to get the formula used below, just in case.
    nPts = int(2*np.sqrt((2*np.pi*R)**2.0 + (2*var*freq)**2.0)) # ensure sufficient points to fill perimeter
    theta = np.linspace(0,2*np.pi, nPts)
    # Initalize xy, array of tuples of x- and y-values
    xy = np.array([[0,0]])
    if freq == 0:
        period = 0
    else:
        period = 2*np.pi/freq
    for i in range(len(theta)):
        pert = fn(theta[i], period, var)
        r = R + pert
        # Convert from polar to Cartesian coordinates
        x = r*np.cos(theta[i]) + xCenter
        y = r*np.sin(theta[i]) + yCenter
        xy = np.concatenate((xy,np.array([[x,y]])))
    return xy[1:len(xy)]
    
def sawtooth(x, period, var):
    """
    Calculates y-value for an increasing sawtooth with amplitude "var" above
    and below y = 0, i.e. it begins at (0,-var), passes through (period/2,0),
    and ends at (period,+var) in each period. 
    """
    if period == 0:
        return x
    else:
        x = x % period
        frac = x/period
        y = (2*frac-1)*var
        return y
    
def sine(x, period, var):
    """
    Calculates y-value for a sine wave starting at the origin.
    """
    return var*np.sin((2*np.pi/period)*x)
    
def triangle(x, period, var):
    """
    Calculates y-value for an increasing-decreasing triangle function with 
    amplitude "var" above and below y = 0, i.e. it begins at (0,-var), 
    increases to (period/2,+var), and decreases to (period,-var).
    """
    if period == 0:
        return x
    else:
        x = x % period
        frac = 2*abs(0.5 - (x/period))
        y = (1-2*frac)*var
        return y
        
def get_angle(pt1,pt2,pt3):
    """
    Uses law of cosines to calculate angle defined by given points, where pt2
    is the vertex of the angle
    """
    a = float(get_distance(pt1,pt2))
    b = float(get_distance(pt2,pt3))
    c = float(get_distance(pt1,pt3))
    angle = np.arccos((a**2 + b**2 - c**2)/(2*a*b)) # Law of Cosines 
    
    return angle
    
def get_corrected_arclength(pts,closed=False):
    """
    Smooths the digital curve defined by the row-column tuples in the numpy
    array "pts" using a 5-point average, i.e., it replaces each tuple with an 
    average of the two previous points, the two succeeding points, and the 
    point itself. The arc-length is then calculated by scaling up the image
    to 3-decimal-place precision, applying the OpenCV arcLength function, and
    scaling back down.
    """
    
    l = len(pts)
    ptsDown2 = np.concatenate((pts[2:l],pts[0:2]))
    ptsDown1 = np.concatenate((pts[1:l],np.array([(pts[0][0],pts[0][1])])))
    ptsUp1 = np.concatenate((np.array([(pts[l-1][0],pts[l-1][1])]),pts[0:l-1]))
    ptsUp2 = np.concatenate((pts[l-2:l],pts[0:l-2]))
    summedPts = ptsDown2 + ptsDown1 + pts + pts + ptsUp1 + ptsUp2
    avePts = summedPts/5.0
    zoomAvePts = np.round(avePts)
    arcLength = cv2.arcLength(zoomAvePts.astype(int),closed)
    
    return arcLength
    
def get_distance(pt1,pt2):
    """
    Calculates distance between two points given as tuples of row and col
    """
    x1 = pt1[1]
    y1 = pt1[0]
    x2 = pt2[1]
    y2 = pt2[0]
    d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d

def get_radial_distance(p1,p2,R):
    """ Get the distance between 2 points, but call the distance zero if it is
    less than a tenth of the radius (area where flow is blocked by jet arm head).
    INPUT:
        p1 = tuple of (x,y) coordinates
        p1 = tuple of (x,y) coordinates
    OUTPUT:
        d = distance between points
    """
    d = np.sqrt((p1[0]-p2[0])**2.0 + (p1[1]-p2[1])**2.0)
    if d < 0.1*R:
        d = 0.0
    elif d > R:
        d = R
    return d
        
def get_theta(p1,p2):
    """
    Compute the angle of the point on the disk relative to the center where
    p2 is the center of the disk.
    """
    
    dy = p1[1] - p2[1]
    dx = p1[0] - p2[0]
    theta = atan2(dy,dx)
    return theta    
    
def get_linear_distance(p1,p2):
    """ Get the distance between 2 points, but call the distance zero if the 
    first point is to the left of the second point.
    INPUT:
        p1 = tuple of (x,y) coordinates
        p1 = tuple of (x,y) coordinates
    OUTPUT:
        d = distance between points
    """
    if p1[0] < p2[0]:
        d = 0
    else:
        d = np.sqrt((p1[0]-p2[0])**2.0 + (p1[1]-p2[1])**2.0)
    return d
    
def convert_flowrate(setting):
    """
    Return the actual flow rate using the flow rate setting and the conversion
    from the pump calibration work.
    The setting and the returned flow rate are in units of mL/min.
    """
    return 0.8991*setting - 62.339
    
def rotate_points(x,y,theta,center=[0,0],units='radians'):
    """
    Take a set of points and rotate them and angle theta about a center point.
    Theta in degrees.
    """
    # Convert theta to radians
    if units == 'degrees':
        theta = theta/180.0*np.pi
    elif units == 'radians':
        pass
    else:
        print 'Invalid input parameter for angle units! Assuming radians'
    
    # Compute rotation matrix constants
    s = np.sin(theta)
    c = np.cos(theta)

    # translate points so that the center is at the to origin:
    x -= center[0]
    y -= center[1]

    # rotate points by multiplying by rotation matrix
    xnew = x*c - y*s
    ynew = x*s + y*c

    # translate points back to original location
    xnew += center[0]
    ynew += center[1]
  
    return xnew,ynew
    
if __name__ == '__main__':
    plt.close('all')
    R = 1
    center = [0,0]
    x,y = generate_circle(R,center,7,t0=0,t1=np.pi/2)
    x1,y1 = rotate_points(x,y,-90,center,'degrees')
    plt.plot(x,y,'ro',x1,y1,'b.')
    plt.axis('equal')