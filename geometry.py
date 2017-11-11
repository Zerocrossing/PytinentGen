import numpy as np
def normalize(array):
    """Normalizes an array to contain floats from [0,1]"""
    array -= array.min()
    array /= array.max()
    return array

def circle(size,r):
    """Returns an array of floats with an inscribed circle of size r where the center is 1.0 and edges are 0"""
    center = size//2
    maxdist = np.sqrt(2*(center**2))
    x,y = np.indices((size,size))
    array = (1-np.abs(np.hypot(x-center,y-center)/r))
    return array.clip(min=0)