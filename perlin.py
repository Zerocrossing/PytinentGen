import numpy as np
#Evil Globals
permutation = None
gradients = None

#----- PUBLIC PERLIN MAP FUNCTIONS -----
def perlin_map(x_size,y_size,scale,as_list=False,normalize = False):
    """Returns a perlin map of size x * y. Scale will adjust fineness of noise"""
    linx = np.linspace(0,scale,x_size)
    liny = np.linspace(0,scale,y_size)
    x,y = np.meshgrid(linx,liny)
    map = __perlin(x,y)
    if normalize:
        map -= map.min()
        map /= map.max()
    if as_list:
        return map.tolist()
    return map

def octave_map(x_size,y_size,num_octaves,base_scale,as_list=False,normalize = False):
    """Returns a map of the sum of num_octaves perlin maps, each scaled down in amplitude by 1/n"""
    output = np.zeros((x_size,y_size))
    for n in range(1,num_octaves+1):
        output += perlin_map(x_size,y_size,base_scale*n)*(1/n)
    if normalize:
        output -= output.min()
        output /= output.max()
    return output

#----- PERMUTATION AND GRADIENT FUNCTIONS -----
#Size shouldn't need to change under normal use
def __make_permutation(size=256):
    """Generates a random permutation of the numbers 0-256 and doubles the array for overflow cases"""
    global permutation
    permutation = np.arange(size,dtype=int)
    np.random.shuffle(permutation)
    permutation = np.append(permutation,permutation)

def __make_gradients(size=256):
    global gradients
    x = np.array(2*np.random.rand(size,2)-1)
    gradients = x/np.linalg.norm(x)
    return gradients

#----- PRIVATE HELPER FUNCTIONS -----
def __perlin(x,y):
    """An implementation of Perlin Noise, uses some updates from the 2002 paper but precomputes random gradients"""
    __make_permutation()
    __make_gradients()
    p = permutation
    X = x.astype(int)
    Y = y.astype(int)
    x = x - X
    y = y - Y
    u = __fade(x)
    v = __fade(y)
    #Hash coordinates
    BL = p[p[X]+Y]
    TL = p[p[X]+Y+1]
    TR = p[p[X+1]+Y+1]
    BR = p[p[X+1]+Y]
    #Gradient Coordinates
    BL = __grad(BL,x,y)
    TL = __grad(TL,x,y-1)
    TR = __grad(TR,x-1,y-1)
    BR = __grad(BR,x-1,y)
    #Interpolate
    x1 = __lerp(u,BL,BR)
    x2 = __lerp(u,TL,TR)
    return (__lerp(v,x1,x2))

def __fade(t):
    """Perlin's smoothing function"""
    return (6 * t**5) - (15 * t**4) + (10 * t**3)

def __lerp(t,a,b):
    """linear interpolation"""
    return a + t * (b - a)

def __grad(hash,x,y):
    "selects a pseudorandom gradient vector based on the hash, and returns the dot of the vector and (x,y)"
    g = gradients[hash]
    return g[:,:,0] * x + g[:,:,1] * y