#----- IMPORTS -----
import perlin
import matplotlib.pyplot as plt
import pylab

#----- CONSTANTS -----
MAP_SIZE = 512
HEIGHTMAP_PASSES = 4 #number of successive octaves
HEIGHTMAP_BASE_SCALE = 4 #overall scale of the map

map = perlin.octave_map(MAP_SIZE,MAP_SIZE,HEIGHTMAP_PASSES,HEIGHTMAP_BASE_SCALE)
plt.imshow(map)
pylab.show()

def print_maps_to_file():
    """Writes every 1-9 combination of passes and scale to seperate png files"""
    for i in range (1,10):
        for j in range (1,10):
            passes = i
            scale = j
            map = perlin.octave_map(MAP_SIZE,MAP_SIZE,passes,scale)
            plt.rcParams['image.cmap'] = 'terrain'
            plt.imshow(map)        
            title = str(MAP_SIZE)+ "x" + str(MAP_SIZE) + " map with " + str(passes) + " octave passes and a base scale of " + str(scale)
            plt.title(title)
            filename = "./maps/"+str(passes)+"P"+str(scale)+"S"+".pdf"
            plt.savefig(filename)