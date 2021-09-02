# Moonsoon - a fantasy map generator powered by numpy
# modified by HHY 2021.9.2 16:21:36

import os
os.environ["OMP_NUM_THREADS"]='6'
import noise
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import to_rgba
from scipy.signal import convolve2d
from math import exp
from tqdm import tqdm
import matplotlib.animation as animation
import heapq
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import Image
from scipy.spatial import Voronoi, voronoi_plot_2d
import skimage.transform
from numba import jit,njit
from numba.typed import Dict as numba_dict

# help(noise)
# http://libnoise.sourceforge.net/glossary/

def sample2D(a,i,j):
    
    # 将坐标i,j裁剪到符合地图尺寸
    i, j = np.clip(i,0,a.shape[0]-1), np.clip(j,0,a.shape[1]-1)
    # floor向下取整, ceil向上取整
    i0, i1, j0, j1 = int(np.floor(i)), int(np.ceil(i)), int(np.floor(j)), int(np.ceil(j)) 
    
    tmp = a[i0,j0]*(i1-i)*(j1-j) \
        +a[i1,j0]*(i-i0)*(j1-j) \
        +a[i0,j1]*(i1-i)*(j-j0) \
        +a[i1,j1]*(i-i0)*(j-j0)
    return tmp

def clip(value, lower, upper):
    return lower if value < lower else upper if value > upper else value

#seeds and parameters
seed = 77
townseed = 7
histseed = 77
size = 400
zoom = 30
zoom2 = 4
travel_zoom = 0.7
villagedense = 50
towndense = 20
castledense = 9
x0,y0 = 2,0
temperature_pos,temperature_zoom = 2.7,2
distortion = 2
distortion2 = 0.5

# 第一步，使用柏林噪声生成大陆主体
gammas = np.zeros((size,size))
xcoord,ycoord=np.mgrid[0:size,0:size]/zoom
xcoord += x0
ycoord += y0
for i in range(size):
    for j in range(size):
        x,y = i/zoom+x0, j/zoom+y0
        dx = distortion2 * noise.pnoise2(x/zoom2,y/zoom2,octaves=2,persistence=0.5,lacunarity=2.0,base=seed+15)
        dy = distortion2 * noise.pnoise2(x/zoom2,y/zoom2,octaves=2,persistence=0.5,lacunarity=2.0,base=seed+23)
        r = np.sqrt((x-0.5*size/zoom)**2 + 0.5*(y-0.5*size/zoom)**2)/zoom2 # do not distort here
        gamma = noise.pnoise2(x/zoom2+dx, y/zoom2+dy, octaves=2, persistence=0.7, lacunarity=2, base=seed+66)*2
        gamma = np.tanh((1-r)*5)+gamma*2-0.1
        gammas[i,j] = np.tanh(gamma)

'''
plt.figure(figsize=(8,8))
plt.imshow(gammas,cmap='viridis')
plt.colorbar()
# plt.imshow(np.ma.masked_where(gammas>0,np.ones(gammas.shape)),cmap='summer')
plt.contour(gammas,[0],colors='black')
plt.show()
'''
print(gammas)
