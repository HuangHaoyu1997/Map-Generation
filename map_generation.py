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
import heapq # 堆队列
# 堆是一个二叉树，它的每个父节点的值都只会小于或等于所有孩子节点（的值）。 
# 它使用了数组来实现：从零开始计数，对于所有的k，都有 heap[k] <= heap[2*k+1] 和 heap[k] <= heap[2*k+2]。
# 为了便于比较，不存在的元素被认为是无限大。 堆最有趣的特性在于最小的元素总是在根结点：heap[0]。
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

###############################
# 第一步，使用柏林噪声生成大陆主体
###############################
gammas = np.zeros((size,size))
xcoord,ycoord = np.mgrid[0:size,0:size]/zoom # np.mgrid[0:size,0:size].shape=(2,size,size)
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


# 使用另一层柏林噪声生成山脉。通过使用单独的噪音层，可以更好地控制大陆上有多少座山
# 为了使地形更有趣，还可以通过另一个柏林噪声扰动坐标以模拟地质构造运动
# 因此可以生成一些弯曲的山脊

# 生成海拔数据
elevation = np.zeros((size,size))
for i in range(size):
    for j in range(size):
        x,y = i/zoom+x0,j/zoom+y0
        dx = distortion * noise.pnoise2(x,y,octaves=4,persistence=0.5,lacunarity=2.0,base=seed+15)
        dy = distortion * noise.pnoise2(x,y,octaves=4,persistence=0.5,lacunarity=2.0,base=seed+23)
        value = noise.pnoise2(x+dx,y+dy,octaves=6,persistence=0.5,lacunarity=2.0,base=seed+0)*2
        value = (value + gammas[i,j])/2
        elevation[i,j] = value




# 为了使大陆更加连续，通过泛洪算法[flood fill](https://en.wikipedia.org/wiki/Flood_fill) 
# 将小的内陆海洋的海拔调整到略高于海平面的高度，这样就得到一些像样的平原

@njit
def fill_ocean(elevation,minsize,threshold=0,target=0.05):
    '''
    elevation：海拔地图
    minsize：小于该面积的ocean将被填充
    threshold：小于该值的区域将被替换成target
    target：目标值
    '''
    visited = elevation < -1 # 海拔低于-1的区域不考虑替换
    dirs = ((0,1),(1,0),(0,-1),(-1,0))
    for i0 in range(size):
        for j0 in range(size):
            if elevation[i0,j0] <= threshold and not visited[i0,j0]:
                opened = []
                closed = []
                count = 0
                opened.append((i0,j0))
                visited[i0,j0] = True
                count += 1
                while len(opened)>0:
                    i,j = opened.pop()
                    closed.append((i,j))
                    for oo,delta in enumerate(dirs):
                        ii,jj = i+delta[0],j+delta[1]
                        if 0<=ii and ii<size and 0<=jj and jj<size: # 得到当前点的上下左右四个邻点，且判断不超过地图范围
                            if not visited[ii,jj] and elevation[ii,jj]<=threshold:
                                opened.append((ii,jj))
                                visited[ii,jj]=True
                                count+=1
                        else:
                            count += 2*zoom
                if count<minsize: # 总面积小于minsize，则认定为"small ocean"，需要被填充
                    for p in closed:
                        elevation[p[0],p[1]]=target
    return elevation
# 移除面积过小的内湖
elevation = fill_ocean(elevation,2*zoom**2,-0.1,0.1)
elevation = fill_ocean(elevation,2*zoom**2,0.01,0.05)
# 移除小岛，替换为海域
elevation = -fill_ocean(-elevation,5,0,0.05)

plt.figure(figsize=(12,12))
plt.imshow(gammas,cmap='Blues')
plt.imshow(np.ma.masked_where(elevation<0,elevation),cmap='summer')
plt.contour(np.clip(elevation,0,1),colors='black',alpha=0.1)
plt.contour(elevation,[0.5],colors='black',alpha=0.5)
plt.contour(elevation,[0],colors='black')
plt.title('Landmass')
plt.show()
