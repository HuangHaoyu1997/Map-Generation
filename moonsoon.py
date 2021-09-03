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
seed = 7
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

'''
plt.figure(figsize=(8,8))
plt.imshow(gammas,cmap='viridis')
plt.colorbar()
# plt.imshow(np.ma.masked_where(gammas>0,np.ones(gammas.shape)),cmap='summer')
plt.contour(gammas,[0],colors='black')
plt.show()
'''

# 使用另一层柏林噪声生成山脉。通过使用单独的噪音层，可以更好地控制大陆上有多少座山
# 为了使地形更有趣，还可以通过另一个柏林噪声扰动坐标以模拟地质构造运动
# 因此可以生成一些弯曲的山脊

# 生成海拔信息，get elevation
elevation = np.zeros((size,size))
for i in range(size):
    for j in range(size):
        x,y = i/zoom+x0,j/zoom+y0
        dx = distortion * noise.pnoise2(x,y,octaves=4,persistence=0.5,lacunarity=2.0,base=seed+15)
        dy = distortion * noise.pnoise2(x,y,octaves=4,persistence=0.5,lacunarity=2.0,base=seed+23)
        value = noise.pnoise2(x+dx,y+dy,octaves=6,persistence=0.5,lacunarity=2.0,base=seed+0)*2
        value = (value + gammas[i,j])/2
        elevation[i,j] = value
'''
plt.figure(figsize=(10,10))
plt.imshow(gammas,cmap='Blues')
plt.colorbar()
plt.imshow(np.ma.masked_where(elevation<0,elevation),cmap='summer') # 显示海拔低于0的区域
plt.colorbar()
plt.contour(np.clip(elevation,0,1),colors='black',alpha=0.1)
plt.contour(elevation,[0.5],colors='black',alpha=0.5)
plt.contour(elevation,[0],colors='black')
plt.title("Perlin Noise")
plt.show()
'''

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
'''
plt.figure(figsize=(12,12))
plt.imshow(gammas,cmap='Blues')
plt.imshow(np.ma.masked_where(elevation<0,elevation),cmap='summer')
plt.contour(np.clip(elevation,0,1),colors='black',alpha=0.1)
plt.contour(elevation,[0.5],colors='black',alpha=0.5)
plt.contour(elevation,[0],colors='black')
plt.title('Landmass')
plt.show()
'''


# 根据海拔和坐标设定气温
temperature = np.cos((xcoord/zoom2-temperature_pos)/temperature_zoom)**2*0.8 + 0.1
temperature *= (1.1 - np.clip(1.2*elevation,0.1,1))

'''

plt.figure(figsize=(10,10))
plt.imshow(temperature,cmap='jet',vmin=0,vmax=1,alpha=.5)
plt.colorbar()

plt.contour(np.clip(elevation,0,1),colors='black',alpha=0.1)
plt.contour(elevation,[0.5],colors='black',alpha=0.5)
plt.contour(elevation,[0],colors='black')

plt.contour(temperature,[.2,.3,.5,.8],colors=['blue','blue','green','red'])
plt.title('Temperature')
plt.show()
'''
#####################################
# 第2步，生成水资源。雨水从海洋中汲取水分，塑造陆地，形成高原、湖泊和河流
# 模拟降水过程
# 我们使用[Dijkstra 算法](https://en.wikipedia.org/wiki/Breadth-first_search)
# （这是一个具有优先级队列的广度优先搜索算法）评估陆地上每个点到海洋的距离
# 水汽传播的成本被海拔和坡度穿透，因此可以为内陆、山区环绕的区域实现一个漂亮的[雨影](https://en.wikipedia.org/wiki/Rain_shadow)。
# 此外，较冷的海洋产生的水蒸气较少，较冷地区的水汽传播成本较小
# 同时也使水蒸气更有可能向东移动，以产生一些各向异性。
# get rainshadow
_inf=float('inf')

@njit 
def _npclip(a,b,c):
    '''
    numba加速的np.clip操作
    将a裁剪到[b,c]或[c,b]
    '''
    return np.minimum(np.maximum(a,b),c)

@njit
def get_rainshadow():
    dist = np.zeros((size,size))
    dist.fill(_inf)
    dirs = ((0,1),(1,0),(0,-1),(-1,0))
    opened = []
    for i in range(size):
        for j in range(size):
            if elevation[i,j] <= 0:
                dist_penalty = _npclip(1-temperature[i,j],0,1)*zoom*0.8
                opened.append((dist_penalty,i,j,0))
    heapq.heapify(opened) # 将list opened转化成堆队列
    while len(opened)>0:
        d,i,j,o = heapq.heappop(opened)
        if dist[i,j] > d:
            dist[i,j] = d
            for oo,delta in enumerate(dirs):
                ii,jj = i+delta[0],j+delta[1]
                if 0<=ii and ii<size and 0<=jj and jj<size:
                    if elevation[ii,jj]>0:
                        cost=0.7*elevation[ii,jj]+0.5*(1-delta[1])+0.5*temperature[ii,jj]
                        if elevation[ii,jj]>0.5:
                            cost += 2
                        if dist[ii,jj]>d+cost:
                            heapq.heappush(opened,(d+cost,ii,jj,oo))
    return dist.copy()/zoom
rainshadow = get_rainshadow()

'''
plt.figure(figsize=(10,10))
plt.imshow(rainshadow,cmap='viridis')
plt.colorbar()

plt.contour(np.clip(elevation,0,1),colors='black',alpha=0.1)
plt.contour(elevation,[0.5],colors='black',alpha=0.5)
plt.contour(elevation,[0],colors='black')
# plt.imshow(np.ma.masked_where(elevation>0,np.zeros(elevation.shape)),cmap='winter')
plt.title('Rain Shadow')
plt.show()
'''

# 降水量precipitation与水汽转移成本呈负指数关系，用黄色和红色绘制了降水量低于0.3和0.1的区域
rain = np.exp(-rainshadow/0.7)
'''
plt.figure(figsize=(10,10))
plt.imshow(rain,cmap='viridis')
plt.colorbar()
plt.contour(rain,[0.1,0.3,0.7],colors=['red','yellow','blue'])

plt.contour(np.clip(elevation,0,1),colors='black',alpha=0.1)
plt.contour(elevation,[0.5],colors='black',alpha=0.5)
plt.contour(elevation,[0],colors='black')

plt.imshow(np.ma.masked_where(elevation>0,np.ones(elevation.shape)),cmap='winter')
plt.title('Precipitation')
plt.show()
'''


# 找到要填补的山谷
# 用降雨和沉积物填满山谷，使它们要么成为高原，要么成为湖泊。
# 然后土地更平坦，海拔高于海平面的地方会有更多的平坦空间。
# 山谷被定义为被群山环绕的地区。更正式地，山谷被定义为这样一个最大区域，该区域中的每个pixel周围的pixel的海拔都比该区域的pixel高
# 这意味着必须有一个[saddle](https://en.wikipedia.org/wiki/Saddle_(landform)) 或山口。
# 水或沉积物会不断淹没整个山谷，直到到达最低的鞍座并流出。
# 填满整个山谷后，可以从陆地上的任何一点找到一条*非上升*的通往海洋的路径。整个过程可以O(NxN)复杂度完成，（NxN 是地图大小）。 
# [这篇文章](https://medium.com/universe-factory/how-i-generated-artificial-rivers-on-imaginary-continents-747df12c5d4c) 中讨论了该算法。
# 基本思想是使用修改后的 Dijkstra 算法（或 BFS）一点一点地提高海平面以找到鞍座，并在新鞍座内填充山谷：我为所有与海相邻的像素保留一个优先级队列，按海拔排序。
# 然后我将“海平面”升高到最低的鞍点，并检查它是否打开了一些山谷以进行洪水填充。
# 所以我扩展了“海”并在队列中添加了更多像素。我迭代这个过程，直到所有像素都被淹没。

elevation_tmp = elevation.copy()
# get lakes and waterflow
# https://medium.com/universe-factory/how-i-generated-artificial-rivers-on-imaginary-continents-747df12c5d4c
@njit
def calculate_water():
    elevation = elevation_tmp.copy()
    fill = _npclip(elevation,0,1)
    orig = np.zeros((size,size),dtype=np.int8)
    drained = elevation<-1
    dirs = ((0,1),(1,0),(0,-1),(-1,0))
    opened = []
    for i in range(size):
        for j in range(size):
            if elevation[i,j] < -0.01:
                opened.append((0.0,i,j,0))
    heapq.heapify(opened)
    #this bfs runs unexpectionally slow. may debug
    np.random.seed(seed + 2564)
    while len(opened)>0:
        f,i,j,o = heapq.heappop(opened)
        if not drained[i,j]:
            drained[i,j] = True
            fill[i,j] = f
            orig[i,j] = (o+2)%4
            for oo,delta in enumerate(dirs):
                ii,jj = i+delta[0],j+delta[1]
                if 0<=ii and ii<size and 0<=jj and jj<size:
                    if not drained[ii,jj]:
                        ff = max(f,_npclip(elevation[ii,jj],0,1))+np.random.random()*0.001
                        #random breaks priority but not break algorithm, why?
                        #ff=max(f,clip(elevation[ii,jj],0,1))+random[ii,jj]*0.01+random[i,j]*0.005
                        heapq.heappush(opened,(ff,ii,jj,oo))
    fill -= _npclip(elevation,0,1)
    return fill,orig
fill,downstream=calculate_water()

plt.figure(figsize=(10,10))
plt.imshow(elevation>0,alpha=0.5)
plt.imshow(fill,cmap='viridis')
plt.colorbar()

plt.contour(np.clip(elevation,0,1),colors='black',alpha=0.1)
plt.contour(elevation,[0.5],colors='black',alpha=0.5)
plt.contour(elevation,[0],colors='black')
plt.imshow(np.ma.masked_where(elevation>0,np.ones(elevation.shape)),cmap='winter')
plt.title('Valley to fill')
plt.show()
