
from data_grid import DataGrid


from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np
import math



#folder with data files

regex_500 = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D.csv"""
regex_600 = """TiNiSn_600C_Y20190219_14x14_t65_(?P<num>.*?)_bkgdSub_1D.csv"""

#CHANGE THIS
path ="/path/to/data/here/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"

dataGrid = DataGrid(path,regex_500)


#cosine similarity function using two grid positions
#based on data peaks
def similarity(d1,d2):
    a = dataGrid.data[d1][:,1]
    b = dataGrid.data[d2][:,1]
    pa, _ = find_peaks(a)
    pb, _ = find_peaks(b)
    p = np.append(pa,pb,axis=0)
    return np.dot(a[p],b[p])/np.linalg.norm(a[p])/np.linalg.norm(b[p])


#create grid
grid = np.zeros(shape=(15,15))


#_____________________________________
# CALCULATE SIMILARITY IN GRID


for val in range(1,178):
    x,y = dataGrid.coord(val)
    #keys = ['up','left']
    keys = ['up', 'left', 'right', 'down']
    neigh = [dataGrid.neighbors(val)[k] for k in dataGrid.neighbors(val).keys() if k in keys]
    sim_values = [similarity(val,x) for x in neigh]
    if len(sim_values) == 0:
        grid[y-1][15-x] = 1
        continue
    grid[y-1][15-x] = np.min(sim_values)



#_____________________________________


# for plotting

def getPointsX(x,list):
    if len(list) == 0:
        return []
    dir = list.pop(0)
    if dir in ['u','d','s']:
        return [x] + getPointsX(x,list)
    elif dir == 'l':
        return [x-1] + getPointsX(x-1,list)
    else:
        return [x+1] + getPointsX(x+1,list)

def getPointsY(y,list):
    if len(list) == 0:
        return []
    dir = list.pop(0)
    if dir in ['l','r','s']:
        return [y] + getPointsY(y,list)
    elif dir == 'd':
        return [y-1] + getPointsY(y-1,list)
    else:
        return [y+1] + getPointsY(y+1,list)

l1 = list('suuluuulululllldldll')
l1x = getPointsX(9.5,l1.copy())
l1y = getPointsY(-.5,l1.copy())

l2 = list('sululull')
l2x = getPointsX(12.5,l2.copy())
l2y = getPointsY(1.5,l2.copy())

l3 = list('sululululllldldldl')
l3x = getPointsX(9.5,l3.copy())
l3y = getPointsY(4.5,l3.copy())

l4 = list('sululululuuuuuluulu')
l4x = getPointsX(13.5,l4.copy())
l4y = getPointsY(2.5,l4.copy())

l5 = list('sldl')
l5x = getPointsX(9.5,l5.copy())
l5y = getPointsY(8.5,l5.copy())

l6 = list('sullllld')
l6x = getPointsX(8.5,l6.copy())
l6y = getPointsY(8.5,l6.copy())

l7 = list('sulluuuu')
l7x = getPointsX(6.5,l7.copy())
l7y = getPointsY(9.5,l7.copy())

l8 = list('sdddldlll')
l8x = getPointsX(3.5,l8.copy())
l8y = getPointsY(13.5,l8.copy())

l9 = list('sddldl')
l9x = getPointsX(1.3,l9.copy())
l9y = getPointsY(9.5,l9.copy())

grid[grid==0] = np.nan
plt.imshow(grid)
plt.gca().invert_yaxis()


# plotting clusters
if True:
    plt.plot(l1x,l1y,color="red")
    plt.plot(l2x,l2y,color="red")
    plt.plot(l3x,l3y,color="red")
    plt.plot(l4x,l4y,color="red")
    plt.plot(l5x,l5y,color="red")
    plt.plot(l6x,l6y,color="red")
    plt.plot(l7x,l7y,color="red")
    plt.plot(l8x,l8y,color="red")
    plt.plot(l9x,l9y,color="red")

plt.show()
