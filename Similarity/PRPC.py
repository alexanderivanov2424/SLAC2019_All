from data_grid import DataGrid
import matplotlib.pyplot as plt
import matplotlib


from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import math

"""
################################

Peak Reduced PCA Clustering (PRPC)

Perform clustering using peak based dimension reduction (DBSCAN), then
PCA reduction, and then L2 based agglomerative clustering.
################################
"""


"""
Load Data and Peak Data
"""
#################################################################
### Path to Diffraction Data
##############################
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSub_1D.csv"""
path ="/home/sasha/Desktop/TiNiSn_500C-20190604T152446Z-001/TiNiSn_500C/"



dataGrid = DataGrid(path,regex)

#################################################################
### Path to PeakBBA csv files
###############################
#Note: must be the peakParams not curveParams (incompatable format atm)
data_dir = "/home/sasha/Desktop/TiNiSn_500C_PeakData_0.5/"
regex = """TiNiSn_500C_Y20190218_14x14_t60_(?P<num>.*?)_bkgdSu_peakParams.csv"""



peakGrid = DataGrid(data_dir,regex)
#################################################################
### Arguments
################
#True - clustering is based on presence of peak
#False - clustering is based on presence and amplitude of peak
boolean_peak_detection = True

#number of clusters to make
num_clusters = 6

#For PCA either use solver or force number of components in reduction
usePCASolver = True
PCA_components = 20
#file to save
# = "" to skip saving
save_as_file = ""




#remove column headers
for k in peakGrid.data.keys():
    peakGrid.data[k] = peakGrid.data[k][1:,:]


"""
Create a list of peaks in the form [x,y,p]
"""
SCALE = 100
def to_point(x,y,p):
    return [(x-1)/15.,(y-1)/15.,SCALE*float(p)/5]

peaks = []
for k in peakGrid.data.keys():
    x,y = peakGrid.coord(k)
    [peaks.append(to_point(x,y,p)) for p in peakGrid.data_at_loc(k)[:,1]]


"""
Cluster the peaks into C clusters
"""
X = np.array(peaks)

clustering = DBSCAN(eps=0.25, min_samples=10).fit(X)

C = len(set(clustering.labels_).difference(set([-1])))
"""
REDUCE DIMENSIONS BASED ON PEAK CLUSTERING
"""
M = np.zeros(shape=(peakGrid.size,C))

for k in peakGrid.data.keys():
    x,y = peakGrid.coord(k)
    V = np.zeros(shape=C)
    for i,p in enumerate(peakGrid.data_at_loc(k)[:,1]):
        loc = clustering.labels_[peaks.index(to_point(x,y,p))]
        if loc == -1:
            continue

        M[k-1,loc] = 1 if boolean_peak_detection else peakGrid.data_at_loc(k)[i,3]


"""
PCA ON REDUCED DIFFRACTION DATA
"""

if usePCASolver:
    pca = PCA(n_components = 'mle',svd_solver='full').fit_transform(M)
else:
    pca = PCA(n_components = PCA_components,svd_solver='full').fit_transform(M)

print("Dimension before PCA:",len(M[0]))
print("Dimension after PCA:",len(pca[0]))


"""
Grid Clustering based on similarity matrix
"""

def get_cluster_grids(i):
    agg = AgglomerativeClustering(n_clusters=i).fit(pca)

    hues = [float(float(x)/float(i)) for x in range(1,i+1)]

    cluster_centers = np.zeros(i)
    #sum vectors in each cluster
    cluster_sums = np.zeros(shape=(i,len(pca[0])))
    for val in range(1,178):
        cluster = agg.labels_[val-1]
        cluster_sums[cluster] = cluster_sums[cluster] + pca[0]

    #divide by cluster size to get average point
    for cluster in range(0,i):
        count = np.count_nonzero(agg.labels_==cluster)
        cluster_sums[cluster] = cluster_sums[cluster] / count

    for cluster,center_v in enumerate(cluster_sums):
        cluster_v = pca[np.where(agg.labels_==cluster)[0]]
        index = np.argmin(np.sum(np.square(cluster_v - center_v),axis=1))
        cluster_centers[cluster] = np.where(agg.labels_==cluster)[0][index] + 1

    #Cluster grid (colors)
    cluster_grid = np.zeros(shape = (15,15,3))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        cluster_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([hues[cluster],1,1])


    peak_max_counts = np.zeros(i)
    for val in range(1,178):
        cluster = agg.labels_[val-1]
        peak_max_counts[cluster] = max(peak_max_counts[cluster],len(peakGrid.data_at_loc(val)[:,1]))

    peak_grid = np.zeros(shape =(15,15,3))
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        k = len(peakGrid.data_at_loc(val)[:,1])/peak_max_counts[cluster]
        peak_grid[y-1][15-x] = matplotlib.colors.hsv_to_rgb([1,1,k])

    width_max = 0
    for val in range(1,178):
        width_max = max(width_max,np.nanmax(peakGrid.data_at_loc(val)[:,2].astype(np.float)))

    width_grid = np.zeros(shape =(15,15))
    width_grid.fill(np.nan)
    for val in range(1,178):
        x,y = dataGrid.coord(val)
        cluster = agg.labels_[val-1]
        k = np.nanmax(peakGrid.data_at_loc(val)[:,2].astype(np.float))
        width_grid[y-1][15-x] = k

    return cluster_grid, peak_grid, width_grid, cluster_centers,agg.labels_


"""
Plotting
"""
fig = plt.figure(figsize=(15,8))
cg,pg,wg,centers,labels = get_cluster_grids(num_clusters)
cg_next,pg_next,wg_next,centers_next,_ = get_cluster_grids(num_clusters+1)



ax1 = fig.add_subplot(2,3,1)
ax1.imshow(cg)
for j in range(dataGrid.size):
    x,y = dataGrid.coord(j+1)
    if (j+1) in centers:
        ax1.scatter(x-1-.1,y-1-.1,marker='o',color="white",s=70)
    ax1.annotate(str(j+1),(15-x-.4,y-1-.4),size=6)
ax1.axis("off")
ax1.invert_yaxis()
ax1.title.set_text(num_clusters)

ax2 = fig.add_subplot(2,3,2)
ax2.imshow(pg)
for j in range(dataGrid.size):
    x,y = dataGrid.coord(j+1)
    ax2.annotate(str(len(peakGrid.data_at_loc(j+1)[:,2])),(15-x-.4,y-1-.4),size=6,color="blue")
ax2.axis("off")
ax2.invert_yaxis()

"""
ax3 = fig.add_subplot(2,3,3)
ax3.imshow(cg)
for j in range(dataGrid.size):
    x,y = dataGrid.coord(j+1)
    ax3.annotate(str(len(peakGrid.data_at_loc(j+1)[:,2])),(x-1-.4,y-1-.4),size=6)
ax3.axis("off")
ax3.invert_yaxis()
"""

ax3 = fig.add_subplot(2,3,3)
#ax3.imshow(wg)
heatmap = ax3.pcolor(np.flip(wg,axis=0),cmap="viridis_r")
plt.colorbar(heatmap)
ax3.axis("off")
ax3.invert_yaxis()

ax4 = fig.add_subplot(2,3,4)
ax4.imshow(cg_next)
for j in range(dataGrid.size):
    x,y = dataGrid.coord(j+1)
    if (j+1) in centers_next:
        ax4.scatter(15-x-.1,y-1-.1,marker='o',color="white",s=70)
    ax4.annotate(str(j+1),(15-x-.4,y-1-.4),size=6)

ax4.axis("off")
ax4.invert_yaxis()
ax4.title.set_text(num_clusters+1)


split_locations = list(set(centers_next).difference(set(centers)))
k1 = int(split_locations[0])
if len(split_locations) == 2:
    k2 = int(split_locations[1])
else:
    k2 = int(centers[labels[k1-1]])

ax_plot = plt.subplot2grid((2,3), (1,1), colspan=2, rowspan=1)
x = dataGrid.data_at_loc(k1)[:,0]
y = dataGrid.data_at_loc(k1)[:,1]
ax_plot.plot(x,y,label=str(k1))
for peak in peakGrid.data_at_loc(k1)[:,1]:
    p = np.argmin(np.abs(x-float(peak)))
    ax_plot.plot(x[p],y[p],'x',color="blue")

x = dataGrid.data_at_loc(k2)[:,0]
y = dataGrid.data_at_loc(k2)[:,1]
ax_plot.plot(x,y,label=str(k2))
for peak in peakGrid.data_at_loc(k2)[:,1]:
    p = np.argmin(np.abs(x-float(peak)))
    ax_plot.plot(x[p],y[p],'x',color="orange")
ax_plot.legend()


plt.subplots_adjust(left=.1,right=.9,top=.9,bottom=.1)

if not save_as_file == "":
    plt.savefig(save_as_file + str(num_clusters) + ".png")
plt.show()
