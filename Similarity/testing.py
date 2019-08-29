from appJar import gui
import numpy as np

dataDir = ""
peakDir = ""
curveDir = ""







# create a GUI variable called app
app = gui("Clustering Analysis","800x400")
app.addLabel("title", "Welcome to appJar")
app.setLabelBg("title", "red")



#fileMenus = ["Open", "Save", "Save as...", "-", "Close"]
#app.addMenuList("File", fileMenus, menuPress)
#app.createMenu("File")

#app.addStatusbar(fields=1)
#app.setStatusbar("Data Directory: " + str(dataDir), 0)

def loadData(button):
    if button == "Load Data":
        dataDir = app.getEntry("data")
        peakDir = app.getEntry("peaks")



app.startPanedFrame("options", row=0, column=0)
app.setSticky("news")
app.setStretch("column")
app.addLabel("opt_title", "Data Options")
app.setLabelBg("opt_title", "grey")

app.addLabel("Background Subtracted 1D Data")
app.addDirectoryEntry("data")

app.addLabel("Peak Parameters")
app.addDirectoryEntry("curves")

app.addButtons(["Load Data"],loadData)
app.stopPanedFrame()

#app.directoryBox(title=None, dirName=None, parent=None)

app.startFrame("all plots",row=0,column=1)

app.startFrame("plots",row=0,column=0)

app.startFrame("clustering_options",row=0,column=0)
app.addLabel("l", "CLustering Options")
app.setLabelBg("l", "red")
app.stopFrame()

app.startFrame("clustering_plot",row=0,column=1)
fig = app.addPlotFig("cluster")
ax = fig.add_subplot(1,1,1)
ax.imshow(np.zeros(shape=(5,5)))
app.refreshPlot("cluster")
app.stopFrame()

app.stopFrame()

app.startFrame("diffraction",row=1,column=0)
fig = app.addPlotFig("diff")
ax = fig.add_subplot(1,1,1)
ax.plot([i for i in range(10)],[np.random.random() for i in range(10)])
app.refreshPlot("diff")
app.stopFrame()

app.stopFrame()


app.go()
