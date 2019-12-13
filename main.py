from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import random
from MyDicom import MyDicom
from pca import reduce_dimensionality
from clustering import clustering
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from math import ceil, acos, sqrt, log
import copy
# from matplotlib import pyplot as plt
     
class MatplotlibWidget(QMainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("main_window.ui",self)
        self.menuBar().setNativeMenuBar(False)
        self.setWindowTitle("Whole Body MRI")
        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
        
        self.dicom_data = None
        self.plane = 'tra'
        self.view = 'mri'
        self.pressed = False
        self.markROI = False
        self.polygon = []

        self.cluster_dialog = loadUi("clustering.ui")
        self.labeling_dialog = loadUi("labeling.ui")

        self.connect_handlers()

    def connect_handlers(self):
        self.menu_load.triggered.connect(self.loadData)
        self.menu_add.triggered.connect(self.addBox)
        self.menu_pca.triggered.connect(lambda: reduce_dimensionality(self.dicom_data.data_array, 3))
        self.menu_clustering.triggered.connect(self.open_clustering_dialog)
        self.menu_labeling.triggered.connect(self.labeling_dialog.show)
        self.menu_mri.triggered.connect(lambda: self.set_view('mri'))
        self.menu_segmented.triggered.connect(lambda: self.set_view('segments'))
        self.actionCoronal.triggered.connect(lambda: self.plane_change('cor'))
        self.actionSagittal.triggered.connect(lambda: self.plane_change('sag'))
        self.actionTransversal.triggered.connect(lambda: self.plane_change('tra'))
       
        self.slider_slice.valueChanged.connect(self.refresh)
        self.slider_stack.valueChanged.connect(self.refresh)

        self.MplWidget.canvas.mpl_connect('button_press_event', self.pressed_handler)
        self.MplWidget.canvas.mpl_connect('button_release_event', self.released_handler)
        self.MplWidget.canvas.mpl_connect('motion_notify_event', self.moved_handler)

        self.cluster_dialog.button_run.clicked.connect(self.perform_clustering)
        self.labeling_dialog.button_label.clicked.connect(self.save_labels)

    def loadData(self):
        fileDialog = QFileDialog()
        fileDialog.setFileMode(QFileDialog.Directory)
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.dicom_data = MyDicom(path+'/')
        self.slider_slice.setMinimum(0)
        self.slider_slice.setMaximum(self.dicom_data.data_array.shape[2]-1)
        self.slider_stack.setMinimum(0)
        self.slider_stack.setMaximum(self.dicom_data.data_array.shape[3]-1)
        # self.dicom_data.raport()
        self.refresh()

    def addBox(self):
        fileDialog = QFileDialog()
        fileDialog.setFileMode(QFileDialog.Directory)
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.dicom_data.addBox(path+'/')
        self.slider_slice.setMinimum(0)
        self.slider_slice.setMaximum(self.dicom_data.data_array.shape[2]-1)
        self.refresh()
        
    def refresh(self):
        if self.view == 'mri':
            data_array = self.dicom_data.data_array
        else:
            data_array = self.dicom_data.cluster_array

        z = self.slider_slice.value()
        s = self.slider_stack.value()
        if self.plane == 'tra':
            nz = data_array.shape[2]
            asp = self.dicom_data.pixelSpacing[0]/self.dicom_data.pixelSpacing[1]
            self.slider_slice.setMaximum(data_array.shape[2]-1)
        if self.plane == 'cor':
            nz = data_array.shape[0]
            asp = self.dicom_data.pixelSpacing[1]/self.dicom_data.pixelSpacing[2]
            self.slider_slice.setMaximum(data_array.shape[0]-1)
        if self.plane == 'sag':
            nz = data_array.shape[1]
            asp = self.dicom_data.pixelSpacing[0]/self.dicom_data.pixelSpacing[2]
            self.slider_slice.setMaximum(data_array.shape[1]-1)
        self.slider_stack.setMaximum(data_array.shape[3]-1)
        ns = data_array.shape[3]
        txt_slice = '{}/{}'.format(str(z), nz-1)
        txt_stack = 'b='+str(self.dicom_data.getTag('DiffusionBValue', slice=z, stack=s))
        if txt_stack == 'b=missing':
            txt_stack = '{}/{}'.format(str(s), ns-1)
        self.label_slice.setText(txt_slice)
        self.label_stack.setText(txt_stack)

        self.MplWidget.canvas.axes.clear()
        slice = self.slider_slice.value()
        stack = self.slider_stack.value()
        image = self.dicom_data.getImg(slice, stack, self.plane, self.view)
        self.MplWidget.canvas.axes.imshow(image, cmap='jet', aspect=asp)
        self.MplWidget.canvas.draw()

    def plane_change(self, plane):
        self.plane = plane
        self.refresh()

    def open_clustering_dialog(self):
        x = self.dicom_data.data_array.shape[0]
        y = self.dicom_data.data_array.shape[1]
        slices = self.dicom_data.data_array.shape[2]
        matrix = self.dicom_data.data_array[:,:,:,0].reshape([x, y, slices, 1])
        self.dicom_data.cluster_array = copy.deepcopy(matrix)
        self.cluster_dialog.show()

    def perform_clustering(self):
        if self.cluster_dialog.checkbox_pca.isChecked():
            dim = int(self.cluster_dialog.line_pca.text())
            matrix = reduce_dimensionality(self.dicom_data.data_array, dim)
        else:
            matrix = self.dicom_data.data_array

        coordinates = self.cluster_dialog.checkbox_coords.isChecked()    
        clusters = int(self.cluster_dialog.line_clusters.text())

        if self.cluster_dialog.kmean.isChecked():
            function = MiniBatchKMeans
        elif self.cluster_dialog.agglomerative.isChecked():
            function = AgglomerativeClustering
        elif self.cluster_dialog.dbscan.isChecked():
            function = DBSCAN
        elif self.cluster_dialog.optics.isChecked():
            function = OPTICS
        
        if self.cluster_dialog.slicewise.isChecked():
            w = matrix.shape[0]
            h = matrix.shape[1]
            slices = matrix.shape[2]
            stacks = matrix.shape[3]
            z = self.slider_slice.value()
            

            if self.plane == 'tra':
                matrix = matrix[:,:,z,:].reshape([w, h, 1, stacks])
                self.dicom_data.cluster_array[:,:,z,0] = clustering(matrix, function, clusters, coordinates)[:,:,0,0]
            if self.plane == 'cor':
                matrix = matrix[z,:,:,:].reshape([1, h, slices, stacks])
                self.dicom_data.cluster_array[z,:,:,0] = clustering(matrix, function, clusters, coordinates)[0,:,:,0]
            if self.plane == 'sag':
                matrix = matrix[:,z,:,:].reshape([w, 1, slices, stacks])
                self.dicom_data.cluster_array[:,z,:,0] = clustering(matrix, function, clusters, coordinates)[:,0,:,0]
            
        else:
            self.dicom_data.cluster_array = clustering(matrix, function, clusters, coordinates)
        self.cluster_dialog.textEdit.setText('Custering acomplished!')
        self.set_view('segments')

    def set_view(self, mode):
        self.view = mode
        self.refresh()
            
    def pressed_handler(self, event):
        try:
            self.pressed = True
            x = int(event.xdata)
            y = int(event.ydata)
            self.clickPoint = (x, y)
            if (self.labeling_dialog.isVisible() and 
                self.labeling_dialog.cluster.isChecked() and
                self.view == 'segments'):
                region = self.region_growing(x, y)
                self.drawRegion(region)
        except Exception as e:
            print('click error:', e)
        
    def released_handler(self, event):
        try:
            self.pressed = False
            if (self.labeling_dialog.isVisible() and self.labeling_dialog.roi.isChecked()):
                self.drawPolygon()
            self.polygon.clear()
            del self.clickPoint     
        except Exception as e:
            print('release error:', e)   
    
    def moved_handler(self, event):
        try:
            x = event.xdata
            y = event.ydata
            if (self.pressed and 
               self.labeling_dialog.isVisible() and 
               self.labeling_dialog.roi.isChecked()):
                if ((x,y) not in self.polygon):
                    self.MplWidget.canvas.axes.scatter(x, y, s=3)
                    self.polygon.append((x, y))
                self.MplWidget.canvas.draw()
        except Exception as e:
            print('move error:',e)
            
    def isInside(self, point):
        total = 0.0
        deltaX0 = self.polygon[0][0] - point[0]
        deltaY0 = self.polygon[0][1] - point[1]
        deltaXprev = deltaX0
        deltaYprev = deltaY0
        
        for borderPoint in self.polygon:
            deltaX = borderPoint[0]-point[0]
            deltaY = borderPoint[1]-point[1]
            
            arg = (deltaXprev*deltaX + deltaYprev * deltaY) / \
            (sqrt(deltaXprev**2 + deltaYprev**2)*sqrt(deltaX**2 + deltaY**2))
            if arg>1:
                arg=1
            if arg<-1:
                arg=-1
            angle = acos(arg) 
            
            if (deltaX*deltaYprev - deltaY * deltaXprev < 0):
                total += angle
            else:
                total -= angle
            
            deltaXprev = deltaX
            deltaYprev = deltaY
        
        # the first point connected to the last one
        arg = (deltaXprev*deltaX0 + deltaYprev * deltaY0) / \
        (sqrt(deltaXprev**2 + deltaYprev**2)*sqrt(deltaX0**2 + deltaY0**2))
        if arg>1:
            arg=1
        if arg<-1:
            arg=-1
        angle = acos(arg)
        
        if (deltaX0*deltaYprev - deltaY0 * deltaXprev < 0):    
            total += angle
        else:
            total -= angle
            
        return abs(total) > 3.14

    def getPolygonBox(self):
        maxX, maxY = list(map(max,*self.polygon))
        minX, minY = list(map(min,*self.polygon))
        return (minX, maxX, minY, maxY)

    def drawPolygon(self):
        box = self.getPolygonBox()
        for x in range(int(box[0]-1),int(box[1]+1)):
            for y in range(int(box[2]-1),int(box[3]+1)):
                inside = self.isInside([x+0.5,y+0.5])
                if inside:
                    self.MplWidget.canvas.axes.scatter(x, y, s=3)
        self.MplWidget.canvas.draw()
        print('drawn')

    def drawRegion(self, region):
        for point in region:
            self.MplWidget.canvas.axes.scatter(point[1], point[0], s=1)
        self.MplWidget.canvas.draw()

    def isInside(self, point):
        total = 0.0
        deltaX0 = self.polygon[0][0] - point[0]
        deltaY0 = self.polygon[0][1] - point[1]
        deltaXprev = deltaX0
        deltaYprev = deltaY0
        
        for borderPoint in self.polygon:
            deltaX = borderPoint[0]-point[0]
            deltaY = borderPoint[1]-point[1]
            
            arg = (deltaXprev*deltaX + deltaYprev * deltaY) / \
            (sqrt(deltaXprev**2 + deltaYprev**2)*sqrt(deltaX**2 + deltaY**2))
            if arg>1:
                arg=1
            if arg<-1:
                arg=-1
            angle = acos(arg) 
            
            if (deltaX*deltaYprev - deltaY * deltaXprev < 0):
                total += angle
            else:
                total -= angle
            
            deltaXprev = deltaX
            deltaYprev = deltaY
        
        # the first point connected to the last one
        arg = (deltaXprev*deltaX0 + deltaYprev * deltaY0) / \
        (sqrt(deltaXprev**2 + deltaYprev**2)*sqrt(deltaX0**2 + deltaY0**2))
        if arg>1:
            arg=1
        if arg<-1:
            arg=-1
        angle = acos(arg)
        
        if (deltaX0*deltaYprev - deltaY0 * deltaXprev < 0):    
            total += angle
        else:
            total -= angle
            
        return abs(total) > 3.14

    def region_growing(self, y, x):
        z = self.slider_slice.value()
        if self.plane == 'tra':
            image = self.dicom_data.cluster_array[:,:,z,0]
        if self.plane == 'cor':
            image = self.dicom_data.cluster_array[z,:,:,0]
        if self.plane == 'sag':
            image = self.dicom_data.cluster_array[:,z,:,0]
        value = image[x,y]
        
        region = [(x,y),]
        seeds = [(x,y),]

        while seeds:
            seed = seeds.pop()
            x = seed[0]
            y = seed[1]
            for i in range(x-1, x+2):
                for j in range(y-1, y+2):
                    try:
                        if image[i,j] == value:
                            point = (i,j)
                            if point not in region:
                                seeds.append(point)
                                region.append(point)
                    except:
                        pass
        return region

    def save_labels(self):
        pass

    def closeEvent(self, event):
        print('bye')
        self.cluster_dialog.accept()
        self.labeling_dialog.accept()
        event.accept()

##########################################################################################

app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()