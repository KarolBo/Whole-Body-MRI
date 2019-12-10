from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import random
from MyDicom import MyDicom
     
class MatplotlibWidget(QMainWindow):
    
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("main_window.ui",self)
        self.menuBar().setNativeMenuBar(False)
        self.setWindowTitle("Whole Body MRI")
        self.menu_load.triggered.connect(self.loadData)
        self.actionCoronal.triggered.connect(lambda: self.plane_change('cor'))
        self.actionSagittal.triggered.connect(lambda: self.plane_change('sag'))
        self.actionTransversal.triggered.connect(lambda: self.plane_change('tra'))
        self.addToolBar(NavigationToolbar(self.MplWidget.canvas, self))
        self.slider_slice.valueChanged.connect(self.refresh)
        self.slider_stack.valueChanged.connect(self.refresh)
        self.dicom_data = None
        self.plane = 'tra'

    def loadData(self):
        fileDialog = QFileDialog()
        fileDialog.setFileMode(QFileDialog.Directory)
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.dicom_data = MyDicom(path+'/')
        self.slider_slice.setMinimum(0)
        self.slider_slice.setMaximum(self.dicom_data.nOfSlices-1)
        self.slider_stack.setMinimum(0)
        self.slider_stack.setMaximum(self.dicom_data.nOfStacks-1)
        self.dicom_data.raport()
        self.refresh()

    def display(self, asp='auto'):
        self.MplWidget.canvas.axes.clear()
        slice = self.slider_slice.value()
        stack = self.slider_stack.value()
        image = self.dicom_data.getImg(slice, stack, self.plane)
        self.MplWidget.canvas.axes.imshow(image, cmap='jet', aspect=asp)
        self.MplWidget.canvas.draw()

    def refresh(self):
        z = self.slider_slice.value()
        s = self.slider_stack.value()
        if self.plane == 'tra':
            nz = self.dicom_data.nOfSlices
        if self.plane == 'cor':
            nz = self.dicom_data.x
        if self.plane == 'sag':
            nz = self.dicom_data.y
        ns = self.dicom_data.nOfStacks
        txt_slice = '{}/{}'.format(str(z), nz-1)
        txt_stack = 'b='+str(self.dicom_data.getTag('DiffusionBValue', slice=z, stack=s))
        if txt_stack == 'missing':
            txt_stack = '{}/{}'.format(str(s), ns-1)
        self.label_slice.setText(txt_slice)
        self.label_stack.setText(txt_stack)

        self.display()

    def plane_change(self, plane):
        self.plane = plane
        if plane == 'tra':
            asp = self.dicom_data.pixelSpacing[0]/self.dicom_data.pixelSpacing[1]
            self.slider_slice.setMaximum(self.dicom_data.nOfSlices-1)
        elif plane == 'cor':
            asp = self.dicom_data.pixelSpacing[1]/self.dicom_data.pixelSpacing[2]
            self.slider_slice.setMaximum(self.dicom_data.x-1)
        elif plane == 'sag':
            asp = self.dicom_data.pixelSpacing[0]/self.dicom_data.pixelSpacing[2]
            self.slider_slice.setMaximum(self.dicom_data.y-1)
        self.refresh()
        

app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()