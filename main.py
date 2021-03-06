from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import random
from MyDicom import MyDicom
from pca import reduce_dimensionality
from clustering import clustering
from sklearn.cluster import MiniBatchKMeans
#from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
#from sklearn.cluster import OPTICS
from math import ceil, acos, sqrt, log
import copy
import pickle
import matplotlib.pyplot as plt
from math import log as ln
from visualiser import *
     
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
        self.region = []
        self.label_array = None
        self.clf = None

        self.windowing_dialog = loadUi("windowing.ui")
        self.cluster_dialog = loadUi("clustering.ui")
        self.labeling_window = loadUi("labeling.ui")
        self.labeling_window.addToolBar(NavigationToolbar(self.labeling_window.MplWidget.canvas, self.labeling_window))
        self.clusters_view_dialog = loadUi("clusters_view.ui")
        self.svm_window = loadUi("svm.ui")
        self.svm_window.addToolBar(NavigationToolbar(self.svm_window.MplWidget.canvas, self.svm_window))
        self.labeling_window.menuBar().setNativeMenuBar(False)

        self.box_paths = []
        self.box_paths.append('/media/mrv6/ADATA SD700/Lukas_IVIM/vol1/IVIM1')
        self.box_paths.append('/media/mrv6/ADATA SD700/Lukas_IVIM/vol1/IVIM2')
        self.box_paths.append('/media/mrv6/ADATA SD700/Lukas_IVIM/vol1/IVIM3')
        self.box_paths.append('/media/mrv6/ADATA SD700/Lukas_IVIM/vol1/IVIM4')
        self.reload()
        self.open_labeling_dialog()
        self.label_array = np.load('/media/mrv6/ADATA SD700/Whole Body/labels/vol_1_no_muscles.npy')
        self.refresh()

        # self.calculate_attenuation()

        self.connect_handlers()

    def connect_handlers(self):
        self.menu_load.triggered.connect(self.loadData)
        self.menu_add.triggered.connect(self.addBox)
        self.menu_reload.triggered.connect(self.reload)
        self.menu_pca_2.triggered.connect(self.perform_pca_2)
        self.menu_clustering.triggered.connect(self.open_clustering_dialog)
        self.menu_labeling.triggered.connect(self.open_labeling_dialog)
        self.menu_mri.triggered.connect(lambda: self.set_view('mri'))
        self.menu_segmented.triggered.connect(lambda: self.set_view('segments'))
        self.menu_mask_none.triggered.connect(lambda: self.mask_change(self.menu_mask_none))
        self.menu_mask_positive.triggered.connect(lambda: self.mask_change(self.menu_mask_positive))
        self.menu_mask_negative.triggered.connect(lambda: self.mask_change(self.menu_mask_negative))
        self.actionCoronal.triggered.connect(lambda: self.plane_change('cor'))
        self.actionSagittal.triggered.connect(lambda: self.plane_change('sag'))
        self.actionTransversal.triggered.connect(lambda: self.plane_change('tra'))
        self.menu_windowing.triggered.connect(self.windowing_dialog.show)
        self.menu_svm.triggered.connect(self.svm_window.show)
        self.menu_plots.triggered.connect(self.get_b_plots)
        self.menu_overlay.triggered.connect(self.display_overlay)

        self.slider_slice.valueChanged.connect(self.slice_change)
        self.slider_stack.valueChanged.connect(self.refresh)
        self.windowing_dialog.slider_min.valueChanged.connect(self.refresh)
        self.windowing_dialog.slider_max.valueChanged.connect(self.refresh)

        self.MplWidget.canvas.mpl_connect('button_press_event', self.pressed_handler)

        self.clusters_view_dialog.slicewise.toggled.connect(self.scatter_clusters)
        self.clusters_view_dialog.slider.valueChanged.connect(self.scatter_clusters)

        self.cluster_dialog.button_run.clicked.connect(self.perform_clustering)
        self.cluster_dialog.button_display.clicked.connect(self.show_scatter_dialog)
        self.cluster_dialog.closeEvent = self.close_cluster_event
        
        self.labeling_window.closeEvent = self.close_labeling_event
        self.labeling_window.lineEdit.textChanged.connect(self.create_label_list)
        self.labeling_window.menu_label_save.triggered.connect(self.save_labels)
        self.labeling_window.menu_label_load.triggered.connect(self.load_labels)
        self.labeling_window.menu_add_labels.triggered.connect(self.add_labels)
        self.labeling_window.menu_clear_labels.triggered.connect(self.clear_labels)
        self.labeling_window.menu_scatter.triggered.connect(self.display_scatter)
        self.labeling_window.menu_3d.triggered.connect(self.vis_3d)
        self.labeling_window.button_label.clicked.connect(self.select_labeled_ared)
        self.labeling_window.button_clear.clicked.connect(self.clear_roi)
        self.labeling_window.button_mark.clicked.connect(self.mark_roi)
        self.labeling_window.button_back.clicked.connect(self.remove_point)

        self.svm_window.menu_load_model.triggered.connect(self.load_clf)
        # self.svm_window.menu_save_model.triggered.connect(self.save_svm_model)
        self.svm_window.button_train.clicked.connect(self.train_clf)
        self.svm_window.button_classify.clicked.connect(self.perform_clf)

    def vis_3d(self):
        m = MyDialog()
        m.spacing = self.dicom_data.pixelSpacing
        m.label = self.label_array
        m.rest = self.dicom_data.data_array[:,:,:,0]
        m.display()
        m.configure_traits()

    def calculate_attenuation(self):
        x = self.dicom_data.data_array.shape[0]
        y = self.dicom_data.data_array.shape[1]
        z = self.dicom_data.data_array.shape[2]
        n = self.dicom_data.data_array.shape[3]
        array = np.zeros([x, y, z, n-1])
        for b in range(0, n-1):
            idx_0 = np.where(self.dicom_data.data_array[:,:,:,0] == 0)
            idx_1 = np.where(self.dicom_data.data_array[:,:,:,0] != 0)
            array[idx_0[0],idx_0[1],idx_0[2],b] = 0
            array[idx_1[0],idx_1[1],idx_1[2],b] = np.divide(self.dicom_data.data_array[idx_1[0],idx_1[1],idx_1[2],b+1],
                                                            self.dicom_data.data_array[idx_1[0],idx_1[1],idx_1[2],0])
        idx = np.where(array != 0)
        array[idx] = np.log(array[idx])
        self.dicom_data.data_array = array

    def get_b_plots(self):
        class_list = ('air', 'eye', 'fluid', 'gray matter', 
                      'white matter', 'kidney', 'muscle', 
                      'pancreas', 'limphatic nodes', 'spleen', 
                      'testicle', 'liever')
        b_values = (0, 10, 50, 150, 200, 300, 500, 800)
        n_classes = int(self.label_array.max()) + 1
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for n in range(n_classes):
            idx = np.where(self.label_array == n)
            values = self.dicom_data.data_array[idx[0], idx[1], idx[2], :]
            data = values.mean(axis=0)
            sd = values.std(axis=0)
            color = plt.cm.Paired(n)
            ax.plot(b_values, data, color=color, label=class_list[n])
            plt.xlabel('b')
            plt.ylabel('S')
            ax.legend()
        plt.show()
            
    def train_clf_old(self):
        self.clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=16, n_jobs=-1)
        x = self.dicom_data.data_array.shape[0]
        y = self.dicom_data.data_array.shape[1]
        z = self.dicom_data.data_array.shape[2]
        n = self.dicom_data.data_array.shape[3]
        if self.svm_window.include_coordinates.isChecked():
            X = self.add_space_features(self.dicom_data.data_array)
            X = X.reshape([x*y*z, n+3])
        else:
            X = self.dicom_data.data_array.reshape([x*y*z, n])
        y = self.label_array.reshape([x*y*z])

        test_size = float(self.svm_window.val_set.text())/100.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101) 

        print("training...")
        self.clf.fit(X_train, y_train)
        print("The forest has grown!")
        
        # clf_file = open('random_forest', 'wb')
        # pickle.dump(self.clf, clf_file)
        # file = open('random_forest', 'rb')
        # self.clf = pickle.load(file)

        print("predicting...")
        pred = self.clf.predict(X_test)
        print("done!")
        acc = accuracy_score(y_test, pred, normalize=True)
        score = self.clf.score(X_test, y_test)
        print('accuracy:', acc)
        print('score:', score)

        self.visualize_clf_borders(X_test, y_test)

    def train_clf(self):
        self.clf = RandomForestClassifier(n_estimators=100, max_leaf_nodes=32, n_jobs=-1)
        x = self.dicom_data.data_array.shape[0]
        y = self.dicom_data.data_array.shape[1]
        z = self.dicom_data.data_array.shape[2]
        n = self.dicom_data.data_array.shape[3]

        if self.svm_window.include_coordinates.isChecked():
            X = self.add_space_features(self.dicom_data.data_array)
            n += 3
        else:
            X = self.dicom_data.data_array

        test_slice = X[156,:,:,:]
        test_labels = self.label_array[156,:,:]
        X_test = test_slice.reshape([x*z, n])
        y_test = test_labels.reshape([x*z])

        # X_train = np.delete(X, 156, 0)
        # labels = np.delete(self.label_array, 156, 0)
        X_train = X
        labels = self.label_array
        X_train = X_train.reshape([(x)*y*z, n])
        y_train = labels.reshape([(x)*y*z]) 

        print("training...")
        self.clf.fit(X_train, y_train)
        print("The forest has grown!")

        print("predicting...")
        X_pred = self.clf.predict(X_test)
        print("done!")
        acc = accuracy_score(y_test, X_pred, normalize=True)
        score = self.clf.score(X_test, y_test)
        print('accuracy:', acc)
        print('score:', score)

        predicted_slice = X_pred.reshape([x,z])
        asp = self.dicom_data.pixelSpacing[1]/self.dicom_data.pixelSpacing[2]
        self.svm_window.MplWidget.canvas.axes.clear()
        self.svm_window.MplWidget.canvas.axes.imshow(predicted_slice, cmap='jet', aspect=asp)
        self.svm_window.MplWidget.canvas.draw()

        self.visualize_clf_borders(X_test, y_test)

    def load_clf(self):
        # fileDialog = QFileDialog()
        # fileDialog.setFileMode(QFileDialog.ExistingFile)
        filename = str(QFileDialog.getOpenFileName(self)[0])
        print(filename)
        file = open(filename, 'rb')
        self.clf = pickle.load(file)
        print(type(self.clf))
        tree = self.clf.estimators_[5]
        self.depict_tree(tree)

    def depict_tree(self, tree):
        class_list = ('air', 'eye', 'fluid', 'gray matter', 
                      'white matter', 'kidney', 'muscle', 
                      'pancreas', 'limphatic nodes', 'spleen', 
                      'testicle', 'liever')
        export_graphviz(tree, out_file='tree.dot', 
                class_names = class_list,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

    def display_scatter(self):
        try:
            n_classes = int(self.label_array.max()) + 1
            print(n_classes, 'classes')
            idx = np.where(self.label_array != 0)
            X = self.dicom_data.data_array[idx[0], idx[1], idx[2]]
            y = self.label_array
            plt.scatter(X[:, 0], X[:, 1], s=1, cmap='jet')
            plt.xlabel('x2')
            plt.ylabel('x1')
            plt.show()
        except Exception as e:
            print(e)

    def visualize_clf_borders(self, X, y):
        try:
            colors = 'rgbcmy'
            plot_step = 0.05
            n_classes = int(self.label_array.max()) + 1

            a, X, b, y = train_test_split(X, y, test_size=0.05, random_state=101)
            print(X.shape)

            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                                 np.arange(y_min, y_max, plot_step))

            print("predicting...")
            Z = self.clf.predict(np.c_[xx.ravel(), yy.ravel()])
            print("done!")
            Z = Z.reshape(xx.shape)
            cs = plt.contourf(xx, yy, Z, cmap='jet')

            plt.axis("tight")

            # Plot the training points
            print(n_classes, 'classes')
            for i in range(n_classes):
                idx = np.where(y == i)
                color = [plt.cm.Paired(i),]
                plt.scatter(X[idx, 0], X[idx, 1], cmap='jet', s=1)

            plt.axis("tight")
            plt.show()
        except Exception as e:
            print(e)

    def perform_clf(self):
        try:
            x = self.dicom_data.data_array.shape[0]
            y = self.dicom_data.data_array.shape[1]
            z = self.dicom_data.data_array.shape[2]
            n = self.dicom_data.data_array.shape[3]
            if self.svm_window.include_coordinates.isChecked():
                X = self.add_space_features(self.dicom_data.data_array)
                X = X.reshape([x*y*z, n+3])
            else:
                X = self.dicom_data.data_array.reshape([x*y*z, n])
            y = self.label_array.reshape([x*y*z])
            test_size = float(self.svm_window.val_set.text())/100.0
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101) 

            file = open('random_forest', 'rb')
            self.clf = pickle.load(file)

            print("predicting...")
            pred = self.clf.predict(X_test)
            print("done!")
            acc = accuracy_score(y_test, pred, normalize=True)
            score = self.clf.score(X_test, y_test)
            print('accuracy:', acc)
            print('score:', score)

            self.visualize_clf_borders(X_test, y_test)
        except Exception as e:
            print(e)

    def add_space_features(self, matrix):
        x = matrix.shape[0]
        y = matrix.shape[1]
        z = matrix.shape[2]
        features = matrix.shape[3]
        new_matrix = np.zeros([x, y, z, features+3])
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    for f in range(features):
                        new_matrix[i,j,k,f] = matrix[i,j,k,f]
                        new_matrix[i,j,k,features] = i
                        new_matrix[i,j,k,features+1] = j
                        new_matrix[i,j,k,features+2] = k

        return new_matrix

    def reload(self):
        self.dicom_data = MyDicom(self.box_paths[0]+'/')
        for i in range(1, len(self.box_paths)):
            self.dicom_data.addBox(self.box_paths[i]+'/')
        self.refresh()

    def mask_change(self, menu):
        self.menu_mask_none.setChecked(False)
        self.menu_mask_positive.setChecked(False)
        self.menu_mask_negative.setChecked(False)
        menu.setChecked(True)

        if self.menu_mask_positive.isChecked():
            idx = np.where(self.label_array == 0)
            self.dicom_data.data_array[idx[0], idx[1], idx[2], :] = 0
        elif self.menu_mask_negative.isChecked():
            idx = np.where(self.label_array != 0)
            self.dicom_data.data_array[idx[0], idx[1], idx[2], :] = 0
        # self.dicom_data.data_array = np.ma.masked_equal(self.dicom_data.data_array, 0)

        self.refresh()

    def show_scatter_dialog(self):
        if self.dicom_data.data_array.shape[3] == 2:
            z = self.dicom_data.data_array.shape[2]
            self.clusters_view_dialog.show()
            self.clusters_view_dialog.slider.setRange(0, z-1)
            self.scatter_clusters()
        else:
            self.cluster_dialog.textEdit.setText('First, perform PCA!')

    def scatter_clusters(self):
        if self.clusters_view_dialog.slicewise.isChecked():
            groups = self.dicom_data.cluster_array.max()
            cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}
            self.clusters_view_dialog.MplWidget.canvas.axes.clear()
            z = self.clusters_view_dialog.slider.value()
            for g in range(int(groups)+1):
                idx = np.where(self.dicom_data.cluster_array[:,:,z,:] == g)
                x = self.dicom_data.data_array[idx[0],idx[1],z,0].flatten()
                y = self.dicom_data.data_array[idx[0],idx[1],z,1].flatten()
                self.clusters_view_dialog.MplWidget.canvas.axes.scatter(x, y, s=1, c=cdict[g%3])
                self.clusters_view_dialog.MplWidget.canvas.draw()
        else:
            groups = self.dicom_data.cluster_array.max()
            cdict = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}
            self.clusters_view_dialog.MplWidget.canvas.axes.clear()
            for g in range(int(groups)+1):
                idx = np.where(self.dicom_data.cluster_array == g)
                x = self.dicom_data.data_array[idx[0],idx[1],idx[2],0].flatten()
                y = self.dicom_data.data_array[idx[0],idx[1],idx[2],1].flatten()
                self.clusters_view_dialog.MplWidget.canvas.axes.scatter(x, y, s=1, c=cdict[g%3])
                self.clusters_view_dialog.MplWidget.canvas.draw()

    def perform_pca_2(self):
        self.dicom_data.data_array = reduce_dimensionality(self.dicom_data.data_array, 2)
        self.slider_stack.setMaximum(self.dicom_data.data_array.shape[3]-1)

    def slice_change(self):
        self.refresh()
        max_value = self.windowing_dialog.slider_min.maximum()
        min_value = self.windowing_dialog.slider_min.minimum()
        self.windowing_dialog.slider_min.setValue(min_value)
        self.windowing_dialog.slider_max.setValue(max_value)
        if self.labeling_window.isVisible():
            self.refresh_label_view()

    def loadData(self):
        fileDialog = QFileDialog()
        fileDialog.setFileMode(QFileDialog.Directory)
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.dicom_data = MyDicom(path+'/')
        self.slider_slice.setMinimum(0)
        self.slider_slice.setMaximum(self.dicom_data.data_array.shape[2]-1)
        self.slider_stack.setMinimum(0)
        self.slider_stack.setMaximum(self.dicom_data.data_array.shape[3]-1)
        self.slice_change()
        self.box_paths=[path,]
        print(path)

    def addBox(self):
        fileDialog = QFileDialog()
        fileDialog.setFileMode(QFileDialog.Directory)
        path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.dicom_data.addBox(path+'/')
        self.slider_slice.setMinimum(0)
        self.slider_slice.setMaximum(self.dicom_data.data_array.shape[2]-1)
        self.refresh()
        self.box_paths.append(path)
        print(path)

    def getImg(self):
        if self.view == 'mri':
            matrix = copy.deepcopy(self.dicom_data.data_array)
        else:
            matrix = copy.deepcopy(self.dicom_data.cluster_array)
        
        z = self.slider_slice.value()
        stack = self.slider_stack.value()

        if self.plane == 'tra':
            img = matrix[:, :, z, stack]
        elif self.plane == 'cor':
            img = matrix[z, :, :, stack]
        elif self.plane == 'sag':
            img = matrix[:, z, :, stack]

        return img
        
    def refresh(self):
        self.region = []

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

        image = self.getImg()
        self.set_windowing_sliders(image)
        val_min = self.windowing_dialog.slider_min.value()
        val_max = self.windowing_dialog.slider_max.value()
        image[image<val_min] = val_min
        image[image>val_max] = val_max
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.imshow(image, cmap='jet', aspect=asp)
        self.MplWidget.canvas.draw()

        plt.show()

    def display_overlay(self):
        z = self.slider_slice.value()
        image = self.getImg()
        if self.plane == 'tra':
            asp = self.dicom_data.pixelSpacing[0]/self.dicom_data.pixelSpacing[1]
            label_image = self.label_array[:,:,z]
        if self.plane == 'cor':
            asp = self.dicom_data.pixelSpacing[1]/self.dicom_data.pixelSpacing[2]
            label_image = self.label_array[z,:,:]
        if self.plane == 'sag':
            asp = self.dicom_data.pixelSpacing[0]/self.dicom_data.pixelSpacing[2]
            label_image = self.label_array[:,z,:]
        label_image = np.ma.masked_array(label_image, label_image == 0)
        plt.imshow(image, cmap='gray', aspect=asp)
        plt.imshow(label_image, cmap='jet', aspect=asp, alpha=0.6)
        plt.show()

    def set_windowing_sliders(self, img):
        val_min = np.amin(img)
        val_max = np.amax(img)
        self.windowing_dialog.slider_min.setRange(val_min, val_max)
        self.windowing_dialog.slider_max.setRange(val_min, val_max)

    def refresh_label_view(self):
        z = self.slider_slice.value()

        if self.plane == 'tra':
            asp = self.dicom_data.pixelSpacing[0]/self.dicom_data.pixelSpacing[1]
            self.slider_slice.setMaximum(self.label_array.shape[2]-1)
            image = self.label_array[:, :, z]
        if self.plane == 'cor':
            asp = self.dicom_data.pixelSpacing[1]/self.dicom_data.pixelSpacing[2]
            self.slider_slice.setMaximum(self.label_array.shape[0]-1)
            image = self.label_array[z, :, :]
        if self.plane == 'sag':
            asp = self.dicom_data.pixelSpacing[0]/self.dicom_data.pixelSpacing[2]
            self.slider_slice.setMaximum(self.label_array.shape[1]-1)
            image = self.label_array[:, z, :]

        self.labeling_window.MplWidget.canvas.axes.clear()
        self.labeling_window.MplWidget.canvas.axes.imshow(image, cmap='jet', aspect=asp)
        self.labeling_window.MplWidget.canvas.draw()

    def plane_change(self, plane):
        self.plane = plane
        self.refresh()

    def open_clustering_dialog(self):
        x = self.dicom_data.data_array.shape[0]
        y = self.dicom_data.data_array.shape[1]
        slices = self.dicom_data.data_array.shape[2]
        matrix = self.dicom_data.data_array[:,:,:,0].reshape([x, y, slices, 1])
        self.dicom_data.cluster_array = copy.deepcopy(matrix)
        self.cluster_dialog.fr.setText('0')
        self.cluster_dialog.to.setText(str(slices-1))
        self.cluster_dialog.show()

    def open_labeling_dialog(self):
        x = self.dicom_data.data_array.shape[0]
        y = self.dicom_data.data_array.shape[1]
        slices = self.dicom_data.data_array.shape[2]
        self.label_array = np.zeros([x, y, slices])
        self.labeling_window.show()

    def perform_clustering(self):
        self.cluster_dialog.textEdit.setText('Custering...')

        if self.cluster_dialog.checkbox_pca.isChecked():
            dim = int(self.cluster_dialog.line_pca.text())
            matrix = reduce_dimensionality(self.dicom_data.data_array, dim)
        else:
            matrix = self.dicom_data.data_array

        fr = int(self.cluster_dialog.fr.text())
        to = int(self.cluster_dialog.to.text())
        matrix[:,:,:fr,:] = 0
        matrix[:,:,to:,:] = 0

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
        self.labeling_window.lineEdit.setText(str(clusters))
        self.set_view('segments')

    def set_view(self, mode):
        self.view = mode
        self.refresh()
            
    def pressed_handler(self, event):
        try:
            self.pressed = True
            x = int(event.xdata)
            y = int(event.ydata)
            z = self.slider_slice.value()
            self.clickPoint = (x, y)
            if self.labeling_window.isVisible():
                if self.labeling_window.cluster.isChecked() and self.view == 'segments':
                    self.region = self.region_growing(x, y)
                elif self.labeling_window.pixel.isChecked():
                    self.region.append((y,x,z))
                elif self.labeling_window.roi.isChecked():
                    self.polygon.append((y,x,z))
                    self.MplWidget.canvas.axes.scatter(x, y, s=1)
                    self.MplWidget.canvas.draw()
                self.drawRegion()
        except Exception as e:
            print('click error:', e)
            
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
        maxX = max([x[0] for x in self.polygon])
        maxY = max([x[1] for x in self.polygon])
        minX = min([x[0] for x in self.polygon])
        minY = min([x[1] for x in self.polygon])
        return (minX, maxX, minY, maxY)

    def drawRegion(self):
        for point in self.region:
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

    def region_growing(self, y, x): # mind it!
        z = self.slider_slice.value()
        if self.plane == 'tra':
            image = self.dicom_data.cluster_array[:,:,z,0]
        if self.plane == 'cor':
            image = self.dicom_data.cluster_array[z,:,:,0]
        if self.plane == 'sag':
            image = self.dicom_data.cluster_array[:,z,:,0]
        value = image[x,y]
        
        region = [(x,y,z),]
        seeds = [(x,y,z),]
        visited = {(x,y,z),}

        while seeds:
            seed = seeds.pop()
            x = seed[0]
            y = seed[1]
            for i in range(x-1, x+2):
                for j in range(y-1, y+2):
                    point = (i,j,z)
                    try:
                        if image[i,j] == value:
                            if point in visited:
                                continue
                            seeds.append(point)
                            region.append(point)
                        visited.add(point)
                    except:
                        pass
        return region

    def clear_roi(self):
        self.region = []
        self.polygon = []
        self.refresh()

    def mark_roi(self):
        box = self.getPolygonBox()
        z = self.slider_slice.value()
        for x in range(int(box[0]-1),int(box[1]+1)):
            for y in range(int(box[2]-1),int(box[3]+1)):
                inside = self.isInside([x+0.5,y+0.5])
                if inside:
                    self.region.append((x,y,z))
        self.drawRegion()
        self.polygon = []
        region = []

    def remove_point(self):
        self.polygon.pop()
        self.refresh()
        for point in self.polygon:
            self.MplWidget.canvas.axes.scatter(point[1], point[0], s=1)
        self.MplWidget.canvas.draw()

    def select_labeled_ared(self):
        try:
            if self.labeling_window.whole_cluster.isChecked():
                cluster_class = self.labeling_window.cluster_num.text()
                cluster_label = self.labeling_window.cluster_label.text()
                idx = np.where(self.dicom_data.cluster_array[:,:,:,0] == int(cluster_class))
                self.label_array[idx] = int(cluster_label)
                self.refresh_label_view()
            elif self.labeling_window.roi.isChecked():
                z = self.slider_slice.value()
                label = self.labeling_window.class_combo.currentText()
                np_region = np.array(self.region)
                self.region = []

                if self.plane == 'tra':
                    self.label_array[tuple(np_region[:,0]), tuple(np_region[:,1]), tuple(np_region[:,2])] = int(label)
                elif self.plane == 'cor':
                    self.label_array[tuple(np_region[:,2]), tuple(np_region[:,0]), tuple(np_region[:,1])] = int(label)
                elif self.plane == 'sag':
                    self.label_array[tuple(np_region[:,0]), tuple(np_region[:,2]), tuple(np_region[:,1])] = int(label)
            else:
                z = self.slider_slice.value()
                label = self.labeling_window.class_combo.currentText()
                np_region = np.array(self.region)
                self.region = []

                if self.plane == 'tra':
                    self.label_array[tuple(np_region[:,0]), tuple(np_region[:,1]), tuple(np_region[:,2])] = int(label)
                elif self.plane == 'cor':
                    self.label_array[tuple(np_region[:,2]), tuple(np_region[:,0]), tuple(np_region[:,1])] = int(label)
                elif self.plane == 'sag':
                    self.label_array[tuple(np_region[:,0]), tuple(np_region[:,2]), tuple(np_region[:,1])] = int(label)

                self.refresh_label_view()
        except Exception as e:
            print(e)

    def create_label_list(self, text):
        n = int(text)
        self.labeling_window.class_combo.clear()
        for i in range(n+1):
            self.labeling_window.class_combo.addItem(str(i))

    def save_labels(self):
        path = QFileDialog.getSaveFileName(self)[0]
        np.save(path, self.label_array)

    def load_labels(self):
        path = QFileDialog.getOpenFileName(self)[0]
        if self.labeling_window.radio_load_as.isChecked():
            self.label_array = np.load(path)
            idx = np.where(self.label_array)
            self.label_array[idx] = int(self.labeling_window.load_as.text())
        else:
            self.label_array = np.load(path)
        self.refresh_label_view()

    def add_labels(self):
        path = QFileDialog.getOpenFileName(self)[0]
        if self.labeling_window.radio_load_as.isChecked():
            new_label_array = np.load(path)
            idx = np.where(new_label_array)
            self.label_array[idx] = int(self.labeling_window.load_as.text())
        else:
            self.label_array = self.label_array + np.load(path)
        self.refresh_label_view()

    def clear_labels(self):
        self.label_array = np.zeros(self.label_array.shape)
        self.refresh_label_view()

    def closeEvent(self, event):
        print('bye')
        self.cluster_dialog.accept()
        self.windowing_dialog.accept()
        self.labeling_window.close()
        self.svm_window.close()
        event.accept()

    def close_cluster_event(self, event):
        self.set_view('mri')
        del self.dicom_data.cluster_array
        self.clusters_view_dialog.accept()
        event.accept()

    def close_labeling_event(self, event):
        self.label_array = None
        self.region = []
        event.accept()

##########################################################################################

app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()
