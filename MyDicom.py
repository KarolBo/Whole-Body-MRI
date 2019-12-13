import pydicom
import os
import numpy as np
from math import ceil

###############################################################################

class MyDicom(object):
    
    def __init__(self, path):
        fileList = self.loadData(path)
        self.pixelSpacing = None
        self.data_array = self.load_array(fileList)
        self.cluster_array = None
        
###############################################################################

    def loadData(self, path):
        fileList = []
        allFiles = os.listdir(path)
        for filename in allFiles:
            if os.path.isfile(path+filename) and \
                (".dcm" in filename.lower() or "." not in filename):
                fileList.append(path+filename)
        fileList = sorted(fileList, key=lambda x: pydicom.dcmread(x).ImagePositionPatient[-1])

        return fileList
         
###############################################################################
    
    def getNumOfStacks(self, someImage, fileList):
        sliceDict =	dict()
        for n in range(0, len(fileList)):
            location = pydicom.dcmread(fileList[n]).ImagePositionPatient[-1]
            if location in sliceDict:
                sliceDict[location] = sliceDict.get(location) + 1 
            else:
                sliceDict[location] = 1
        return list(sliceDict.values())[0]
 
###############################################################################
    
    def getImg(self, z, stack, plane='tra', view='mri'):
        if view == 'mri':
            data_array = self.data_array
        else:
            data_array = self.cluster_array
            
        if plane == 'tra':
            return data_array[:, :, z, stack]
        elif plane == 'cor':
            return data_array[z, :, :, stack]
        elif plane == 'sag':
            return data_array[:, z, :, stack]
                   
###############################################################################

    def getTag(self, name, default='missing', slice=0, stack=0):
        try:
            imageNum = slice*self.nOfStacks + stack
            ds = pydicom.dcmread(self.fileList[imageNum])
            tag = ds.get(name, default)
        except:
            tag = 'missing'
        return tag

###############################################################################

    def load_array(self, fileList):
        first = pydicom.dcmread(fileList[0])
        pixel_spacing = (float(first.PixelSpacing[0]),
                             float(first.PixelSpacing[1]), 
                             float(first.SpacingBetweenSlices))
        if self.pixelSpacing == None:
            self.pixelSpacing = pixel_spacing
        elif self.pixelSpacing != pixel_spacing:
            raise exception('Datasets are not compatibile!')

        x = int(first.Rows)
        y = int(first.Columns)
       
        n = len(fileList) 
        sliceDict = dict()
        array = np.zeros([x, y, n])

        for i, file in enumerate(fileList):
            dicom = pydicom.dcmread(file)
            location = dicom.ImagePositionPatient[-1]
            array[:, :, i] = dicom.pixel_array
            if location in sliceDict:
                sliceDict[location] = sliceDict.get(location) + 1 
            else:
                sliceDict[location] = 1
        
        num_of_stacks = list(sliceDict.values())[0]
        num_of_slices = len(fileList) // num_of_stacks
        
        return array.reshape([x, y, num_of_slices, num_of_stacks])

###############################################################################

    def addBox(self, path):
        file_list = self.loadData(path)
        # self.setImageParameters(file_list)
        new_box_array = self.load_array(file_list)
        self.data_array = np.concatenate([new_box_array, self.data_array], axis=2)