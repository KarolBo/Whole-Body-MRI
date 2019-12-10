import pydicom
import os
import numpy as np
from math import ceil

###############################################################################

class MyDicom(object):
    
    def __init__(self, path):
        self.path = path
        self.fileList = self.loadData()
        self.setImageParameters()
        self.data_array = self.load_array()
        
###############################################################################

    def loadData(self):
        fileList = []
        allFiles = os.listdir(self.path)
        for filename in allFiles:
            if os.path.isfile(self.path+filename) and \
                (".dcm" in filename.lower() or "." not in filename):
                fileList.append(self.path+filename)
        fileList = sorted(fileList, key=lambda x: pydicom.dcmread(x).ImagePositionPatient[-1])

        return fileList
    
###############################################################################
        
    def setImageParameters(self):
       first = pydicom.dcmread(self.fileList[0])
       self.modality = first.Modality
       self.nOfStacks = self.getNumOfStacks(first)
       self.nOfSlices = int(len(self.fileList)/self.nOfStacks)
       self.x = int(first.Rows)
       self.y = int(first.Columns)
       self.pixelSpacing = (float(first.PixelSpacing[0]), \
                float(first.PixelSpacing[1]), float(first.SliceThickness))
       self.dataType = first.pixel_array.dtype
       self.fov = (self.x*self.pixelSpacing[0], self.y*self.pixelSpacing[1])
       self.imagePosition = first.ImagePositionPatient
       try:
           self.sliceSpacing = first.SpacingBetweenSlices
       except:
           self.sliceSpacing = self.pixelSpacing[2]
         
###############################################################################
    
    def getNumOfStacks(self, someImage):
        sliceDict =	dict()
        for n in range(0, len(self.fileList)):
            location = pydicom.dcmread(self.fileList[n]).ImagePositionPatient[-1]
            if location in sliceDict:
                sliceDict[location] = sliceDict.get(location) + 1 
            else:
                sliceDict[location] = 1
        return list(sliceDict.values())[0]
 
###############################################################################
    
    def getImg(self, z, stack, plane='tra'):
        if plane == 'tra':
            return self.data_array[:, :, z, stack]
        elif plane == 'cor':
            return self.data_array[z, :, :, stack]
        elif plane == 'sag':
            return self.data_array[:, z, :, stack]

###############################################################################
    
    def sliceAtPosition(self, pos):
        firstSlice = pydicom.dcmread(self.fileList[0]).SliceLocation
        return int(0.5+abs(pos-firstSlice)/self.sliceSpacing)

###############################################################################
       
    def raport(self):
        print("X dimension: %d"%self.x)
        print("Y dimension: %d"%self.y)
        print("Number of slices: "+str(self.nOfSlices))
        print("Number of stacks: "+str(self.nOfStacks))
        print("Pixel spacing %f, %f, %f"%(self.pixelSpacing[0],
              self.pixelSpacing[1], self.pixelSpacing[2]))
        print("Data type: "+str(self.dataType))
        print("FoV: %f x %f mm"%self.fov)
        print("Image position (upper left corner): %f, %f" \
              %(self.imagePosition[0], self.imagePosition[1]))
        print("Spacing between slices: %f"%self.sliceSpacing)
        
###############################################################################
        
    def getSlicePos(self, n):
        return pydicom.dcmread(self.fileList[n]).SliceLocation
        
###############################################################################
    
    def getSubimage(self, corner, fov, z, stack):
        "returns subimage for a given upper left corner [mm] and FoV [mm]"
        cx = self.imagePosition[0]-corner[0]
        cy = self.imagePosition[1]-corner[1]
        x1, y1 = self.mm2vox((cx,cy))
        extent = self.mm2vox(fov)
        x2 = x1 + extent[0]
        y2 = y1 + extent[1]
#        img = self.getStackSum(z)
        img = self.getImg(z,stack)
        ret = img[y1:y2,x1:x2]
        return ret
        
###############################################################################

    def getStackSum(self, z):
        sumImg = np.zeros((self.x, self.y))
        for stack in range(self.nOfStacks):  
            sumImg = np.add(sumImg, self.getImg(z, stack))
        return sumImg
            
 ###############################################################################

    def mm2vox(self, mm):
        dimX = mm[0]*float(self.x)/self.fov[0]
        dimY = mm[1]*float(self.y)/self.fov[1]
        return (ceil(dimX), int(dimY))            
            
###############################################################################

    def getDistance(self, corner):
        cx = self.imagePosition[0]-corner[0]
        cy = self.imagePosition[1]-corner[1]
        x1, y1 = self.mm2vox((cx,cy))  
        return (x1,y1)

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

    def load_array(self):
        array = np.zeros([self.x, self.y, self.nOfSlices, self.nOfStacks])
        for stack in range(self.nOfStacks):
            for slice in range(self.nOfSlices):
                imageNum = slice*self.nOfStacks + stack
                array[:, :, slice, stack] = pydicom.dcmread(self.fileList[imageNum]).pixel_array
        return array

###############################################################################