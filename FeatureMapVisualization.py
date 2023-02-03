# -*- coding: utf-8 -*-
# -*- Created by Hny on 2023.01.30 -*-
__Author__ = {"author":"Hny","email":"632678809@qq.com"}
############################################
# FeatureMapVisualization: 深度学习网络特征图可视化
# Import method: import FeatureMapVisualization as Fmv
############################################
"""
FeatureMapVisualization
=====

This class provides a way to visualize the feature maps of deep learning networks.
----------------------------
How to use the FeatureMapVisualization class:
Initialization flattens the network structure and stores the layers sequentially in a list.
    >> import FeatureMapVisualization as Fmv

    >> model = models.alexnet(pretrained=True)

    >> modelLayer = list(model.children())

    >> modelVisualization = Fmv.FeatureMapVisualization(modelLayer)

----------------------------
Provides functions and methods:

This method displays the layers of the flattened network:

    >> modelVisualization.showStructure()


This method shows the visualization of the feature map of the specified layer:

    >> modelVisualization.visualization(input,k=0,type='pdf',resolution=(256,256))


This method shows the visualization of all layer feature maps:

    >> modelVisualization.visualizationEveryLayer(input,type='pdf',resolution=(256,256))


"""


import torch,os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
plt.rcParams['font.sans-serif'] = ['STSong']

class FeatureMapVisualization:
    def __init__(self,model:list,isShow=True):
        self.maxItem = 0
        if ~(type(model) == list):
            self._callBack("2")
            return
        self.modelLayer = []
        self._flat(model)
        self.maxItem = len(self.modelLayer)
        if isShow:
            self.showStructure()

    def _flat(self, Nested_lists):
        try:
            length = len(Nested_lists)
            for i in range(length):
                self._flat(Nested_lists[i])
        except:
            self.modelLayer.append(Nested_lists)

    def showStructure(self):
        print("The model can be flattened to the following layers: ")
        for i in range(len(self.modelLayer)):
            print("[layer-%d]>>"%i,self.modelLayer[i])

    def _getImageInfo(self, imageFile):
        imageInfo = Image.open(imageFile).convert('RGB')
        imageTransform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        imageInfo = imageTransform(imageInfo)
        imageInfo = imageInfo.unsqueeze(0)
        return imageInfo

    def _getLayerFeatureMap(self, imageInfo, k=0):
        with torch.no_grad():
            for index, layer in enumerate(self.modelLayer):
                try:
                    imageInfo = layer(imageInfo)
                except:
                    imageInfo = imageInfo.view(1,imageInfo.shape[3]*imageInfo.shape[1]*imageInfo.shape[2])
                    imageInfo = layer(imageInfo)
                    imageInfo = imageInfo.view(imageInfo.shape[0],imageInfo.shape[1],1,1)
                if k == index:
                    return imageInfo

    def _makeDirs(self, thePath):
        import os
        # thePath = os.getcwd()+'featureMap\\layer-'+str(k)
        if not os.path.exists(thePath):
            supPath, _ = os.path.split(thePath)
            if not os.path.exists(supPath):
                self._makeDirs(supPath)
            os.mkdir(thePath)

    def visualization(self, imageFile, k=0, type='pdf',resolution=(256,256)):
        if k > self.maxItem or k < 0:
            self._callBack('1')
            return
        self._makeDirs(os.getcwd() + '\\featureMap\\layer-' + str(k))
        imageInfo = self._getImageInfo(imageFile)
        featureMap = self._getLayerFeatureMap(imageInfo, k = k)
        featureMap = featureMap.squeeze(0)

        featureMap = featureMap.view(1, featureMap.shape[0], featureMap.shape[1],featureMap.shape[2])
        upsample = torch.nn.UpsamplingBilinear2d(size=resolution)
        featureMap = upsample(featureMap)
        featureMap = featureMap.view(featureMap.shape[1], featureMap.shape[2], featureMap.shape[3])

        featureMapNum = featureMap.shape[0]
        row_Num = np.ceil(np.sqrt(featureMapNum))
        row_Num = int(row_Num)
        plt.figure()
        for index in range(1, featureMapNum + 1):
            plt.subplot(row_Num, row_Num, index)
            # plt.imshow(featureMap[index - 1], cmap='gray')  # feature_map[0].shape=torch.Size([55, 55])
            # plt.imshow(transforms.ToPILImage()(featureMap[index - 1]))
            plt.imshow(featureMap[index - 1])
            plt.axis('off')
            plt.imsave('featureMap//layer-'+str(k)+"//featureMap-" + str(index) + ".png", featureMap[index - 1])
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45)
        plt.savefig('featureMap//layer-'+str(k)+'//featureMap.'+type, bbox_inches='tight')
        plt.show()

    def visualizationEveryLayer(self, imageFile, type='pdf',resolution=(256,256)):
        for i in range(self.maxItem):
            print("[{r}/{t}]:{layer}".format(r=i,t=self.maxItem-1,layer=self.modelLayer[i]))
            self.visualization(imageFile, k=i, type=type,resolution=resolution)

    def _callBack(self,type:str):
        callBackTips = {
            "1":'"Error":The number of layers "k" does not match the number of layers of the model([0,%d]).'%(self.maxItem-1),
            "2":'"Error":The data type of the input model must be a list.'
        }
        print("\n\033[31m{}\033[0m".format(callBackTips[type]))

