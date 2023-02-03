# # -*- coding: utf-8 -*-
# # -*- Created by Hny on 2023.01.20 -*-
# ############################################
# # FeatureMapVisualization:深度学习网络特征图可视化
# #
# ############################################

from torchvision import models
from FeatureMapVisualization import FeatureMapVisualization

model = models.resnet50(pretrained = True)
modelLayer = list(model.children())
Photo = r"butterfly.jpg"
visual = FeatureMapVisualization(modelLayer,isShow=True)
visual.visualization(Photo,k=0)
visual.visualizationEveryLayer(Photo,type='pdf')


# # classification
# Photo = get_image_info(Photo)
# with torch.no_grad():
#     for i ,layer in enumerate(visual.modelLayer):
#         try:
#             Photo = layer(Photo)
#         except:
#             Photo = Photo.view(Photo.shape[0],Photo.shape[1]*Photo.shape[2]*Photo.shape[3])
#             Photo = layer(Photo)
# print(torch.argmax(Photo))