# FeatureMapVisualization
This package provides a class that can be used to visualize feature maps at various layers of a deep learning network.

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
