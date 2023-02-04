# **FeatureMapVisualization**

The same class, FeatureMapVisualization, is available in both python and matlab versions. This class provides a way to visualize the feature maps of deep learning networks.

More details (introduction of parameters, etc.) are described in the class documentation.

---------------------

## Python

This package provides a class that can be used to visualize feature maps at various layers of a deep learning network.

How to use the FeatureMapVisualization class:

Initialization flattens the network structure and stores the layers sequentially in a list.

```python
import FeatureMapVisualization as Fmv

model = models.alexnet(pretrained=True)

modelLayer = list(model.children())

modelVisualization = Fmv.FeatureMapVisualization(modelLayer)
```

----------------------------
Provides functions and methods:

This method displays the layers of the flattened network:

```python
modelVisualization.showStructure()
```


This method shows the visualization of the feature map of the specified layer:

```python
modelVisualization.visualization(input,k=0,type='pdf',resolution=(256,256))
```


This method shows the visualization of all layer feature maps:

```python
modelVisualization.visualizationEveryLayer(input,type='pdf',resolution=(256,256))
```

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Matlab

How to use the FeatureMapVisualization class:

Feed in a model for instantiation and initialization.

~~~matlab
a = alexnet;
Fmv = FeatureMapVisualization(a,isShow=true);
~~~

------

Provides functions and methods:

This method shows the visualization of the feature map of the specified layer:

~~~matlab
Fmv.visualization(input,k=2,cmap=hsv,resolution=[256,256]);
~~~

This method shows the visualization of all layer feature maps:

~~~matlab
Fmv.visualizationEveryLayer(input,cmap=gray,isSave=false);
~~~
