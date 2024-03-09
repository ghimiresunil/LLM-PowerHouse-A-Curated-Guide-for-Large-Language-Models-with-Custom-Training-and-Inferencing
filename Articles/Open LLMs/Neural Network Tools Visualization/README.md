# Tools to Design or Visualize Architecture of Neural Network

1. [Net2Vis](https://viscom.net2vis.uni-ulm.de/OG1Br2BAkYSwwrV6CADl4X5EfErFjUzvuUwXWDdLbdsIXNhb9L): Net2Vis automatically generates abstract visualizations for convolutional neural networks from Keras code.

2. [visualkeras](https://github.com/paulgavrikov/visualkeras/): Visualkeras is a Python package to help visualize Keras (either standalone or included in tensorflow) neural network architectures. It allows easy styling to fit most needs. As of now it supports layered style architecture generation which is great for CNNs (Convolutional Neural Networks) and a grap style architecture.
```python
import visualkeras

model = ...

visualkeras.layered_view(model).show() # display using your system viewer
visualkeras.layered_view(model, to_file='output.png') # write to disk
visualkeras.layered_view(model, to_file='output.png').show() # write and show

visualkeras.layered_view(model)
```

3. [draw_convnet](https://github.com/gwding/draw_convnet): Python script for illustrating Convolutional Neural Network (ConvNet)

4. [NNSVG](https://alexlenail.me/NN-SVG/LeNet.html)

5. [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet): Latex code for drawing neural networks for reports and presentation. Have a look into examples to see how they are made. Additionally, lets consolidate any improvements that you make and fix any bugs to help more people with this code.

6. [Tensorboard](https://www.tensorflow.org/tensorboard/graphs): TensorBoard’s Graphs dashboard is a powerful tool for examining your TensorFlow model.

7. [Caffe](https://github.com/BVLC/caffe/blob/master/python/caffe/draw.py): In Caffe you can use [caffe/draw.py](https://github.com/BVLC/caffe/blob/master/python/caffe/draw.py) to draw the NetParameter protobuffer:

8. [Matlab](https://www.mathworks.com/help/deeplearning/ref/view.html;jsessionid=1a23781fbbc7052874fa6a04d3c3)

9. [Keras.js](https://transcranial.github.io/keras-js/#/inception-v3)

10. [keras-sequential-ascii](https://github.com/stared/keras-sequential-ascii/): A library for Keras for investigating architectures and parameters of sequential models.

### VGG 16 Architecture
```code
           OPERATION           DATA DIMENSIONS   WEIGHTS(N)   WEIGHTS(%)

              Input   #####      3  224  224
         InputLayer     |   -------------------         0     0.0%
                      #####      3  224  224
      Convolution2D    \|/  -------------------      1792     0.0%
               relu   #####     64  224  224
      Convolution2D    \|/  -------------------     36928     0.0%
               relu   #####     64  224  224
       MaxPooling2D   Y max -------------------         0     0.0%
                      #####     64  112  112
      Convolution2D    \|/  -------------------     73856     0.1%
               relu   #####    128  112  112
      Convolution2D    \|/  -------------------    147584     0.1%
               relu   #####    128  112  112
       MaxPooling2D   Y max -------------------         0     0.0%
                      #####    128   56   56
      Convolution2D    \|/  -------------------    295168     0.2%
               relu   #####    256   56   56
      Convolution2D    \|/  -------------------    590080     0.4%
               relu   #####    256   56   56
      Convolution2D    \|/  -------------------    590080     0.4%
               relu   #####    256   56   56
       MaxPooling2D   Y max -------------------         0     0.0%
                      #####    256   28   28
      Convolution2D    \|/  -------------------   1180160     0.9%
               relu   #####    512   28   28
      Convolution2D    \|/  -------------------   2359808     1.7%
               relu   #####    512   28   28
      Convolution2D    \|/  -------------------   2359808     1.7%
               relu   #####    512   28   28
       MaxPooling2D   Y max -------------------         0     0.0%
                      #####    512   14   14
      Convolution2D    \|/  -------------------   2359808     1.7%
               relu   #####    512   14   14
      Convolution2D    \|/  -------------------   2359808     1.7%
               relu   #####    512   14   14
      Convolution2D    \|/  -------------------   2359808     1.7%
               relu   #####    512   14   14
       MaxPooling2D   Y max -------------------         0     0.0%
                      #####    512    7    7
            Flatten   ||||| -------------------         0     0.0%
                      #####       25088
              Dense   XXXXX ------------------- 102764544    74.3%
               relu   #####        4096
              Dense   XXXXX -------------------  16781312    12.1%
               relu   #####        4096
              Dense   XXXXX -------------------   4097000     3.0%
            softmax   #####        1000
```

11. [Netron](https://github.com/lutzroeder/Netron)
12. [DotNet](https://github.com/martisak/dotnets)
13. [Graphviz : Tutorial](https://tgmstat.wordpress.com/2013/06/12/draw-neural-network-diagrams-graphviz/)