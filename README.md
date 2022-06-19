# GCC
Code for the paper "Geometry-guided Compact Compression for Light Field Image using Graph Convolutional Networks"
## train
'train.py' is used to train the GCC, it generate the pt model file.
## test
'test.py' is used to run the forward network of the model, which can reconstruct the image of the light field. 
## model
'model.py' is the concrete structure of the network.
## util
'util.py' contains a preprocessor function for the light field data set, graph modeling function, performance evaluation function.
## Dataset
The datasets used in the article is available from:

http://plenodb.jpeg.org/lf/epfl

https://lightfield-analysis.uni-konstanz.de/

## PS
This section of code was tested on ubuntu20.04 and RTX 2080 ti.

The HEVC compressed version is HM-16.20.

Select the DGL version corresponding to cuda.

Please change the file path of the dataset in code.
