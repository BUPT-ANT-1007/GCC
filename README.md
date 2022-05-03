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
## PS
This section of code was tested on ubuntu20.04 and RTX 2080 ti.
Please change the file path of the dataset in code.
