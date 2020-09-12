# My-First-Convolutional-Neural-Network

Download the Dataset from Tensorflow Datasets : [https://www.tensorflow.org/datasets/catalog/rock_paper_scissors]
HOMEPAGE : [http://www.laurencemoroney.com/rock-paper-scissors-dataset/]

## My Model Summary :

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 75, 75, 64)        1792      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 37, 37, 64)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 35, 35, 128)       73856     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
dropout (Dropout)            (None, 7, 7, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 6272)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               1605888   
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 771       
=================================================================
Total params: 1,829,891
Trainable params: 1,829,891
Non-trainable params: 0
_________________________________________________________________
