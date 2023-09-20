## NumberOrdering Transformer
Ordering numbers with transformer

#Variable Settings

![variables_setting](https://github.com/baesh/NumberOrdering_Transformer/assets/18441461/3753be9d-8e54-43dc-9f2b-71cf56a122a4)

<br>
#Task

Given 10 random numbers in range 1~10, wish to sort them in ascending order by generating numbers with transformer.

ex)  10 3 5 1 1 5 2 2 9 2 --> 1 1 2 2 2 3 5 5 9 10

<br>
#Ideas

-Change numbers to array by one-hot encoding ( ex) 3 --> [0,0,1,0,0,0,0,0,0,0] )

-Do not use positional encoding since the input numbers' positions are not important

-learning rate: 0.001

-loss: sum of MSELoss

-optimizer: Adam

<br>
#Codes

-data generating: data.py (delete '#' for ussage)

-training: train.py (while training, saves the states of optimizer and layers in 'model state' folder, and save the intermediate results of accuracy and loss in 'accuracynloss_during_traing' folder)

-testing: test.py

-graphing loss and accuracy: graphing.py (show graphs of accuracy and loss which were calculated during training)

-etc: layer.py, functions.py ('layer.py' contains Transformer layers, 'functions.py' contains functions used in codes)

<br>
#Result

![Result](https://github.com/baesh/NumberOrdering_Transformer/assets/18441461/8ae46cac-604b-4b66-94da-ea0334a6b549)

-Accuracy(rate of correct generation) has reached to around 80 percent.

#etc
-unzip files in 'train_set' and 'test_set' for usage
