#-------------------------------------
# Helper file to get Data store in Tensors
#-------------------------------------
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
from util.common_functions import plot_conv_weights, plot_conv_layer, display, readData
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
 To check the schema of a Model

 Parameters:
 -----------
 path :         String representing the model's path
 train_file:    String representing the training model data
------------

Returns:
----------
    Schema of Tensors
----------

"""

path = 'modelsBalanced/model1/'
train_file = '../signals_database/traffic-signs-data/train_4ProcessedBalanced.p'

IMAGE_TO_DISPLAY = 0
X_train, y_train = readData(train_file)
imagenconv = X_train[IMAGE_TO_DISPLAY]
#display(imagenconv, True, 32)
#"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(path + 'model-30960.meta')
    saver.restore(sess, tf.train.latest_checkpoint(path + '.'))
    print("Modelo restaurado", tf.train.latest_checkpoint(path + '.'))

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #print "tensors: ",len(all_vars)
    #print sess.run(all_vars[0])[4][4]
    #print sess.run(all_vars[8])[1023]

    #for ind in range(0, len(all_vars) ):
        #print(ind, all_vars[ind])
        #print(ind,sess.run(all_vars[ind]))

    #plot_conv_weights(sess.run(all_vars[0]),8,4,0)
    #plot_conv_weights(sess.run(all_vars[2]),8,8,0)
    #plot_conv_layer(sess.run(all_vars[2]),8,8)


#"""
"""0 <tf.Variable 'convolucion1/W:0' shape=(3, 3, 1, 32) dtype=float32_ref>
1 <tf.Variable 'convolucion1/B:0' shape=(32,) dtype=float32_ref>
2 <tf.Variable 'convolucion2/W:0' shape=(5, 5, 32, 64) dtype=float32_ref>
3 <tf.Variable 'convolucion2/B:0' shape=(64,) dtype=float32_ref>
4 <tf.Variable 'convolucion3/W:0' shape=(5, 5, 64, 128) dtype=float32_ref>
5 <tf.Variable 'convolucion3/B:0' shape=(128,) dtype=float32_ref>
6 <tf.Variable 'FC1/W:0' shape=(3584, 1024) dtype=float32_ref>
7 <tf.Variable 'FC1/B:0' shape=(1024,) dtype=float32_ref>
8 <tf.Variable 'FC2/W:0' shape=(1024, 43) dtype=float32_ref>
9 <tf.Variable 'FC2/B:0' shape=(43,) dtype=float32_ref>
10 <tf.Variable 'entrenamiento/iterac_entren:0' shape=() dtype=int32_ref>
11 <tf.Variable 'entrenamiento/beta1_power:0' shape=() dtype=float32_ref>
12 <tf.Variable 'entrenamiento/beta2_power:0' shape=() dtype=float32_ref>
13 <tf.Variable 'convolucion1/W/Adam:0' shape=(3, 3, 1, 32) dtype=float32_ref>
14 <tf.Variable 'convolucion1/W/Adam_1:0' shape=(3, 3, 1, 32) dtype=float32_ref>
15 <tf.Variable 'convolucion1/B/Adam:0' shape=(32,) dtype=float32_ref>
16 <tf.Variable 'convolucion1/B/Adam_1:0' shape=(32,) dtype=float32_ref>
17 <tf.Variable 'convolucion2/W/Adam:0' shape=(5, 5, 32, 64) dtype=float32_ref>
18 <tf.Variable 'convolucion2/W/Adam_1:0' shape=(5, 5, 32, 64) dtype=float32_ref>
19 <tf.Variable 'convolucion2/B/Adam:0' shape=(64,) dtype=float32_ref>
20 <tf.Variable 'convolucion2/B/Adam_1:0' shape=(64,) dtype=float32_ref>
21 <tf.Variable 'convolucion3/W/Adam:0' shape=(5, 5, 64, 128) dtype=float32_ref>
22 <tf.Variable 'convolucion3/W/Adam_1:0' shape=(5, 5, 64, 128) dtype=float32_ref>
23 <tf.Variable 'convolucion3/B/Adam:0' shape=(128,) dtype=float32_ref>
24 <tf.Variable 'convolucion3/B/Adam_1:0' shape=(128,) dtype=float32_ref>
25 <tf.Variable 'FC1/W/Adam:0' shape=(3584, 1024) dtype=float32_ref>
26 <tf.Variable 'FC1/W/Adam_1:0' shape=(3584, 1024) dtype=float32_ref>
27 <tf.Variable 'FC1/B/Adam:0' shape=(1024,) dtype=float32_ref>
28 <tf.Variable 'FC1/B/Adam_1:0' shape=(1024,) dtype=float32_ref>
29 <tf.Variable 'FC2/W/Adam:0' shape=(1024, 43) dtype=float32_ref>
30 <tf.Variable 'FC2/W/Adam_1:0' shape=(1024, 43) dtype=float32_ref>
31 <tf.Variable 'FC2/B/Adam:0' shape=(43,) dtype=float32_ref>
32 <tf.Variable 'FC2/B/Adam_1:0' shape=(43,) dtype=float32_ref>
"""