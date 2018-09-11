#agrego RELU en convoluciones
#batch size aumento'
# COn augment Data
import pickle
from pandas.io.parsers import read_csv
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import time
from datetime import timedelta
from matplotlib import pyplot as plt
from funcionesAuxiliares import readData, display, plot_confusion_matrix, plot_example_errors, plot_conv_layer
from subprocess import check_output
from sklearn.utils import shuffle

#pip install tqdm
from tqdm import tqdm
from tqdm import trange

import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#np.set_printoptions(threshold=np.nan)
from sklearn.model_selection import train_test_split
NOMBRE_MODELO = 'model'
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'
NOMBRE_TENSOR_SALIDA_CALCULADA = 'outputYCalculada'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"

ADAM_OPTIMIZER = True
#ADAM_OPTIMIZER = False #Activate when LEARNING_DECAY = True
LEARNING_DECAY = False
#tensorboard --logdir modelsBalanced/model1/

#--------------FOR BALANCED DATASET-----------------------
#"""
#rutaDeModelo = 'modelsBalanced/model1/'  
rutaDeModelo = 'modelsBalanced/model2/'  

TASA_APRENDIZAJE = 5e-4

NUM_CLASSES = 0  #43
NUM_TRAIN = 0  #270900 *0.75 = 203175
NUM_TEST = 0  #270900 *0.25 = 67725
IMAGE_SHAPE = 0  #(32,32,1)

BATCH_SIZE = 525
ITER_PER_EPOCA = 387  # = (203175 / 525)

#ITERACIONES_ENTRENAMIENTO: (ITER_PER_EPOCA * EPOCAS)
ITERACIONES_ENTRENAMIENTO = ITER_PER_EPOCA * 10

CHKP_GUARDAR_MODELO = ITER_PER_EPOCA
CHKP_REVISAR_PROGRESO = 50
#"""
#-----------------------------------------------------------
"""#--------------FOR 10 TIMES DATASET-----------------------
rutaDeModelo = 'models10extend/' #with same kernels size

TASA_APRENDIZAJE = 5e-4 

NUM_CLASSES = 0  #43
NUM_TRAIN = 0  #698918 *0.75 = 524188
NUM_TEST = 0  #698918 *0.25 = 174730
IMAGE_SHAPE = 0  #(32,32,1)

BATCH_SIZE = 679
ITER_PER_EPOCA = 772  # = (524188 / 679)

#ITERACIONES_ENTRENAMIENTO: (ITER_PER_EPOCA * EPOCAS)
ITERACIONES_ENTRENAMIENTO = ITER_PER_EPOCA * 10

CHKP_GUARDAR_MODELO = ITER_PER_EPOCA
CHKP_REVISAR_PROGRESO = 100

#-----------------------------------------------------------
"""

def siguiente_batch_entren(batch_size, cant_imag_entrenamiento):

    #NUM_TRAIN = 39209
    global imagenes_entrenam
    global clases_entrenam
    global clases_entrenam_flat
    global indice_en_epoca
    global epocas_completadas
    global TASA_APRENDIZAJE
    comienzo = indice_en_epoca
    indice_en_epoca += batch_size

    # Cuando ya se han utilizado todos los datos de entrenamiento, se reordena aleatoriamente.
    if indice_en_epoca > cant_imag_entrenamiento:
        # epoca finalizada
        #TASA_APRENDIZAJE = 1e-3
        epocas_completadas += 1
        # barajear los datos
        perm = np.arange(cant_imag_entrenamiento)
        np.random.shuffle(perm)
        #perm = stratified_shuffle(clases_entrenam_flat, 10)
        imagenes_entrenam = imagenes_entrenam[perm]
        clases_entrenam = clases_entrenam[perm]
        clases_entrenam_flat = clases_entrenam_flat[perm]
        # comenzar nueva epoca
        comienzo = 0
        indice_en_epoca = batch_size
        assert batch_size <= cant_imag_entrenamiento
    end = indice_en_epoca
    return imagenes_entrenam[comienzo:end], clases_entrenam[comienzo:end]


def one_hot(X, y):

    #Organizar las clases de las imagenes en un solo vector
    y_flatten = y
    # convertir tipo de clases de escalares a vectores de activacion de 1s
    # 0 => [1 0 0 0 0 0 0 0 0 0....0 0 0 0]
    # 1 => [0 1 0 0 0 0 0 0 0 0....0 0 0 0]
    # ...
    # 43 => [0 0 0 0 0 0 0 0 0 0....0 0 0 1]
    y_onehot = np.zeros((y.shape[0], NUM_CLASSES))
    for i, onehot_label in enumerate(y_onehot):
        onehot_label[y[i]] = 1.0
    y = y_onehot

    return X, y, y_flatten


def inicializar_pesos(shape):
    initial = tf.truncated_normal(shape, stddev=0.05)
    w = tf.Variable(initial, name="W")
    tf.summary.histogram("pesos", w)
    return w


def inicializar_bias(shape):
    initial = tf.constant(0.05, shape=shape)
    b = tf.Variable(initial, name="B")
    tf.summary.histogram("biases", b)
    return b


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    #print("Layer shape: ", layer_shape)
    num_features = layer_shape[1:4].num_elements()
    # The number of features is: img_height * img_width * num_channels
    layer_flat = tf.reshape(layer, [-1, num_features])
    #print("layer_flat: : ", layer_flat)

    return layer_flat, num_features


def conv_layer(nombre, entrada, num_inp_channels, filter_size, num_filters,
               use_pooling):
    with tf.name_scope(nombre):
        forma = [filter_size, filter_size, num_inp_channels, num_filters]
        pesos = inicializar_pesos(shape=forma)
        biases = inicializar_bias([num_filters])
        convolucion = tf.nn.conv2d(
            input=entrada, filter=pesos, strides=[1, 1, 1, 1], padding='SAME')

        convolucion += biases
        print( nombre,"conv: ",convolucion.get_shape(),"\n***********")
        #puede ser despues de pooling
        convolucion = tf.nn.relu(convolucion)

        if use_pooling:
            convolucion = doPool(convolucion,2)
        print( nombre,"after pooling: ",convolucion.get_shape(),"\n***********\n")
    return convolucion, pesos


def doPool(convolucion, size):
    return tf.nn.max_pool(
        value=convolucion,
        ksize=[1, size, size, 1],
        strides=[1, size, size, 1],
        padding='VALID')


def capa_fc(nombre, entrada, num_inputs, num_outputs, use_relu=True):
    with tf.name_scope(nombre):
        #print( entrada.get_shape()," multiplica ",num_inputs,num_outputs)
        pesos = inicializar_pesos(shape=[num_inputs, num_outputs])
        biases = inicializar_bias([num_outputs])
        layer = tf.matmul(entrada, pesos) + biases

        if use_relu:
            "Uso relu"
            layer = tf.nn.relu(layer)
        else:
            "No uso Relu"

    return layer, pesos


def printArquitecture(tam_filtro1,tam_filtro2,tam_filtro3,num_filtro1,num_filtro2,num_filtro3,dropout_conv1,dropout_conv2,dropout_conv3,fc1_inputs,fc2_inputs,dropout_fc1, L2_lambda):
    print("=================== DATA ====================")
    print("            Training set: {} examples".format(NUM_TRAIN))
    print("          Validation set: {} examples".format(NUM_TEST))
    print("              Batch size: {}".format(BATCH_SIZE))
    print("       Number of classes: {}".format(NUM_CLASSES)) 
    print("        Image data shape: {}".format(IMAGE_SHAPE))

    print("=================== MODEL ===================")
    print("--------------- ARCHITECTURE ----------------")  
    print(" %-*s %-*s %-*s %-*s" % (10, "", 10, "Type", 8, "Size", 15, "Dropout (keep p)"))    
    print(" %-*s %-*s %-*s %-*s" % (10, "Layer 1", 10, "{}x{} Conv".format(tam_filtro1, tam_filtro1), 8, str(num_filtro1), 15, str(dropout_conv1)))    
    print(" %-*s %-*s %-*s %-*s" % (10, "Layer 2", 10, "{}x{} Conv".format(tam_filtro2, tam_filtro2), 8, str(num_filtro2), 15, str(dropout_conv2)))    
    print(" %-*s %-*s %-*s %-*s" % (10, "Layer 3", 10, "{}x{} Conv".format(tam_filtro3, tam_filtro3), 8, str(num_filtro3), 15, str(dropout_conv3)))    
    print(" %-*s %-*s %-*s %-*s" % (10, "Layer 4", 10, "FC", 8, str(fc1_inputs), 15, str(dropout_fc1)))
    print(" %-*s %-*s %-*s %-*s" % (10, "Layer 5", 10, "FC", 8, str(fc2_inputs), 15, str("--")))    
    print("---------------- PARAMETERS -----------------")
    print("     Learning rate decay: " + ("Enabled" if LEARNING_DECAY else "Disabled"))
    print("           Learning rate: {}".format(TASA_APRENDIZAJE))
    print("               OPTIMIZER: " + ("ADAM Optimizer" if ADAM_OPTIMIZER else "Gradient Descent Optimizer"))
    print("       L2 Regularization: " + ("Enabled (L2 Lambda = {})\n\n".format(L2_lambda)))


def create_cnn():
    #Los tensores placeholder sirven como entrada al grafico computacional de TensorFlow que podemos cambiar cada vez que ejecutamos el grafico.
    #None significa que el tensor puede contener un numero arbitrario de imagenes , donde cada imagen es un vector de longitud dada

    entradas = tf.placeholder(
        'float',
        shape=[None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]],
        name=NOMBRE_TENSOR_ENTRADA)
    print("Numero clases", NUM_CLASSES)
    # clases
    y_deseada = tf.placeholder(
        'float', shape=[None, NUM_CLASSES], name=NOMBRE_TENSOR_SALIDA_DESEADA)

    #print( "entradas shape ",entradas.get_shape() # =>(?, 32, 32, 1))

    #Aplicarmos dropout entre la capa FC y la capa de salida
    #keep_prob = tf.placeholder('float',name=NOMBRE_PROBABILIDAD)
    is_training = tf.placeholder(tf.bool, name=NOMBRE_PROBABILIDAD)

    tam_filtro1 = 3
    num_filtro1 = 32
    dropout_conv1 = 0.8
    capa_conv1, pesos_conv1 = conv_layer(
        nombre="convolucion1",
        entrada=entradas,
        num_inp_channels=IMAGE_SHAPE[2],
        filter_size=tam_filtro1,
        num_filters=num_filtro1,
        use_pooling=True)
    capa_conv1_drop = tf.cond(is_training,
                              lambda: tf.nn.dropout(capa_conv1, keep_prob=dropout_conv1),
                              lambda: capa_conv1)

    tam_filtro2 = 5
    num_filtro2 = 64
    dropout_conv2 = 0.7
    capa_conv2, pesos_conv2 = conv_layer(
        nombre="convolucion2",
        entrada=capa_conv1_drop,
        num_inp_channels=num_filtro1,
        filter_size=tam_filtro2,
        num_filters=num_filtro2,
        use_pooling=True)
    capa_conv2_drop = tf.cond(is_training,
                              lambda: tf.nn.dropout(capa_conv2, keep_prob=dropout_conv2),
                              lambda: capa_conv2)

    tam_filtro3 = 5
    num_filtro3 = 128
    dropout_conv3 = 0.6
    capa_conv3, pesos_conv3 = conv_layer(
        nombre="convolucion3",
        entrada=capa_conv2_drop,
        num_inp_channels=num_filtro2,
        filter_size=tam_filtro3,
        num_filters=num_filtro3,
        use_pooling=True)
    capa_conv3_drop = tf.cond(is_training,
                              lambda: tf.nn.dropout(capa_conv3, keep_prob=dropout_conv3),
                              lambda: capa_conv3)

    capa_conv1_drop = doPool(capa_conv1_drop, size=4)
    print( "after pooling:capa_conv1_drop: ",capa_conv1_drop.get_shape(),"\n***********\n")
    capa_conv2_drop = doPool(capa_conv2_drop, size=2)
    print("after pooling:capa_conv2_drop: ",capa_conv2_drop.get_shape(),"\n***********\n")

    layer_flat1, num_fc_layers1 = flatten_layer(capa_conv1_drop)
    layer_flat2, num_fc_layers2 = flatten_layer(capa_conv2_drop)
    layer_flat3, num_fc_layers3 = flatten_layer(capa_conv3_drop)
    print("NUM FC LAYER:",num_fc_layers1,num_fc_layers2,num_fc_layers3 )
    
    layer_flat = tf.concat([layer_flat1, layer_flat2, layer_flat3],1)  #(?, 7168)
    print( "1:",layer_flat.get_shape())
    num_fc_layers = num_fc_layers1 + num_fc_layers2 + num_fc_layers3  # (7168)
    print( "2:",num_fc_layers)
    #"""

    #---------------------------Capa totalmente conectada--------------------------------------------
    #The previous layer.,  Num. inputs from prev. layer. , Num. outputs.
    fc1_inputs = num_fc_layers
    fc1_outputs = 1024
    dropout_fc1 = 0.5
    capa_fc1, pesos_fc1 = capa_fc(
        nombre="FC1",
        entrada=layer_flat,
        num_inputs=fc1_inputs,
        num_outputs=fc1_outputs,
        use_relu=True)

    fc_capa1_drop = tf.cond(is_training,
                            lambda: tf.nn.dropout(capa_fc1, keep_prob=dropout_fc1),
                            lambda: capa_fc1)

    fc2_inputs = 1024
    fc2_outputs = NUM_CLASSES
    capa_fc2, pesos_fc2 = capa_fc(
        nombre="FC2",
        entrada=fc_capa1_drop,
        num_inputs=fc2_inputs,
        num_outputs=fc2_outputs,
        use_relu=False)
    #y_calculada = capa_fc2
    y_calculada = tf.nn.softmax(capa_fc2, name=NOMBRE_TENSOR_SALIDA_CALCULADA)
    predictor = tf.argmax(y_calculada, axis=1)

    tf.add_to_collection("predictor", predictor)
    tf.add_to_collection("acuracia", y_calculada)
    tf.summary.histogram('activations', y_calculada)

    #--------------------------------------------------------------------------
    with tf.name_scope("Regular_Loss"):
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=capa_fc2, labels=y_deseada)
        
        cross_entropy_mean = tf.reduce_mean(softmax_cross_entropy, name="error")
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
    
    with tf.name_scope("L2_Regularization_Method"):
        regularizers = (tf.nn.l2_loss(pesos_conv1) + tf.nn.l2_loss(pesos_conv2) +
                        tf.nn.l2_loss(pesos_conv3) + tf.nn.l2_loss(pesos_fc1) +
                        tf.nn.l2_loss(pesos_fc2))
        L2_lambda = 0.0001
        error =  cross_entropy_mean + L2_lambda * regularizers
        tf.summary.scalar('error_Acc', error)
    #---------------------------------------------------------------------------

    with tf.name_scope("entrenamiento"):
        #Funcion de optimizacion
        iterac_entren = tf.Variable(0, name='iterac_entren', trainable=False)
        #A las 15 iteraciones se cambio el (iterperepoca *5) -> (iterperepoca)
        if LEARNING_DECAY:
            lr = tf.train.exponential_decay(TASA_APRENDIZAJE, iterac_entren, ITER_PER_EPOCA, 0.99, staircase=True, name='ExponentialDecay')
            tf.summary.scalar('learning_rate_decay',lr)
        else:
            lr = TASA_APRENDIZAJE
            tf.summary.scalar('learning_rate',lr)
        #optimizador = tf.train.AdamOptimizer(TASA_APRENDIZAJE).minimize(error, global_step=iterac_entren)
        if ADAM_OPTIMIZER:
            optimizador = tf.train.AdamOptimizer(lr).minimize(error, global_step=iterac_entren)
        else:
            optimizador = tf.train.GradientDescentOptimizer(lr).minimize(error, global_step=iterac_entren)
    
    with tf.name_scope("Evaluacion"):
        # evaluacion
        prediccion_correcta = tf.equal(
            tf.argmax(y_calculada, 1), tf.argmax(y_deseada, 1))
        acierto = tf.reduce_mean(tf.cast(prediccion_correcta, 'float'))
        tf.add_to_collection("mean_acc", acierto)
        tf.summary.scalar("acierto", acierto)

    resumen = tf.summary.merge_all()
    
    printArquitecture(tam_filtro1,tam_filtro2,tam_filtro3,num_filtro1,num_filtro2,num_filtro3,dropout_conv1,dropout_conv2,dropout_conv3,fc1_inputs,fc2_inputs,dropout_fc1, L2_lambda)

    return resumen, entradas, y_deseada, is_training, iterac_entren, optimizador, acierto, error, predictor

#===============================================================================================
#===============================================================================================

def print_write_validationSet(sess, resumen, acierto, error, feed_dictx, evalua_writer, i ):
    [resu, aciertos_eval, er] = sess.run([resumen, acierto, error], feed_dict=feed_dictx)
    evalua_writer.add_summary(resu, i)
    print('En la iteracion %d:'%(i+1))
    print(('Acc. Valid Set => %.4f' %(aciertos_eval)))
    print(('Error Valid Set => %.4f \n' %(er)))

def print_write_trainSet(sess, resumen, acierto, error, feed_dictx, i ):
    [resu, aciertos_train, er] = sess.run([resumen, acierto, error], feed_dict=feed_dictx)
    print('En la iteracion %d:'%(i+1))
    print(('Acc. Train Set => %.4f ' % (aciertos_train)))
    print(('Error Train Set => %.4f \n' % (er)))

def saveModel(sess, i , rutaDeModelo, saver):
    print(('Guardando modelo en %d iteraciones....' % (i + 1)))
    saver.save(
        sess,
        rutaDeModelo + NOMBRE_MODELO,
        global_step=i + 1,
        write_meta_graph=True)

if __name__ == "__main__":

    train_file = '../signals_database/traffic-signs-data/train_4ProcessedBalanced.p'
    #train_file = '../signals_database/traffic-signs-data/train_4Processed10.p'
    signnames = read_csv(
        "../signals_database/traffic-signs-data/signnames.csv").values[:, 1]

    print("reading trianing dataset")
    X_train, y_train = readData(train_file)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.25)

    #print( X_train[0])
    NUM_TRAIN = y_train.shape[0]
    NUM_TEST = y_validation.shape[0]
    IMAGE_SHAPE = X_train[0].shape
    NUM_CLASSES = len(set(y_train))


    class_indices, examples_per_class, class_counts = np.unique(
        y_train, return_index=True, return_counts=True)
    class_indicesTest, examples_per_classTest, class_countsTest = np.unique(
        y_validation, return_index=True, return_counts=True)
    #plot_histograms('Class Distribution across Training Data', class_indices, class_counts)
    #plot_histograms('Class Distribution across Testing Data', class_indicesTest, class_countsTest)
    #plt.show()

    imagenes_entrenam, clases_entrenam, clases_entrenam_flat = one_hot(
        X_train, y_train)
    imagenes_eval, clases_eval, clases_eval_flat = one_hot(
        X_validation, y_validation)

    #cv2.imwrite('zexample1.png',X_train[200])
    #display(X_train[10])
    #print( X_train[10])
    #print( "**********************")
    #print( X_validation[0])
    
    #--------------------------------CREACION DE LA RED------------------------------------------------
    #"""
    print("Inicio de creacion de la red")
    tf.reset_default_graph()
    sess = tf.Session()
    resumen, entradas, y_deseada, is_training, iterac_entren, optimizador, acierto, error, predictor = create_cnn(
    )

    sess.run(tf.global_variables_initializer())

    entren_writer = tf.summary.FileWriter(rutaDeModelo + 'entrenamiento',
                                          sess.graph)
    #entren_writer.add_graph(sess.graph)
    evalua_writer = tf.summary.FileWriter(rutaDeModelo + 'evaluacion',
                                          sess.graph)

    saver = tf.train.Saver(max_to_keep=None)
    ckpt = tf.train.get_checkpoint_state(rutaDeModelo + '.')

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(("Sesion restaurada de: %s" % ckpt.model_checkpoint_path))

    else:
        print(("No se encontro puntos de control."))

    ultima_iteracion = iterac_entren.eval(sess)
    print("Ultimo modelo en la iteracion: ", ultima_iteracion)

    epocas_completadas = 0
    indice_en_epoca = 0

    clases_calc = np.zeros(shape=NUM_TEST, dtype=np.int)

    comienzo_time = time.time()
    #"""
    #Desde la ultima iteracion hasta el ITERACIONES_ENTRENAMIENTO dado
    for i in trange(ultima_iteracion, ITERACIONES_ENTRENAMIENTO):
        #Obtener nuevo subconjunto(batch) de (BATCH_SIZE =100) imagenes
        batch_img_entrada, batch_img_clase = siguiente_batch_entren(
            BATCH_SIZE, NUM_TRAIN)
        # Entrenar el batch
        [resu, _] = sess.run(
            [resumen, optimizador],
            feed_dict={
                entradas: batch_img_entrada,
                y_deseada: batch_img_clase,
                is_training: True
            })

        entren_writer.add_summary(resu, i)

        # Observar el progreso cada 'CHKP_REVISAR_PROGRESO' iteraciones
        if (i + 1) % CHKP_REVISAR_PROGRESO == 0:
            print(
                ('Time in %d iteraciones: %s ' %
                 (i + 1 - ultima_iteracion,
                  str( timedelta( seconds=int(round(time.time() - comienzo_time)))))))
            #--------------------------------------------------------------
            feed_dictx = {
                entradas: batch_img_entrada,
                y_deseada: batch_img_clase,
                is_training: False#dont use dropout in Testing
            }
            
            print_write_trainSet(sess, resumen, acierto, error, feed_dictx, i )
            #--------------------------------------------------------------
            imagenes_eval, clases_eval = shuffle(imagenes_eval, clases_eval)
            feed_dictx = {
                entradas: imagenes_eval[:2500],
                y_deseada: clases_eval[:2500],
                is_training: False#dont use dropout in Testing
            }
            print_write_validationSet(sess, resumen, acierto, error, feed_dictx, evalua_writer, i)
            #--------------------------------------------------------------

        #Crear 'punto de control' cuando se llego a las CHKP_GUARDAR_MODELO iteraciones
        if (i + 1) % CHKP_GUARDAR_MODELO == 0:
            saveModel(sess, i , rutaDeModelo, saver)
    #--------------------------------------------------------------
    feed_dictx = {
        entradas: batch_img_entrada,
        y_deseada: batch_img_clase,
        is_training: False#dont use dropout in Testing
    }
    print_write_trainSet(sess, resumen, acierto, error, feed_dictx, i )
    #--------------------------------------------------------------
    imagenes_eval, clases_eval = shuffle(imagenes_eval, clases_eval)
    feed_dictx = {
        entradas: imagenes_eval[:2500],
        y_deseada: clases_eval[:2500],
        is_training: False#dont use dropout in Testing
    }
    print_write_validationSet(sess, resumen, acierto, error, feed_dictx, evalua_writer, i)
    #--------------------------------------------------------------

    saveModel(sess, i , rutaDeModelo, saver)

    #Fin del proceso de entrenamiento y validacion del modelo
    end_time = time.time()
    # Cuanto tiempo tomo el entrenamiento
    time_dif = end_time - comienzo_time

    # Imprimir tiempo
    print(('Tiempo usado en %d iteraciones: %s ' %
           (ITERACIONES_ENTRENAMIENTO - ultima_iteracion,
            str(timedelta(seconds=int(round(time_dif)))))))

    #"""
