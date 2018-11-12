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
NOMBRE_MODELO = 'model'
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'
NOMBRE_TENSOR_SALIDA_CALCULADA = 'outputYCalculada'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"


#tensorboard --logdir models_Peru/model1/

#--------------FOR BALANCED DATASET-----------------------
#"""
rutaDeModelo = 'models_Peru/model1/'
TASA_APRENDIZAJE = 5e-4

NUM_CLASSES = 0  #7
NUM_TRAIN = 0  #31314 *0.75 = 23485
NUM_VALID = 0  #31314 *0.10 = 3131 --- Test: 4698 (15%)
IMAGE_SHAPE = 0  #(60,60,1)

BATCH_SIZE = 305
ITER_PER_EPOCA = 77  # = (23485 / 427)

EPOCS = 100
#ITERACIONES_ENTRENAMIENTO: (ITER_PER_EPOCA * EPOCS)
ITERACIONES_ENTRENAMIENTO = ITER_PER_EPOCA * EPOCS

CHKP_GUARDAR_MODELO = ITER_PER_EPOCA * 5 # normal is 5
CHKP_REVISAR_PROGRESO = 77

LEARNING_DECAY = True
L2_REG = True
L2_lambda = 0.0001
DECAY_RATE = 0.99#(the smaller the lower to learn)
fourLayers = False
#"""
#-----------------------------------------------------------

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
    print("Layer shape: ", layer_shape)
    num_features = layer_shape[1:4].num_elements()
    # The number of features is: img_height * img_width * num_channels
    print("num_features: ", num_features)
    layer_flat = tf.reshape(layer, [-1, num_features])
    print("layer_flat: : ", layer_flat)
    print("=======================")
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
        #puede ser despues de pooling
        convolucion = tf.nn.relu(convolucion)

        if use_pooling:
            convolucionPool = doPool(convolucion,2)
            print( nombre,"after pooling: ",convolucion.get_shape(),"\n***********\n")
    return convolucionPool, pesos, convolucion


def doPool(convolucion, size):
    return tf.nn.max_pool(
        value=convolucion,
        ksize=[1, size, size, 1],
        strides=[1, size, size, 1],
        padding='VALID')


def capa_fc(nombre, entrada, num_inputs, num_outputs, use_relu=True):
    with tf.name_scope(nombre):
        print( entrada.get_shape()," multiplica ",num_inputs,num_outputs)
        pesos = inicializar_pesos(shape=[num_inputs, num_outputs])
        biases = inicializar_bias([num_outputs])
        layer = tf.matmul(entrada, pesos) + biases

        if use_relu:
            "Uso relu"
            layer = tf.nn.relu(layer)
        else:
            "No uso Relu"

    return layer, pesos


def printArquitecture(tam_filtro,num_filtro,dropout_conv,fc1_inputs,fc2_inputs,dropout_fc1):
    print("=================== DATA per {} epocs ====================".format(EPOCS))
    print("            Training set: {} examples".format(NUM_TRAIN))
    print("          Validation set: {} examples".format(NUM_VALID))
    print("              Batch size: {}".format(BATCH_SIZE))
    print("       Number of classes: {}".format(NUM_CLASSES))
    print("        Image data shape: {}".format(IMAGE_SHAPE))

    print("=================== MODEL ===================")
    print("--------------- ARCHITECTURE ----------------")
    print(" %-*s %-*s %-*s %-*s" % (10, "", 10, "Type", 8, "Size", 15, "Dropout (keep p)"))
    for i in range(0, len(tam_filtro)):
        print(" %-*s %-*s %-*s %-*s" % (10, "Layer "+str(i+1), 10, str(tam_filtro[i])+"x"+str(tam_filtro[i])+ "Conv", 8, str(num_filtro[i]), 15, str(dropout_conv[i])))
    print(" %-*s %-*s %-*s %-*s" % (10, "Layer 4", 10, "FC", 8, str(fc1_inputs), 15, str(dropout_fc1)))
    print(" %-*s %-*s %-*s %-*s" % (10, "Layer 5", 10, "FC", 8, str(fc2_inputs), 15, str("--")))
    print("---------------- PARAMETERS -----------------")
    if(LEARNING_DECAY):
        print("       Learning rate decay: " + ("Enabled (Decay rate = {})".format(DECAY_RATE)))
    else:
        print("       Learning rate decay: Disabled")
    print("           Learning rate: {}".format(TASA_APRENDIZAJE))
    print("               OPTIMIZER: ADAM Optimizer")
    if(L2_REG):
        print("       L2 Regularization: " + ("Enabled (L2 Lambda = {})\n\n".format(L2_lambda)))
    else:
        print("       L2 Regularization: Disabled \n\n")


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

    print( "entradas shape ",entradas.get_shape()) # =>(?, 60, 60, 1))

    #Aplicarmos dropout entre la capa FC y la capa de salida
    #keep_prob = tf.placeholder('float',name=NOMBRE_PROBABILIDAD)
    is_training = tf.placeholder(tf.bool, name=NOMBRE_PROBABILIDAD)

    tam_filtro1 = 3
    num_filtro1 = 32
    dropout_conv1 = 0.8
    capa_conv1, pesos_conv1,nopool1 = conv_layer(
        nombre="convolucion1",
        entrada=entradas,
        num_inp_channels=IMAGE_SHAPE[2],
        filter_size=tam_filtro1,
        num_filters=num_filtro1,
        use_pooling=True)
    capa_conv1_drop = tf.cond(is_training,
                              lambda: tf.nn.dropout(capa_conv1, keep_prob=dropout_conv1),
                              lambda: capa_conv1)

    print("capa_conv1 ",capa_conv1)

    tam_filtro2 = 5
    num_filtro2 = 64
    dropout_conv2 = 0.7
    capa_conv2, pesos_conv2,nopool2 = conv_layer(
        nombre="convolucion2",
        entrada=capa_conv1_drop,
        num_inp_channels=num_filtro1,
        filter_size=tam_filtro2,
        num_filters=num_filtro2,
        use_pooling=True)
    capa_conv2_drop = tf.cond(is_training,
                              lambda: tf.nn.dropout(capa_conv2, keep_prob=dropout_conv2),
                              lambda: capa_conv2)

    print("capa_conv2 ",capa_conv2)

    tam_filtro3 = 5
    num_filtro3 = 128
    dropout_conv3 = 0.6
    capa_conv3, pesos_conv3,nopool3 = conv_layer(
        nombre="convolucion3",
        entrada=capa_conv2_drop,
        num_inp_channels=num_filtro2,
        filter_size=tam_filtro3,
        num_filters=num_filtro3,
        use_pooling=True)
    capa_conv3_drop = tf.cond(is_training,
                              lambda: tf.nn.dropout(capa_conv3, keep_prob=dropout_conv3),
                              lambda: capa_conv3)
    print("capa_conv3 ",capa_conv3)
    if(fourLayers):
        tam_filtro4 = 7
        num_filtro4 = 128
        dropout_conv4 = 0.6
        capa_conv4, pesos_conv4,nopool4 = conv_layer(
            nombre="convolucion4",
            entrada=capa_conv3_drop,
            num_inp_channels=num_filtro3,
            filter_size=tam_filtro4,
            num_filters=num_filtro4,
            use_pooling=True)
        capa_conv4_drop = tf.cond(is_training,
                                lambda: tf.nn.dropout(capa_conv4, keep_prob=dropout_conv4),
                                lambda: capa_conv4)
        print("capa_conv4 ",capa_conv4)

    if(fourLayers):
        capa_conv1_drop = doPool(capa_conv1_drop, size=8)
    else:
        capa_conv1_drop = doPool(capa_conv1_drop, size=4)
    print( "after pooling:capa_conv1_drop: ",capa_conv1_drop.get_shape(),"\n***********\n")


    if(fourLayers):
        capa_conv2_drop = doPool(capa_conv2_drop, size=4)
    else:
        capa_conv2_drop = doPool(capa_conv2_drop, size=2)
    print("after pooling:capa_conv2_drop: ",capa_conv2_drop.get_shape(),"\n***********\n")

    if(fourLayers):
        capa_conv3_drop = doPool(capa_conv3_drop, size=2)
        print("after pooling:capa_conv3_drop: ",capa_conv3_drop.get_shape(),"\n***********\n")
        layer_flat4, num_fc_layers4 = flatten_layer(capa_conv4_drop)

    layer_flat1, num_fc_layers1 = flatten_layer(capa_conv1_drop)
    layer_flat2, num_fc_layers2 = flatten_layer(capa_conv2_drop)
    layer_flat3, num_fc_layers3 = flatten_layer(capa_conv3_drop)

    if(fourLayers):
        print("NUM FC LAYER:",num_fc_layers1,num_fc_layers2,num_fc_layers3, num_fc_layers4 )
        layer_flat = tf.concat([layer_flat1, layer_flat2, layer_flat3, layer_flat4],1)  #(?, NUm)
        num_fc_layers = num_fc_layers1 + num_fc_layers2 + num_fc_layers3+num_fc_layers4  # (NUm)
    else:
        print("NUM FC LAYER:",num_fc_layers1,num_fc_layers2,num_fc_layers3 )
        layer_flat = tf.concat([layer_flat1, layer_flat2, layer_flat3],1)  #(?, NUm)
        num_fc_layers = num_fc_layers1 + num_fc_layers2 + num_fc_layers3  # (NUm)
    print( "1:",layer_flat.get_shape())
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
        #softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #    logits=capa_fc2, labels=y_deseada)
        softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=capa_fc2, labels=y_deseada)

        cross_entropy_mean = tf.reduce_mean(softmax_cross_entropy, name="error")
        tf.summary.scalar('cross_entropy', cross_entropy_mean)
    if(L2_REG):
        with tf.name_scope("L2_Regularization_Method"):
            regularizers = (tf.nn.l2_loss(pesos_conv1) + tf.nn.l2_loss(pesos_conv2) +
                            tf.nn.l2_loss(pesos_conv3) + tf.nn.l2_loss(pesos_fc1) +
                            tf.nn.l2_loss(pesos_fc2))
            if(fourLayers):
                regularizers = regularizers + tf.nn.l2_loss(pesos_conv4)
            error =  cross_entropy_mean + L2_lambda * regularizers
            tf.summary.scalar('error_Acc', error)
    else:
        error = cross_entropy_mean
    #---------------------------------------------------------------------------

    with tf.name_scope("Entrenamiento"):
        #Funcion de optimizacion
        iterac_entren = tf.Variable(0, name='iterac_entren', trainable=False)
        #A las 15 iteraciones se cambio el (iterperepoca *5) -> (iterperepoca)
        if LEARNING_DECAY:
            lr = tf.train.exponential_decay(TASA_APRENDIZAJE, iterac_entren, ITER_PER_EPOCA, DECAY_RATE, staircase=True, name='ExponentialDecay')
            tf.summary.scalar('learning_rate_decay',lr)
        else:
            lr = TASA_APRENDIZAJE
            tf.summary.scalar('learning_rate',lr)
        #optimizador = tf.train.AdamOptimizer(TASA_APRENDIZAJE).minimize(error, global_step=iterac_entren)
        optimizador = tf.train.AdamOptimizer(lr).minimize(error, global_step=iterac_entren)

    with tf.name_scope("Validacion"):
        # evaluacion
        prediccion_correcta = tf.equal(
            tf.argmax(y_calculada, 1), tf.argmax(y_deseada, 1))
        acierto = tf.reduce_mean(tf.cast(prediccion_correcta, 'float'))
        tf.add_to_collection("mean_acc", acierto)
        tf.summary.scalar("acierto", acierto)

    resumen = tf.summary.merge_all()
    if(fourLayers):
        tam_filtro = [tam_filtro1,tam_filtro2,tam_filtro3,tam_filtro4]
        num_filtro = [num_filtro1,num_filtro2,num_filtro3,num_filtro4]
        dropout_conv = [dropout_conv1,dropout_conv2,dropout_conv3,dropout_conv4]
        internal_layers = [capa_conv1, capa_conv2,capa_conv3,capa_conv4, nopool1, nopool2, nopool3,nopool4,capa_fc1,capa_fc2]
    else:
        tam_filtro = [tam_filtro1,tam_filtro2,tam_filtro3]
        num_filtro = [num_filtro1,num_filtro2,num_filtro3]
        dropout_conv = [dropout_conv1,dropout_conv2,dropout_conv3]
        internal_layers = [capa_conv1, capa_conv2,capa_conv3, nopool1, nopool2, nopool3,capa_fc1,capa_fc2]

    printArquitecture(tam_filtro,num_filtro,dropout_conv,fc1_inputs,fc2_inputs,dropout_fc1)


    return resumen, entradas, y_deseada, is_training, iterac_entren, optimizador, acierto, error, predictor,internal_layers

#===============================================================================================
#===============================================================================================

def print_write_validationSet(i, sess, resumen, acierto, feed_dictx, evalua_writer):
    [res, aciertos_eval] = sess.run([resumen,acierto], feed_dict=feed_dictx)
    evalua_writer.add_summary(res, i)
    print('En la iteracion %d:'%(i+1))
    print(('Acc. Valid Set => %.4f \n' %(aciertos_eval)))

def print_write_trainSet(i, acierto, error ):
    #[aciertos_train, er] = sess.run([ acierto, error], feed_dict=feed_dictx)
    print('En la iteracion %d:'%(i+1))
    print(('Acc. Train Set => %.4f ' % (acierto)))
    print(('Error Train Set => %.4f \n' % (error)))

def saveModel(sess, i , rutaDeModelo, saver):
    print(('Guardando modelo en %d iteraciones....' % (i + 1)))
    saver.save(
        sess,
        rutaDeModelo + NOMBRE_MODELO,
        global_step=i + 1,
        write_meta_graph=True)

def showLayersForImage(imageConvol, internal_layers,entradas):
    #cv2.imwrite('zexample1.png',X_train[200])

    #imageConvol = (imageConvol * 255.).astype(np.float32)
    #imageConvol = 0.299 * imageConvol[ :, :, 0] + 0.587 * imageConvol[ :, :, 1] + 0.114 * imageConvol[ :, :, 2]
    #imageConvol = imageConvol.reshape(imageConvol.shape + (1, ))
    print( imageConvol.shape)
    display(imageConvol,True,64)

    feed_dictcon = {entradas: [imageConvol], is_training: False}
    #for i in range(6,7):
    #    valuesConv = sess.run(internal_layers[i], feed_dict=feed_dictcon)
    #    plot_conv_layer(valuesConv)


if __name__ == "__main__":

    train_split_balanced = '../signals_database/peru-signs-data/pickleFiles/train5_split_50.p'
    validation_split_balanced = '../signals_database/peru-signs-data/pickleFiles/validation5_split_50.p'
    signnames = read_csv(
        "../signals_database/traffic-signs-data/signnames.csv").values[:, 1]

    print("reading trianing dataset")
    X_train, y_train = readData(train_split_balanced)
    X_validation, y_validation = readData(validation_split_balanced)

    #print( X_train[0])
    NUM_TRAIN = y_train.shape[0]
    NUM_VALID = y_validation.shape[0]
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

    #--------------------------------CREACION DE LA RED------------------------------------------------
    print("Inicio de creacion de la red")
    tf.reset_default_graph()
    sess = tf.Session()
    resumen, entradas, y_deseada, is_training, iterac_entren, optimizador, acierto, error, predictor, internal_layers = create_cnn(
    )

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=None)
    ckpt = tf.train.get_checkpoint_state(rutaDeModelo + '.')

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(("Sesion restaurada de: %s" % ckpt.model_checkpoint_path))

    else:
        print(("No se encontro puntos de control."))

    ultima_iteracion = iterac_entren.eval(sess)
    print("Ultimo modelo en la iteracion: ", ultima_iteracion)

    #showLayersForImage(imagenes_entrenam[20], internal_layers, entradas)
    #"""
    entren_writer = tf.summary.FileWriter(rutaDeModelo + 'entrenamiento',
                                          sess.graph)
    #entren_writer.add_graph(sess.graph)
    evalua_writer = tf.summary.FileWriter(rutaDeModelo + 'validacion')#you don want the the graph
    epocas_completadas = 0
    indice_en_epoca = 0

    clases_calc = np.zeros(shape=NUM_VALID, dtype=np.int)

    comienzo_time = time.time()

    avg_loss = avg_acc = 0.
    #Desde la ultima iteracion hasta el ITERACIONES_ENTRENAMIENTO dado
    for i in trange(ultima_iteracion, ITERACIONES_ENTRENAMIENTO):
        #Obtener nuevo subconjunto(batch) de (BATCH_SIZE =100) imagenes
        batch_img_entrada, batch_img_clase = siguiente_batch_entren(
            BATCH_SIZE, NUM_TRAIN)
        # Entrenar el batch
        [resu, _, acc, err] = sess.run(
            [resumen, optimizador, acierto, error],
            feed_dict={
                entradas: batch_img_entrada,
                y_deseada: batch_img_clase,
                is_training: True
            })

        entren_writer.add_summary(resu, i)
        avg_loss += err
        avg_acc += acc

        # Observar el progreso cada 'CHKP_REVISAR_PROGRESO' iteraciones
        if (i + 1) % CHKP_REVISAR_PROGRESO == 0:
            imagenes_eval, clases_eval = shuffle(imagenes_eval, clases_eval)
            print(
                ('Time in %d iteraciones: %s ' %
                 (i + 1 - ultima_iteracion,
                  str( timedelta( seconds=int(round(time.time() - comienzo_time)))))))
            #--------------------------------------------------------------
            avg_loss /= CHKP_REVISAR_PROGRESO
            avg_acc /= CHKP_REVISAR_PROGRESO
            print_write_trainSet(i, avg_acc , avg_loss)
            avg_loss = avg_acc = 0.
            #--------------------------------------------------------------
            feed_dictx = {
                entradas: imagenes_eval[:3131],
                y_deseada: clases_eval[:3131],
                is_training: False#dont use dropout in Testing
            }
            print_write_validationSet(i, sess,resumen, acierto, feed_dictx, evalua_writer)
            #--------------------------------------------------------------

        #Crear 'punto de control' cuando se llego a las CHKP_GUARDAR_MODELO iteraciones
        if (i + 1) % CHKP_GUARDAR_MODELO == 0:
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
