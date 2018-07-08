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
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#np.set_printoptions(threshold=np.nan)

NOMBRE_MODELO = 'model'
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'
NOMBRE_TENSOR_SALIDA_CALCULADA = 'outputYCalculada'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"

#rutaDeModelo = '/media/josuetavara/Gaston/signals/models5/'
rutaDeModelo = 'D:/signals/models5/'
#tensorboard --logdir /media/josuetavara/Gaston/signals/models4

TASA_APRENDIZAJE = 0.0001  #1e-3

# entrenamiento es ejecutado seleccionando subconjuntos de imagenes
BATCH_SIZE = 256
#256*1226 iteraciones=313672 .. 1 epoca(36:30 min)

ITERACIONES_ENTRENAMIENTO = 1226 * 6 + 1226 * 3 + 1226 * 6 + 1226 * 10 + 1226 * 5 + 1226 * 10 + 256

CHKP_GUARDAR_MODELO = 500  #5*300#cada 300 iteraciones
CHKP_REVISAR_PROGRESO = 32  #500#50#iteraciones
#250==50 veces

DROPOUT = 0.5

#fileI = open("bef.txt", "w")
#fileO = open("aft.txt", "w")
NUM_CLASSES = 0  #43
NUM_TRAIN = 0  #313672
NUM_TEST = 0  #12630
IMAGE_SHAPE = 0  #(32,32,1)


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
    #print "Layer shape: ", layer_shape
    num_features = layer_shape[1:4].num_elements()
    #print "num_features: ", num_features
    layer_flat = tf.reshape(layer, [-1, num_features])
    #print "layer_flat: : ", layer_flat

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
            convolucion = tf.nn.max_pool(
                value=convolucion,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME')
        #print nombre,": ",convolucion.get_shape(),"\n***********"
    return convolucion, pesos


def doPool(convolucion, size):
    return tf.nn.max_pool(
        value=convolucion,
        ksize=[1, size, size, 1],
        strides=[1, size, size, 1],
        padding='SAME')


def capa_fc(nombre, entrada, num_inputs, num_outputs, use_relu=True):
    with tf.name_scope(nombre):
        #print entrada.get_shape()," multiplica ",num_inputs,num_outputs
        pesos = inicializar_pesos(shape=[num_inputs, num_outputs])
        biases = inicializar_bias([num_outputs])
        layer = tf.matmul(entrada, pesos) + biases

        if use_relu:
            "Uso relu"
            layer = tf.nn.relu(layer)
        else:
            "No uso Relu"

    return layer, pesos


def create_cnn():
    #Los tensores placeholder sirven como entrada al grafico computacional de TensorFlow que podemos cambiar cada vez que ejecutamos el grafico.
    #None significa que el tensor puede contener un numero arbitrario de imagenes , donde cada imagen es un vector de longitud dada

    entradas = tf.placeholder(
        'float',
        shape=[None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], IMAGE_SHAPE[2]],
        name=NOMBRE_TENSOR_ENTRADA)
    print "Numero clases", NUM_CLASSES
    # clases
    y_deseada = tf.placeholder(
        'float', shape=[None, NUM_CLASSES], name=NOMBRE_TENSOR_SALIDA_DESEADA)

    #print "entradas shape ",entradas.get_shape() # =>(?, 32, 32, 1)

    #Aplicarmos dropout entre la capa FC y la capa de salida
    #keep_prob = tf.placeholder('float',name=NOMBRE_PROBABILIDAD)
    is_training = tf.placeholder(tf.bool, name=NOMBRE_PROBABILIDAD)

    tam_filtro1 = 5
    num_filtro1 = 32
    capa_conv1, pesos_conv1 = conv_layer(
        nombre="convolucion1",
        entrada=entradas,
        num_inp_channels=IMAGE_SHAPE[2],
        filter_size=tam_filtro1,
        num_filters=num_filtro1,
        use_pooling=True)
    capa_conv1_drop = tf.cond(is_training,
                              lambda: tf.nn.dropout(capa_conv1, keep_prob=0.8),
                              lambda: capa_conv1)

    tam_filtro2 = 5
    num_filtro2 = 64
    capa_conv2, pesos_conv2 = conv_layer(
        nombre="convolucion2",
        entrada=capa_conv1_drop,
        num_inp_channels=num_filtro1,
        filter_size=tam_filtro2,
        num_filters=num_filtro2,
        use_pooling=True)
    capa_conv2_drop = tf.cond(is_training,
                              lambda: tf.nn.dropout(capa_conv2, keep_prob=0.7),
                              lambda: capa_conv2)

    tam_filtro3 = 5
    num_filtro3 = 128
    capa_conv3, pesos_conv3 = conv_layer(
        nombre="convolucion3",
        entrada=capa_conv2_drop,
        num_inp_channels=num_filtro2,
        filter_size=tam_filtro3,
        num_filters=num_filtro3,
        use_pooling=True)
    capa_conv3_drop = tf.cond(is_training,
                              lambda: tf.nn.dropout(capa_conv3, keep_prob=0.6),
                              lambda: capa_conv3)

    #---
    # Fully connected
    """
    # 1st stage output
    pool1 = doPool(capa_conv1_drop, size = 4)
    shape = pool1.get_shape().as_list()
    #print "Shpeee ", shape
    pool1 = tf.reshape(pool1, [-1, shape[1] * shape[2] * shape[3]])
    
    # 2nd stage output
    pool2 = doPool(capa_conv2_drop, size = 2)
    shape = pool2.get_shape().as_list()
    pool2 = tf.reshape(pool2, [-1, shape[1] * shape[2] * shape[3]])    
    
    # 3rd stage output
    shape = capa_conv3_drop.get_shape().as_list()
    pool3 = tf.reshape(capa_conv3_drop, [-1, shape[1] * shape[2] * shape[3]])
    
    layer_flat = tf.concat([pool1, pool2, pool3],1)
    print "1:",layer_flat.get_shape()
    num_fc_layers = layer_flat.get_shape()[1]
    print "2:",num_fc_layers
    """
    #with tf.variable_scope('fc4'):
    #fc4 = fully_connected_relu(layer_flat, size = 1024)
    #fc4 = tf.cond(is_training, lambda: tf.nn.dropout(fc4, keep_prob = 0.5), lambda: fc4)
    #with tf.variable_scope('out'):
    #logits = fully_connected(fc4, size = params.num_classes)

    #"""
    capa_conv1_drop = doPool(capa_conv1_drop, size=4)
    capa_conv2_drop = doPool(capa_conv2_drop, size=2)

    layer_flat1, num_fc_layers1 = flatten_layer(capa_conv1_drop)
    layer_flat2, num_fc_layers2 = flatten_layer(capa_conv2_drop)
    layer_flat3, num_fc_layers3 = flatten_layer(capa_conv3_drop)

    layer_flat = tf.concat([layer_flat1, layer_flat2, layer_flat3],
                           1)  #(?, 7168)
    #print "1:",layer_flat.get_shape()
    num_fc_layers = num_fc_layers1 + num_fc_layers2 + num_fc_layers3  # (7168)
    #print "2:",num_fc_layers
    #"""

    #---------------------------Capa totalmente conectada--------------------------------------------
    #The previous layer.,  Num. inputs from prev. layer. , Num. outputs.
    capa_fc1, pesos_fc1 = capa_fc(
        nombre="FC1",
        entrada=layer_flat,
        num_inputs=num_fc_layers,
        num_outputs=1024,
        use_relu=True)

    fc_capa1_drop = tf.cond(is_training,
                            lambda: tf.nn.dropout(capa_fc1, keep_prob=0.5),
                            lambda: capa_fc1)

    capa_fc2, pesos_fc2 = capa_fc(
        nombre="FC2",
        entrada=fc_capa1_drop,
        num_inputs=1024,
        num_outputs=NUM_CLASSES,
        use_relu=False)
    #y_calculada = capa_fc2
    y_calculada = tf.nn.softmax(capa_fc2, name=NOMBRE_TENSOR_SALIDA_CALCULADA)
    predictor = tf.argmax(y_calculada, dimension=1)
    tf.add_to_collection("predictor", predictor)
    tf.add_to_collection("acuracia", y_calculada)

    regularizers = (tf.nn.l2_loss(pesos_conv1) + tf.nn.l2_loss(pesos_conv2) +
                    tf.nn.l2_loss(pesos_conv3) + tf.nn.l2_loss(pesos_fc1) +
                    tf.nn.l2_loss(pesos_fc2))
    #------------------------------------------------------------

    #el error que queremos minimizar va a estar en funcion de lo calculado con lo deseado(real)
    #error = -tf.reduce_sum(y_deseada * tf.log(y_calculada))
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=capa_fc2, labels=y_deseada)
    error = tf.reduce_mean(
        softmax_cross_entropy, name="error") + 0.0001 * regularizers

    with tf.name_scope("entrenamiento"):
        #Funcion de optimizacion
        iterac_entren = tf.Variable(0, name='iterac_entren', trainable=False)
        #exp_lr = tf.train.exponential_decay(TASA_APRENDIZAJE, iterac_entren, 1000, 0.96, staircase=True, name='expo_rate')
        #optimizador = tf.train.AdamOptimizer(exp_lr).minimize(error, global_step=iterac_entren)
        optimizador = tf.train.AdamOptimizer(TASA_APRENDIZAJE).minimize(
            error, global_step=iterac_entren)

    with tf.name_scope("Acierto"):
        # evaluacion
        prediccion_correcta = tf.equal(
            tf.argmax(y_calculada, 1), tf.argmax(y_deseada, 1))
        #prediccion_correcta = tf.equal(predictor, tf.argmax(y_deseada,1)) Antes de entrenamiento7 estaba esto
        acierto = tf.reduce_mean(tf.cast(prediccion_correcta, 'float'))
        tf.add_to_collection("calculador", acierto)
        tf.summary.scalar("acierto", acierto)

    resumen = tf.summary.merge_all()

    return resumen, entradas, y_deseada, is_training, iterac_entren, optimizador, acierto, predictor


if __name__ == "__main__":

    #train_file = 'traffic-signs-data/trainProcessed.p'
    train_file = '../signals_database/traffic-signs-data/trainProcessedShuffled.p'
    test_file = '../signals_database/traffic-signs-data/testProcessedShuffled.p'
    signnames = read_csv(
        "../signals_database/traffic-signs-data/signnames.csv").values[:, 1]

    print "reading trianing dataset"
    X_train, y_train = readData(train_file)
    print "reading testing dataset"
    X_test, y_test = readData(test_file)

    #print X_train[0]
    NUM_TRAIN = y_train.shape[0]
    NUM_TEST = y_test.shape[0]
    IMAGE_SHAPE = X_train[0].shape
    NUM_CLASSES = len(set(y_train))

    print "Number of training examples =", NUM_TRAIN
    print "Number of testing examples =", NUM_TEST
    print "Image data shape =", IMAGE_SHAPE
    print "Number of classes =", NUM_CLASSES

    class_indices, examples_per_class, class_counts = np.unique(
        y_train, return_index=True, return_counts=True)
    class_indicesTest, examples_per_classTest, class_countsTest = np.unique(
        y_test, return_index=True, return_counts=True)
    #plot_histograms('Class Distribution across Training Data', class_indices, class_counts)
    #plot_histograms('Class Distribution across Testing Data', class_indicesTest, class_countsTest)
    #plt.show()

    imagenes_entrenam, clases_entrenam, clases_entrenam_flat = one_hot(
        X_train, y_train)
    imagenes_eval, clases_eval, clases_eval_flat = one_hot(X_test, y_test)

    #cv2.imwrite('zexample1.png',X_train[200])
    #display(X_train[10])
    #print X_train[10]
    #print "**********************"
    #print X_test[0]

    #--------------------------------CREACION DE LA RED------------------------------------------------
    #"""
    print "Inicio de creacion de la red"
    tf.reset_default_graph()
    sess = tf.Session()
    resumen, entradas, y_deseada, is_training, iterac_entren, optimizador, acierto, predictor = create_cnn(
    )

    sess.run(tf.global_variables_initializer())

    entren_writer = tf.summary.FileWriter(rutaDeModelo + 'entrenamiento7',
                                          sess.graph)
    #entren_writer.add_graph(sess.graph)
    evalua_writer = tf.summary.FileWriter(rutaDeModelo + 'evaluacion7',
                                          sess.graph)
    #numero = 1024

    saver = tf.train.Saver(max_to_keep=None)
    ckpt = tf.train.get_checkpoint_state(rutaDeModelo + '.')

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Sesion restaurada de: %s" % ckpt.model_checkpoint_path)

    else:
        print("No se encontro puntos de control.")

    ultima_iteracion = iterac_entren.eval(sess)
    print "Ultimo modelo en la iteracion: ", ultima_iteracion

    epocas_completadas = 0
    indice_en_epoca = 0

    clases_calc = np.zeros(shape=NUM_TEST, dtype=np.int)

    comienzo_time = time.time()

    #Desde la ultima iteracion hasta el ITERACIONES_ENTRENAMIENTO dado
    for i in range(ultima_iteracion, ITERACIONES_ENTRENAMIENTO):
        #Obtener nuevo subconjunto(batch) de (BATCH_SIZE =100) imagenes
        batch_img_entrada, batch_img_clase = siguiente_batch_entren(
            BATCH_SIZE, NUM_TRAIN)
        # Entrenar el batch
        #[resu, _ ] = sess.run([resumen,optimizador], feed_dict={entradas: batch_img_entrada, y_deseada: batch_img_clase, keep_prob: DROPOUT})
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
            print('Tiempo usado en %d iteraciones: %s ' %
                  (i + 1 - ultima_iteracion,
                   str(
                       timedelta(
                           seconds=int(round(time.time() - comienzo_time))))))
            feed_dictx = {
                entradas: batch_img_entrada,
                y_deseada: batch_img_clase,
                is_training: False
            }
            [resu, aciertos_train] = sess.run(
                [resumen, acierto], feed_dict=feed_dictx)
            print('En la iteracion %d , Acierto de Entrenamiento => %.4f ' %
                  (i + 1, aciertos_train))

            feed_dictx = {
                entradas: imagenes_eval[:2000],
                y_deseada: clases_eval[:2000],
                is_training: False
            }
            [resu, aciertos_eval] = sess.run(
                [resumen, acierto], feed_dict=feed_dictx)
            evalua_writer.add_summary(resu, i)
            print('En la iteracion %d , Acierto de Evaluacion => %.4f \n' %
                  (i + 1, aciertos_eval))

        #Crear 'punto de control' cuando se llego a las CHKP_GUARDAR_MODELO iteraciones
        if (i + 1) % CHKP_GUARDAR_MODELO == 0:
            print('Guardando modelo en %d iteraciones....' % (i + 1))
            saver.save(
                sess,
                rutaDeModelo + NOMBRE_MODELO,
                global_step=i + 1,
                write_meta_graph=True)

    feed_dictx = {
        entradas: batch_img_entrada,
        y_deseada: batch_img_clase,
        is_training: False
    }
    [resu, aciertos_train] = sess.run([resumen, acierto], feed_dict=feed_dictx)
    print('En la iteracion %d , Acierto de Entrenamiento => %.4f ' %
          (i + 1, aciertos_train))
    feed_dictx = {
        entradas: imagenes_eval[:2000],
        y_deseada: clases_eval[:2000],
        is_training: False
    }
    [resu, aciertos_eval] = sess.run([resumen, acierto], feed_dict=feed_dictx)
    evalua_writer.add_summary(resu, i)
    print('En la iteracion %d , Acierto de Evaluacion => %.4f \n' %
          (i + 1, aciertos_eval))
    print('Guardando modelo en %d iteraciones....' % (i + 1))
    saver.save(
        sess,
        rutaDeModelo + NOMBRE_MODELO,
        global_step=i + 1,
        write_meta_graph=True)

    #Fin del proceso de entrenamiento y validacion del modelo
    end_time = time.time()
    # Cuanto tiempo tomo el entrenamiento
    time_dif = end_time - comienzo_time

    # Imprimir tiempo
    print('Tiempo usado en %d iteraciones: %s ' %
          (ITERACIONES_ENTRENAMIENTO - ultima_iteracion,
           str(timedelta(seconds=int(round(time_dif))))))

    #"""