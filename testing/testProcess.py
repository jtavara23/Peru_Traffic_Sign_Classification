#-------------------------------------
# Test File for German and Peru Images
"""
To execute run:
    python testProcess.py [peru/german/german-ext] [type of Model] [numb Of Model] [show_confMatrix] [show_plots]
    Examples:
       > python testProcess.py peru modelA 7700 false false
       > python testProcess.py german model7 38700 false true

For tensorboard run:
    > tensorboard --logdir=modelsName/modelA/ --host=127.0.0.1 --port 6006
"""
# Depening on run command it test:
# test_file = '../signals_database/traffic-signs-data/test_2Processed.p'
# test_file = '../signals_database/traffic-signs-data/test_5ExtendedProcessed.p
# test_file = '../signals_database/peru-signs-data/pickleFiles/test_1Processed.p' from : (validation_test_5_split_50)
#-------------------------------------

import numpy as np
import pandas as pd
import pickle
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
import tensorflow as tf
from util.common_functions import readData, plot_example_errors, plot_confusion_matrix_Large, plot_roc,plot_pr_curve
import math
import os
import datetime
import sys
#from keras.backend import manual_variable_initialization as ke

modelo = ""
model_number = ""


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#np.set_printoptions(threshold=np.nan)

NUM_TEST = 0
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'


def procesamiento(X, y,typeOfSignal):
    """
	Preprocess image data, and convert labels into one-hot
	Arguments:
	    * X: Array of images, should be height,width,1 of shape and each pixel between [0,1]
	    * y: Array of labels
	Returns:
	    * Preprocessed X, one-hot version of y
	"""
    # Convert from RGB to grayscale
    #X = rgb_to_gray(X)

    # Make all image array values fall within the range -1 to 1
    # Note all values in original images are between 0 and 255, as uint8
    #X = X.astype('float32')
    #X = X /255.0
    #X = (X-128)/128.0

    #Organizar las clases de las imagenes en un solo vector
    y_flatten = y
    # convertir tipo de clases de escalares a vectores de activacion de 1s
    # 0 => [1 0 0 0 0 0 0 0 0 0....0 0 0 0]
    # 1 => [0 1 0 0 0 0 0 0 0 0....0 0 0 0]
    # ...
    # N => [0 0 0 0 0 0 0 0 0 0....0 0 0 1]
    if(typeOfSignal == "peru"):
        y_onehot = np.zeros((y.shape[0], 7))
    else:
        y_onehot = np.zeros((y.shape[0], 43))

    for i, onehot_label in enumerate(y_onehot):
        onehot_label[y[i]] = 1.0
    y = y_onehot

    return X, y, y_flatten


#63150
def writeResults(msg, test_file):
    outFile = open(rutaDeModelo+"logtestResult.log", "a")
    outFile.write(repr(rutaDeModelo + "model-"+model_number) + "\n")
    outFile.write(test_file + "\n")
    outFile.write(msg)
    outFile.write("\n" + str(datetime.date.today()))
    outFile.write(
        "\n-------------------------------------------------------------\n")


if __name__ == "__main__":
    typeOfSignal = str(sys.argv[1])
    modelo = str(sys.argv[2])
    model_number = str(sys.argv[3])
    show_confMatrix = str(sys.argv[4]).lower()
    show_plots = str(sys.argv[5]).lower()

    BATCH_SIZE = 2000
    num_classes = 43

    #For GERMAN FILES [test_5ExtendedProcessed] or [test_2Processed]
    if(typeOfSignal == "german"):
        rutaDeModelo = 'models_Alemania/'+modelo+'/'
        test_file = '../signals_database/traffic-signs-data/test_2Processed.p'

    elif(typeOfSignal == "peru"):
        num_classes = 7
        rutaDeModelo = 'models_Peru/'+modelo+'/'
        test_file = '../signals_database/peru-signs-data/pickleFiles/test_1Processed.p'#validation_test_5_split_50

    else:#german-ext
        rutaDeModelo = 'models10extend/'+modelo+'/'
        test_file = '../signals_database/traffic-signs-data/test_5ExtendedProcessed.p'


    X_test, y_test = readData(test_file)
    NUM_TEST = y_test.shape[0]

    imagenes_eval, clases_eval, clases_eval_flat = procesamiento(
        X_test, y_test,typeOfSignal)

    #print clases_eval_flat

    #sess = tf.Session()

    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))#"Whether to log device placement

    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph(rutaDeModelo + "model-"+model_number+".meta")
    saver.restore(sess, rutaDeModelo + "model-"+model_number)

    print(modelo +" - "+model_number+" restaurado ",
            rutaDeModelo + "model-"+model_number)

    #Tensor predictor para clasificar la imagen
    predictor = tf.get_collection("predictor")[0]
    #cantidad de imagenes a clasificar
    cant_evaluar = (imagenes_eval.shape[0])
    print("Cantidad a evaluar: ", cant_evaluar)
    clases_pred = np.zeros(shape=cant_evaluar, dtype=np.int)
    #"""
    start = 0
    aux = 0
    print("Prediciendo clases...")
    while start < cant_evaluar:
        end = min(start + BATCH_SIZE, cant_evaluar)
        print(end)

        images_evaluar = imagenes_eval[start:end, :]
        clases_evaluar = clases_eval[start:end, :]

        #Introduce los datos para ser usados en un tensor
        #feed_dictx = {NOMBRE_TENSOR_ENTRADA+":0": images_evaluar, NOMBRE_TENSOR_SALIDA_DESEADA+":0": clases_evaluar,NOMBRE_PROBABILIDAD+":0":1.0}
        feed_dictx = {
            NOMBRE_TENSOR_ENTRADA + ":0": images_evaluar,
            NOMBRE_TENSOR_SALIDA_DESEADA + ":0": clases_evaluar,
            NOMBRE_PROBABILIDAD + ":0": False
        }

        # Calcula la clase predecida , atraves del tensor predictor
        clases_pred[start:end] = sess.run(predictor, feed_dict=feed_dictx)

        # Asigna el indice final del batch actual
        # como comienzo para el siguiente batch
        aux = start
        start = end

    #print clases_pred[aux:end]
    # Convenience variable for the true class-numbers of the test-set.
    clases_deseadas = clases_eval_flat

    # Cree una matriz booleana
    correct = (clases_deseadas == clases_pred)

    # Se calcula el numero de imagenes correctamente clasificadas.
    correct_sum = correct.sum()

    # La precision de la clasificacion es el numero de imgs clasificadas correctamente
    acc = float(correct_sum) / cant_evaluar

    msg = "Acierto en el conjunto de Testing: {0:.2%} ({1} / {2})"
    print(msg.format(acc, correct_sum, cant_evaluar))
    writeResults(msg.format(acc, correct_sum, cant_evaluar), test_file)
    if(show_confMatrix == "true"):
        print("Mostrando Matriz de Confusion")
        plot_confusion_matrix_Large(clases_pred, clases_deseadas,num_classes)

    plot_roc(clases_pred, clases_deseadas,num_classes)
    if(show_plots == "true"):
        plt.show()
    plot_pr_curve(clases_pred, clases_deseadas,num_classes)
    if(show_plots == "true"):
        plt.show()
    # Muestra algunas imagenes que no fueron clasificadas correctamente
    #plot_example_errors(cls_pred=clases_pred, correct=correct,images = imagenes_eval, labels_flat=clases_eval_flat)

    print("Fin de evaluacion")