import numpy as np
import pandas as pd
import pickle
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
import tensorflow as tf
from funcionesAuxiliares import readData, plot_example_errors, plot_confusion_matrix
import math
import os
import datetime
#from keras.backend import manual_variable_initialization as ke

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(threshold=np.nan)

BATCH_SIZE = 400  #421
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'
rutaDeModelo = 'D:/signalsWindows/models5/'
NUM_TEST = 0


def procesamiento(X, y, type):
    """
	Preprocess image data, and convert labels into one-hot
	Arguments:
	    * X: Array of images
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
    # 43 => [0 0 0 0 0 0 0 0 0 0....0 0 0 1]
    y_onehot = np.zeros((y.shape[0], 43))
    for i, onehot_label in enumerate(y_onehot):
        onehot_label[y[i]] = 1.0
    y = y_onehot

    #if type:
    #perm = np.arange(NUM_TRAIN)
    #else:
    #perm = np.arange(NUM_TEST)
    #np.random.shuffle(perm)
    #perm = stratified_shuffle(clases_entrenam_flat, 10)
    #X = X[perm]
    #y = y[perm]
    #y_flatten = y_flatten[perm]

    return X, y, y_flatten


#63150
def writeResults(msg):
    outFile = open("TestExtendedResults.txt", "a")
    outFile.write(repr(tf.train.latest_checkpoint(rutaDeModelo + '.')) + "\n")
    outFile.write(msg)
    outFile.write("\n" + str(datetime.date.today()))
    outFile.write(
        "\n-------------------------------------------------------------\n")


if __name__ == "__main__":
    test_file = '../signals_database/traffic-signs-data/testExtendedProcessed.p'
    X_test, y_test = readData(test_file)
    NUM_TEST = y_test.shape[0]

    imagenes_eval, clases_eval, clases_eval_flat = procesamiento(
        X_test, y_test, 0)

    #print clases_eval_flat

    # Restauramos el ultimo punto de control
    #tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(rutaDeModelo + 'model-49296.meta')
        saver.restore(sess, tf.train.latest_checkpoint(rutaDeModelo + '.'))
        print("Modelo restaurado",
              tf.train.latest_checkpoint(rutaDeModelo + '.'))

        #Tensor predictor para clasificar la imagen
        predictor = tf.get_collection("predictor")[0]
        #cantidad de imagenes a clasificar
        cant_evaluar = (imagenes_eval.shape[0])

        clases_pred = np.zeros(shape=cant_evaluar, dtype=np.int)

        #"""
        start = 0
        aux = 0
        print("Prediciendo clases...")
        while start < cant_evaluar:
            end = min(start + BATCH_SIZE, cant_evaluar)

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

        # Muestra algunas imagenes que no fueron clasificadas correctamente
        #plot_example_errors(cls_pred=clases_pred, correct=correct,images = imagenes_eval, labels_flat=clases_eval_flat)
        print("Mostrando Matriz de Confusion")
        #plot_confusion_matrix(clases_pred, clases_deseadas,43)
        #plt.show()

        writeResults(msg.format(acc, correct_sum, cant_evaluar))
        #"""
        print("Fin de evaluacion")