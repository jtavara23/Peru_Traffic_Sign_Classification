"""
-------------------------------------
 Core file to recognize Signal Traffic Image
 Main function(runAnalyzer) gets call from "Interfaz.py"
 Works for Peru and German Images
-------------------------------------
"""
import numpy as np
import pandas as pd
import cv2
import pickle
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
import tensorflow as tf
from util.common_functions import readData, plot_example_errors, plot_confusion_matrix
import math
import os
import sys
from skimage import exposure
#from keras.backend import manual_variable_initialization as ke

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# np.set_printoptions(threshold=np.nan)

NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'
# important path to extract processed images
PROCESSED_IMAGES_PATH = "imagenes/"
#----------------------------
CLASSES = 0
RESIZE = 0
MODEL_PATH = ""
LAST_MODEL_NAME = ""
TRESHOLD = 0.6
#----------------------------


def apply_configurations(modelType):
    global RESIZE
    global MODEL_PATH
    global LAST_MODEL_NAME
    global CLASSES
    if(modelType == "Peru"):
        print("Analysing Peruvian signal")
        MODEL_PATH = 'models_Peru/modelE/'
        LAST_MODEL_NAME = 'model-7700.meta'
        RESIZE = 60
        CLASSES = 7
    else:
        print("Analysing German signal")
        MODEL_PATH = 'models10extend/model1/'
        LAST_MODEL_NAME = 'model-73340.meta'
        #MODEL_PATH = 'modelsBalanced/modelE/'
        #LAST_MODEL_NAME = 'model-38700.meta'
        RESIZE = 32
        CLASSES = 43


def read_image(imagen):

    data = []
    yy, xx = imagen.shape[:2]

    for x in xrange(0, xx):
        for y in xrange(0, yy):
            data.append(imagen[x, y])
    #outFile.write(repr(data)+ "\n")
    return np.matrix(data)


def procesarIMG(name):
    img = cv2.imread(name)

    img = cv2.resize(img, (RESIZE, RESIZE))
    cv2.imwrite(PROCESSED_IMAGES_PATH + 'a.jpg', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(PROCESSED_IMAGES_PATH + 'b.jpg', img)

    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #cv2.imwrite(PROCESSED_IMAGES_PATH + 'd.jpg', img)

    img = exposure.equalize_adapthist(img)
    img2 = (img * 255).astype(np.int)
    cv2.imwrite(PROCESSED_IMAGES_PATH + 'd.jpg',img2)

    img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    img2 = 0.299 * img2[:, :, 0] + 0.587 * img2[:, :, 1] + 0.114 * img2[:, :, 2]
    cv2.imwrite(PROCESSED_IMAGES_PATH + 'e.jpg', img2)

    np.reshape(img, (1, RESIZE, RESIZE))
    #img = (img / 255.).astype(np.float32) done in CLAHE
    return img

def getSignalName(index):
    outData = []
    global CLASSES
    if(CLASSES == 43):
        inputFile = open("imagenes/signnames.csv", "r")
    else:
        inputFile = open("imagenes/senalnames.csv", "r")
    for line in inputFile.readlines():
        data = [(x) for x in line.strip().split(",") if x != '']
        outData.append(data[1])
    #print('\n'.join(outData[]))
    return outData[index]


def runAnalyzer(pathImage, modelType):
    print("Session iniciada")
    tf.reset_default_graph()
    apply_configurations(modelType)
    with tf.Session() as sess:
        # Restore latest checkpoint
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(MODEL_PATH + LAST_MODEL_NAME)
        ult_pto_control = tf.train.latest_checkpoint(MODEL_PATH + '.')
        saver.restore(sess, ult_pto_control)
        print("Modelo restaurado " + ult_pto_control)

        predictor = tf.get_collection("predictor")[0]
        probabilidad = tf.get_collection("acuracia")[0]

        img = procesarIMG(pathImage)
        # np.set_printoptions(threshold=np.nan)

        print(img.shape)
        img = img[np.newaxis, :, :, np.newaxis]
        print(img.shape)
        # print img
        # """
        # ------------FIN DE PROCESAMIENTO DE IMAGEN------------
        results = [0] * CLASSES
        feed_dictx = {
            NOMBRE_TENSOR_ENTRADA + ":0": img,
            NOMBRE_PROBABILIDAD + ":0": False
        }

        # Calcula la clase usando el predictor de nuestro modelo
        label_pred = sess.run(predictor, feed_dict=feed_dictx)
        print("Señal: ", label_pred)

        # cualquiera es igual
        acc= probabilidad.eval(feed_dict = feed_dictx)
        #acc = sess.run(probabilidad, feed_dict=feed_dictx)
        percentageAcc = float(acc[0][label_pred])
        print("Accuracy: ", str(percentageAcc))
        results = [(xx,acc[0][xx]) for xx in range(0,CLASSES)]
        #for  xx in range(0,43):
        #    results[xx] = (xx,acc[0][xx])
        #    #print("acc["+str(xx)+"]: ",acc[0][xx])
        results = list(reversed(sorted(results, key=lambda tup:tup[1])))
        print(results[:5])
        if(percentageAcc > TRESHOLD):
            resultado = getSignalName(np.asscalar(label_pred)) + " [{0:0.2f}%]".format(percentageAcc*100.0)
            print(resultado)
            return resultado
        else:
            print("Señal de Tránsito Desconocida")
            return "No Reconocida. [0%]"


# if __name__ == "__main__":
#pathImage = (sys.argv[1])
# runAnalyzer(pathImage)
