import numpy as np
import pandas as pd
import cv2
import pickle
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
import tensorflow as tf
from funcionesAuxiliares import readData, plot_example_errors, plot_confusion_matrix
import math
import os
import sys
#from keras.backend import manual_variable_initialization as ke

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# np.set_printoptions(threshold=np.nan)

BATCH_SIZE = 421  # 421
NOMBRE_TENSOR_ENTRADA = 'inputX'
NOMBRE_TENSOR_SALIDA_DESEADA = "outputYDeseada"
NOMBRE_PROBABILIDAD = 'mantener_probabilidad'
MODEL_PATH = 'D:/signalsWindows/models5/'
# important path to extract processed images
PROCESSED_IMAGES_PATH = "D:/SignalsWindows/imagenes/"
NUM_TEST = 0


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

    img = cv2.resize(img, (32, 32))
    cv2.imwrite(PROCESSED_IMAGES_PATH + 'a.jpg', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(PROCESSED_IMAGES_PATH + 'b.jpg', img)

    img2 = img
    img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    cv2.imwrite(PROCESSED_IMAGES_PATH + 'c.jpg', img)
    
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(PROCESSED_IMAGES_PATH + 'd.jpg', img)

    img2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,1,5,3)
    cv2.imwrite(PROCESSED_IMAGES_PATH + 'e.jpg',img2) 

    np.reshape(img, (1, 32, 32))
    img = (img / 255.).astype(np.float32)
    return img

def getSignalName(index):
    outData = []
    inputFile = open("imagenes/signnames.csv", "r")
    for line in inputFile.readlines():
        data = [(x) for x in line.strip().split(",") if x != '']
        outData.append(data[1])
    #print('\n'.join(outData[]))
    return outData[index]
    

def runAnalyzer(pathImage):
    print("Running session")
    with tf.Session() as sess:
        # Restore latest checkpoint
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(MODEL_PATH + 'model-49296.meta')
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
        results = [0] * 43
        feed_dictx = {
            NOMBRE_TENSOR_ENTRADA + ":0": img,
            NOMBRE_PROBABILIDAD + ":0": False
        }

        # Calcula la clase usando el predictor de nuestro modelo
        label_pred = sess.run(predictor, feed_dict=feed_dictx)
        print("Signal: ", label_pred)
        # cualquiera es igual
        #acc= probabilidad.eval(feed_dict = feed_dictx)
        acc = sess.run(probabilidad, feed_dict=feed_dictx)
        print("Accuracy: ", acc[0][label_pred])
        #print(acc) other accuracies
        return getSignalName(np.asscalar(label_pred))


# if __name__ == "__main__":
#pathImage = (sys.argv[1])
# runAnalyzer(pathImage)
